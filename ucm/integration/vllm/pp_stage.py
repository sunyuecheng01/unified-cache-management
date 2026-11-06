"""Stage-local persistent blocks for the ordinary layerwise connector.

Logical request hashes stay unchanged. Only the storage boundary adds the
versioned layout/stage namespace, after the existing TP hash transformation.
This module deliberately has no vLLM or device-runtime dependencies.
"""

import hashlib
import json
import os


def stage_paths(backends, rank):
    roots = backends.split(":") if isinstance(backends, str) else list(backends)
    if not roots or any(not root for root in roots):
        raise ValueError("PP stage storage requires nonempty shared Posix roots")
    return [os.path.join(root, f"pp_{rank}") for root in roots]


def validate_manifest(descriptors, pp_size, tp_size):
    """Return a deterministic manifest; exclude host/device/pointer identity."""
    by_rank = {}
    for descriptor in descriptors:
        rank = (descriptor["pp_rank"], descriptor["tp_rank"])
        if rank in by_rank:
            raise ValueError(f"Duplicate PP/TP layout descriptor: {rank}")
        by_rank[rank] = descriptor
    expected = {(pp, tp) for pp in range(pp_size) for tp in range(tp_size)}
    if set(by_rank) != expected:
        raise ValueError("PP layout handshake must contain every PP/TP worker")
    stages = []
    for pp in range(pp_size):
        layout = by_rank[pp, 0]["layout"]
        if not layout["layer_ids"] or layout["shard_size"] <= 0:
            raise ValueError(f"PP stage {pp} has no KV shards")
        if layout["block_size"] != layout["shard_size"] * len(layout["layer_ids"]):
            raise ValueError(f"Invalid stage-local block geometry for PP {pp}")
        if any(by_rank[pp, tp]["layout"] != layout for tp in range(tp_size)):
            raise ValueError(f"KV layouts differ between TP workers in PP stage {pp}")
        stages.append(layout)
    manifest = {"version": "pp_stage_v1", "tp_size": tp_size, "stages": stages}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(encoded).hexdigest()


def stage_key(key, signature, pp_rank):
    salt = f"ucm:pp_stage_v1:{signature}:{pp_rank}:".encode()
    return hashlib.blake2b(salt + bytes(key), digest_size=16).digest()


def stage_disk_capacity(capacity_gb, manifest, pp_rank):
    """Keep the configured disk budget across stages, weighted by block bytes."""
    capacity = int(capacity_gb)
    if capacity == 0:
        return 0
    sizes = [stage["block_size"] for stage in manifest["stages"]]
    share = capacity * sizes[pp_rank] // sum(sizes)
    if share < 1:
        raise ValueError("posix_capacity_gb is too small for the PP stage layouts")
    return share


class StageWorkerStore:
    """Translate TP-specific keys and global layer IDs at the I/O boundary."""

    def __init__(self, store, signature, pp_rank, layer_ids):
        self.store = store
        self.signature = signature
        self.pp_rank = pp_rank
        self.rows = {layer: row for row, layer in enumerate(layer_ids)}

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _keys(self, keys):
        return [stage_key(key, self.signature, self.pp_rank) for key in keys]

    def load_data(self, block_ids, shard_index, dst_addr, *args, **kwargs):
        return self.store.load_data(
            self._keys(block_ids),
            [self.rows[i] for i in shard_index],
            dst_addr,
            *args,
            **kwargs,
        )

    def dump_data(self, block_ids, shard_index, src_addr, *args, **kwargs):
        return self.store.dump_data(
            self._keys(block_ids),
            [self.rows[i] for i in shard_index],
            src_addr,
            *args,
            **kwargs,
        )

    def lookup(self, block_ids):
        return self.store.lookup(self._keys(block_ids))

    def lookup_on_prefix(self, block_ids):
        return self.store.lookup_on_prefix(self._keys(block_ids))

    def lookup_on_reverse(self, block_ids):
        return self.store.lookup_on_reverse(self._keys(block_ids))

    def prefetch(self, block_ids):
        return self.store.prefetch(self._keys(block_ids))


class StageQueryStore:
    """Scheduler view: a hit requires committed files for every stage/TP key.

    Stores are Posix-only observers. No Cache lookup/prefetch may advertise a
    shard that has not yet been committed as a complete stage-local block.
    """

    def __init__(self, stores, signature, other_rank_hashers):
        if not stores:
            raise ValueError("PP query requires at least one stage")
        self.stores = stores
        self.signature = signature
        self.hashers = [None, *other_rank_hashers]

    def _keys(self, keys, stage, hasher):
        return [
            stage_key(hasher(key) if hasher else key, self.signature, stage)
            for key in keys
        ]

    def lookup_on_prefix(self, block_ids):
        length = len(block_ids)
        for pp, store in enumerate(self.stores):
            for hasher in self.hashers:
                if length == 0:
                    return -1
                length = min(
                    length,
                    store.lookup_on_prefix(self._keys(block_ids[:length], pp, hasher))
                    + 1,
                )
        return length - 1

    def lookup(self, block_ids):
        hits = [True] * len(block_ids)
        for pp, store in enumerate(self.stores):
            for hasher in self.hashers:
                result = store.lookup(self._keys(block_ids, pp, hasher))
                if len(result) != len(hits):
                    raise RuntimeError("Incomplete PP backend lookup result")
                hits = [a and b for a, b in zip(hits, result)]
        return hits

    def lookup_on_reverse(self, block_ids):
        hits = self.lookup(block_ids)
        return next((i for i in range(len(hits) - 1, -1, -1) if hits[i]), -1)

    def prefetch(self, block_ids):
        # Worker-side Cache still loads normally, after committed-only lookup.
        return None
