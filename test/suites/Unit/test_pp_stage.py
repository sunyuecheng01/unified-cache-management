"""CPU-only regression tests for PP stage blocks and committed-prefix lookup."""

import ast
import copy
import hashlib
import importlib.util
import math
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

from test_kv_cache_layout import (
    FakeTensor,
    KVCacheLayout,
    SharedIndexerKVCacheLayout,
    get_store_io_sizes,
)

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "pp_stage", ROOT / "ucm/integration/vllm/pp_stage.py"
)
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


def descriptors(tp_size=1):
    return [
        {
            "pp_rank": stage,
            "tp_rank": tp,
            "host": f"host-{stage}",
            "layout": {
                "layer_ids": layers,
                "shard_size": size,
                "block_size": size * len(layers),
            },
        }
        for stage, layers, size in [(0, [0, 1], 4096), (1, [2, 5, 8], 8192)]
        for tp in range(tp_size)
    ]


class CommitStore:
    """Model Posix's per-process bitmap and committed-only lookup contract."""

    supports_distributed_backend_load = True

    def __init__(self, shards):
        self.shards = shards
        self.pending = {}
        self.committed = set()
        self.last_load = None

    def dump_data(self, keys, shards, ptrs, event=0):
        for key, shard in zip(keys, shards):
            if not 0 <= shard < self.shards:
                raise ValueError("out of range shard")
            bitmap = self.pending.setdefault(key, set())
            bitmap.add(shard)
            if len(bitmap) == self.shards:
                self.committed.add(key)
        return event

    def load_data(self, keys, shards, ptrs, **kwargs):
        self.last_load = (keys, shards, ptrs, kwargs)
        return 42

    def lookup(self, keys):
        return [key in self.committed for key in keys]

    def lookup_on_prefix(self, keys):
        return next(
            (i - 1 for i, key in enumerate(keys) if key not in self.committed),
            len(keys) - 1,
        )


def connector_methods(*names):
    path = ROOT / "ucm/integration/vllm/ucm_connector.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if getattr(n, "name", "") == "UCMLayerWiseConnector"
    )
    methods = [n for n in cls.body if getattr(n, "name", "") in names]
    namespace = dict(
        copy=copy,
        math=math,
        time=time,
        stage_paths=pp.stage_paths,
        validate_manifest=pp.validate_manifest,
        StageQueryStore=pp.StageQueryStore,
        stage_disk_capacity=pp.stage_disk_capacity,
        logger=Mock(),
        uses_partitioned_cache=lambda: True,
    )
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return type("Connector", (), {name: namespace[name] for name in names}), namespace


class PPStageTests(unittest.TestCase):
    def test_worker_collectives_are_tp_then_pp_and_ignore_allocation_size(self):
        cls, ns = connector_methods("_prepare_pp_layout")
        ns.update(
            socket=NS(gethostname=lambda: "worker-host"),
            _get_store_io_sizes=get_store_io_sizes,
            get_tp_group=lambda: NS(rank_in_group=0, cpu_group="tp"),
            get_pp_group=lambda: NS(rank_in_group=0, world_size=2, cpu_group="pp"),
        )
        calls = []

        def gather(output, value, group):
            calls.append(group)
            if group == "tp":
                output[:] = [copy.deepcopy(value) for _ in range(2)]
                output[1]["tp_rank"] = 1
            else:
                output[:] = [copy.deepcopy(value) for _ in range(2)]
                for d in output[1]:
                    d["pp_rank"] = 1
                    d["host"] = "other-host"
                    d["layout"]["layer_ids"] = [2, 80]

        ns["torch"] = NS(distributed=NS(all_gather_object=gather))
        connector = cls()
        connector.layer_ids = [0, 1]
        connector.tp_size = 2
        connector.blocks_per_chunk = 1
        connector.block_size = 128
        connector._vllm_config = NS(parallel_config=NS(pipeline_parallel_size=2))
        signatures = []
        for block_count in [2, 10]:
            tensor = FakeTensor(1234 * block_count, 128, num_blocks=block_count)
            tensor.dtype = "int8"
            connector.kv_caches = {"model.layers.0.self_attn": (tensor,)}
            layout = NS(
                shard_size=128,
                block_size=256,
                tensor_size_lists=np.array([[128], [128]]),
                base_ptrs=np.array([[1000 * block_count], [2000 * block_count]]),
            )
            connector._prepare_pp_layout(layout)
            self.assertEqual(connector._pp_layer_rows, {0: 0, 1: 1})
            self.assertEqual(len(connector._pp_descriptors), 4)
            signatures.append(connector._pp_signature)
        self.assertEqual(calls, ["tp", "pp", "tp", "pp"])
        self.assertEqual(signatures[0], signatures[1])

    def test_layer_load_advances_to_noncontiguous_mtp_row(self):
        cls, _ = connector_methods("wait_for_layer_load")
        connector = cls()
        connector._pp_stage_enabled = True
        connector._connector_metadata = True
        connector.need_load = True
        metadata = NS(request_meta={"request": NS(load_block_ids=([b"key"], [0]))})
        connector._get_connector_metadata = lambda: metadata
        connector.layer_ids = [40, 41, 80]
        connector.layer_name_to_id = {"layer41": 41}
        connector._pp_layer_rows = {40: 0, 41: 1, 80: 2}
        connector.load_tasks = {41: {"request": object()}}
        connector._rank_consistency = NS(wait_load=Mock())
        connector._layerwise_load_bytes = 0
        connector.kv_cache_layout = NS(shard_size=128)
        connector._record_layerwise_load_duration = Mock(return_value=False)
        connector._submit_request_load_tasks_for_layer = Mock()
        connector.wait_for_layer_load("layer41")
        connector._submit_request_load_tasks_for_layer.assert_called_once_with(
            80, 2, metadata
        )

    def test_disk_budget_is_split_without_multiplying_total_capacity(self):
        manifest, _ = pp.validate_manifest(descriptors(), 2, 1)
        self.assertEqual(
            [pp.stage_disk_capacity(100, manifest, i) for i in range(2)], [25, 75]
        )
        self.assertEqual(pp.stage_disk_capacity(0, manifest, 0), 0)
        with self.assertRaises(ValueError):
            pp.stage_disk_capacity(1, manifest, 0)

    def test_manifest_accepts_uneven_stages_and_is_host_independent(self):
        ds = descriptors(2)
        manifest, signature = pp.validate_manifest(ds, 2, 2)
        self.assertEqual([s["block_size"] for s in manifest["stages"]], [8192, 24576])
        moved = copy.deepcopy(ds)
        for d in moved:
            d["host"] = "one-machine"
        self.assertEqual(pp.validate_manifest(moved[::-1], 2, 2)[1], signature)
        for d in moved:
            if d["pp_rank"] == 1:
                d["layout"]["layer_ids"] = [2, 6, 8]
        self.assertNotEqual(pp.validate_manifest(moved, 2, 2)[1], signature)

    def test_incomplete_duplicate_and_incompatible_tp_layouts_rejected(self):
        ds = descriptors(2)
        for invalid in [ds[:-1], ds + [ds[0]]]:
            with self.assertRaises(ValueError):
                pp.validate_manifest(invalid, 2, 2)
        ds[-1]["layout"]["block_size"] += 4096
        with self.assertRaises(ValueError):
            pp.validate_manifest(ds, 2, 2)

    def test_physical_keys_separate_stages_and_layouts(self):
        key = bytes(range(16))
        variants = {
            pp.stage_key(key, signature, stage)
            for signature in ["a", "b"]
            for stage in [0, 1]
        }
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(len(v) == 16 for v in variants))
        self.assertNotIn(key, variants)
        self.assertEqual(
            pp.stage_paths(["/mnt/kv"], 1), [os.path.join("/mnt/kv", "pp_1")]
        )

    def test_each_stage_commits_without_other_stage_shards(self):
        backends = [CommitStore(2), CommitStore(3)]
        workers = [
            pp.StageWorkerStore(backends[i], "sig", i, layers)
            for i, layers in enumerate(([0, 1], [2, 5, 8]))
        ]
        query = pp.StageQueryStore(backends, "sig", [])
        key = b"request-prefix!!"
        workers[0].dump_data([key], [0], [[100]])
        workers[0].dump_data([key], [1], [[200]])
        self.assertEqual(len(backends[0].committed), 1)
        self.assertEqual(query.lookup_on_prefix([key]), -1)
        for layer in [2, 5]:
            workers[1].dump_data([key], [layer], [[300]])
        self.assertEqual(query.lookup_on_prefix([key]), -1)
        self.assertEqual(workers[1].dump_data([key], [8], [[400]], 123), 123)
        self.assertEqual(query.lookup_on_prefix([key]), 0)
        self.assertEqual(backends[1].pending[pp.stage_key(key, "sig", 1)], {0, 1, 2})

    def test_load_uses_same_key_and_dense_row_with_options(self):
        backend = CommitStore(3)
        store = pp.StageWorkerStore(backend, "sig", 1, [2, 5, 8])
        self.assertTrue(store.supports_distributed_backend_load)
        self.assertEqual(
            store.load_data([b"k"], [5], [[456]], distribute_backend_load=True), 42
        )
        self.assertEqual(
            backend.last_load,
            (
                [pp.stage_key(b"k", "sig", 1)],
                [1],
                [[456]],
                {"distribute_backend_load": True},
            ),
        )
        with self.assertRaises(KeyError):
            store.load_data([b"k"], [0], [[456]])

    def test_mha_requires_all_tp_files_and_contiguous_prefix(self):
        hasher = lambda key: hashlib.md5(b"tp1" + key).digest()
        stores = [CommitStore(1), CommitStore(1)]
        keys = [bytes([i]) for i in range(4)]
        for stage, store in enumerate(stores):
            for key in keys:
                store.committed.update(
                    [
                        pp.stage_key(key, "s", stage),
                        pp.stage_key(hasher(key), "s", stage),
                    ]
                )
        query = pp.StageQueryStore(stores, "s", [hasher])
        self.assertEqual(query.lookup_on_prefix(keys), 3)
        stores[1].committed.remove(pp.stage_key(hasher(keys[1]), "s", 1))
        self.assertEqual(query.lookup_on_prefix(keys), 0)
        self.assertEqual(query.lookup(keys), [True, False, True, True])
        self.assertEqual(query.lookup_on_reverse(keys), 3)
        self.assertEqual(query.lookup_on_prefix([]), -1)
        self.assertIsNone(query.prefetch(keys))

    def test_scheduler_builds_posix_observers_with_stage_geometry(self):
        cls, ns = connector_methods("set_xfer_handshake_metadata_pp_aware")
        calls = []

        def factory(name, config):
            calls.append((name, config))
            return CommitStore(config["block_size"] // config["shard_size"])

        ns["UcmConnectorFactoryV1"] = NS(create_connector=factory)
        connector = cls()
        connector._pp_stage_enabled = True
        connector._vllm_config = NS(parallel_config=NS(pipeline_parallel_size=2))
        connector.tp_size = 1
        connector._gc_owner = True
        connector._other_rank_hashers = []
        connector.connector_configs = [
            {
                "ucm_connector_config": {
                    "store_pipeline": "Cache|Posix",
                    "storage_backends": "/mnt/kv",
                }
            }
        ]
        connector.set_xfer_handshake_metadata_pp_aware(
            {(d["pp_rank"], d["tp_rank"]): d for d in descriptors()}
        )
        self.assertEqual([config["block_size"] for _, config in calls], [8192, 24576])
        self.assertTrue(
            all(
                config["store_pipeline"] == "Posix" and config["device_id"] == -1
                for _, config in calls
            )
        )
        self.assertEqual(
            calls[1][1]["storage_backends"], [os.path.join("/mnt/kv", "pp_1")]
        )
        self.assertEqual(
            connector.connector_configs[0]["ucm_connector_config"]["storage_backends"],
            "/mnt/kv",
        )

    def test_cache_domains_and_budget_follow_node_local_stages(self):
        cls, _ = connector_methods("_configure_pp_store")
        connector = cls()
        connector._pp_descriptors = descriptors(2)
        for d in connector._pp_descriptors:
            d["host"] = "same-node"
        connector._pp_descriptor = connector._pp_descriptors[-1]
        connector._pp_rank = 1
        connector.unique_id = "engine"
        connector._pp_signature = "layout"
        config = dict(
            storage_backends=["/mnt/kv"],
            store_pipeline="Cache|Posix",
            cache_buffer_capacity_gb=160,
            device_id=7,
        )
        connector._configure_pp_store(config)
        self.assertEqual((config["local_rank_size"], config["cache_rank"]), (2, 1))
        self.assertEqual(config["device_id"], 7)
        self.assertEqual(config["cache_observer_count"], 0)
        self.assertEqual(config["cache_buffer_capacity_gb"], 80)
        for d in connector._pp_descriptors:
            d["host"] = f"node-{d['pp_rank']}"
        config["cache_buffer_capacity_gb"] = 160
        connector._configure_pp_store(config)
        self.assertEqual(config["cache_buffer_capacity_gb"], 160)

    def test_sparse_layer_ids_and_mtp_count_only_registered_rows(self):
        config = NS(
            parallel_config=NS(pipeline_parallel_size=2),
            model_config=NS(hf_text_config=NS(num_hidden_layers=80)),
        )
        tensors = {
            f"model.layers.{i}.self_attn": (FakeTensor(1000 + i * 100, 128),)
            for i in [80, 41, 40]
        }
        layout = KVCacheLayout(
            tensors, {"_pp_stage_local": True}, config, NS(num_blocks=2)
        )
        self.assertEqual(layout.block_size, 3 * 128)
        self.assertEqual(layout.base_ptrs[:, 0].tolist(), [5000, 5100, 9000])
        self.assertEqual(
            get_store_io_sizes(layout.shard_size, layout.block_size), (4096, 12288)
        )

    def test_shared_indexer_stage_without_independent_indexer(self):
        config = NS(
            parallel_config=NS(pipeline_parallel_size=2),
            model_config=NS(hf_text_config=NS(num_hidden_layers=80)),
            additional_config={
                "enable_sparse_sfa_c8": True,
                "enable_sparse_li_c8": True,
            },
        )
        tensors = {"model.layers.40.self_attn": (FakeTensor(1000, 128),)}
        layout = SharedIndexerKVCacheLayout(
            tensors, {"_pp_stage_local": True}, config, NS(num_blocks=2)
        )
        self.assertEqual(layout.block_size, 128)
        self.assertEqual(layout.tensor_size_list, [128])


if __name__ == "__main__":
    unittest.main()
