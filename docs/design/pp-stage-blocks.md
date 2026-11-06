# UCMConnector 按 PP stage 拆分 block 的设计与实现

更新日期：2026-11-06。

本方案已落实到当前工作区源码。只改普通 `UCMConnector → UCMLayerWiseConnector`，不包含 HLA/HMA/FAWA、Lite 或 CP。本文替换此前草案中的待定接口；不再要求 DP=1，不新增根目录 writer 锁，也不扩展 Cache 通用失败处理。

## 1. 要解决的问题

原实现让多个 PP stage 写同一个 UCM block，各自只写自己拥有的 layer。Posix 的完成位图却保存在每个 Store 进程内，并要求收齐全模型的 shard。因此各 stage 都无法独自收满位图，文件可能一直保持 `.tmp`，scheduler 查不到正式文件。

修改后，每个 stage 保存自己的完整 block，位图只覆盖本 stage。scheduler 汇总各 stage 的查询结果，只有全部必要文件存在时才报告命中。

## 2. 范围与前提

- 普通 `UCMLayerWiseConnector` 且 PP > 1 时自动使用新方案，不需要新增 YAML 开关。
- 数据 Store 为 `UcmPipelineStore`，pipeline 为 `Cache|Posix`；其他存储组合在该模式下明确报错。
- PP 可以同机，也可以跨机。
- scheduler 与全部 worker 必须访问同一份共享文件系统；仅路径字符串相同不代表文件系统共享。
- 同一 stage 的 TP worker 必须具有相同缓存布局；不同 stage 可以不同。
- PP=1 及其他连接器保持原路径。本次不增加 HLA/HMA/CP 的 PP 支持。
- DP 的持久化 key、GC owner 和现有多 writer 语义不另行重设计；这不是新增的 DP 多 writer 修复。

## 3. 目录和 key

配置仍填写根目录：

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/kv
      share_buffer_enable: false
      io_direct: false
```

派生路径为：

```text
/mnt/kv/pp_0/<原有 hash 子目录>/<physical_key>
/mnt/kv/pp_1/<原有 hash 子目录>/<physical_key>
```

不增加 layout 目录，也不新增 engine 目录。

请求元数据继续使用逻辑 key；在 Store 边界转换为物理 key：

```text
rank_key = 原有 TP key 变换(logical_key)
physical_key = BLAKE2b-128(
    "ucm:pp_stage_v1:" + layout_signature + ":" + pp_rank + ":" + rank_key
)
```

- MHA/GQA：保留各 TP rank 独立 key；先做原有 TP 变换，再做 stage 变换。
- MLA：继续共用 key；TP rank 只决定负责保存哪些 token block，不进入 MLA 持久化 key。
- 物理 key 固定 16 字节。
- key 不加入 PID、主机名、启动随机数；相同模型和兼容布局重启后可以复用正式文件。
- 新目录和新 key 不读取原来根目录中的 PP 文件。旧文件不删除、不迁移。

布局签名由全部 stage 的描述进行规范化 JSON 编码并计算 SHA-256，包含协议版本、TP 数、各 stage 层列表、tensor 大小、有效槽、dtype/shape、token block 大小以及对齐后的 shard/block 大小。设备地址、物理 KV block 数和主机身份不进入签名。原有请求哈希中的模型及量化兼容信息继续保留。

## 4. 本 stage 的布局

例如：

```text
PP0 实际缓存层：[0, 1]
PP1 实际缓存层：[2, 5, 8]  # 可以包含 MTP 等非连续编号

PP0 shard index：[0, 1]
PP1 shard index：[0, 1, 2]
```

使用显式映射 `global_layer_id → local_row`。保存和加载采用同一映射，下一层预取按实际层列表推进，不再依赖全局编号加一。

大小计算：

```text
local_num_shards = 实际注册的缓存行数
store_shard_size = align_up(local_logical_shard_size, 4096)
store_block_size = local_num_shards × store_shard_size
```

每个 stage 可以拥有不同的 shard 字节数和层数。MTP 只计算实际注册的缓存行，不预留全局 MTP 空行。

共享 indexer 模型中，若某个 stage 没有独立 indexer，只注册 attention tensor，允许该 stage 使用 attention-only 布局。其他 stage 仍按各自实际的 indexer/scale 布局保存。

## 5. 初始化时序

1. worker 注册本 rank 的 KV caches，构造本地布局。
2. 在本 engine 的 TP CPU 通信组内交换描述，再在 PP CPU 通信组内交换 TP 描述列表。
3. 每个 worker 得到全部 PP/TP 描述，校验完整性及同 stage 的 TP 布局一致性，计算统一签名。
4. worker 创建本 stage 的数据 Store。
5. worker 通过 `get_handshake_metadata()` 返回自己的布局描述。
6. vLLM 在 KV 注册结束后收集 `(pp_rank, tp_rank) → descriptor`，传给 scheduler 的 `set_xfer_handshake_metadata_pp_aware()`。
7. scheduler 校验描述并为每个 stage 创建纯 Posix 查询 Store，使用对应 stage 的准确 block 大小。

不使用全局 world 组交换布局，避免其他 DP engine 的重复 PP/TP 编号混入。不新增跨机缓存查询 RPC，也不再依赖单个进程发布一个全局 block 大小的本地文件。

集成版本必须具备上述 PP-aware handshake 调用链。本地 vLLM 与 vLLM-Ascend 源码均已有该调用链。

## 6. Store 与 Cache

每个 worker 仍有一个数据 Store，处理自己的 stage。scheduler 每个 stage 有一个纯 Posix 查询 Store。例如 PP=2、TP=8、DP=1 时有 16 个 worker 数据 Store、2 个 scheduler 查询 Store；pipeline 内部的 Cache/Posix 原生对象另计。

Cache 运行时身份增加 stage 和布局签名。成员只包括“同节点、同 stage”的 worker：

- `local_rank_size` 为该成员集合的大小。
- `cache_rank` 为集合内的连续编号，与实际 `device_id` 分离。
- Cache v2 用该编号选择共享分区和后端加载分工；设备操作仍使用真实 device ID。
- scheduler 不加入 stage Cache 域，因此 stage 模式设置 `cache_observer_count=0`。
- 不设置新参数的已有路径保留原来的 device modulo rank 映射和一个 observer 默认值。

`share_buffer_enable` 保留当前安装的 Cache 实现原有语义，不作为 PP 开关，也不强制关闭 Cache。

容量规则：

- 原有 Cache 容量助手先按现有 DP/私有 worker 规则分配。
- 对共享 Cache 域，再按本机实际 stage 数分配预算，避免同机每个 stage 都申请完整预算。
- legacy 私有 Cache 已按本机 worker 分配，不重复按 stage 除。
- 显式设置正数 `posix_capacity_gb` 时，按各 stage 的物理 block 字节占比分配，向下取整；某 stage 不足 1 GB 时启动报错。
- `posix_capacity_gb=0` 保留原有不设容量上限的语义。
- GC owner 仍按原来的 DP0 scheduler 规则，各 stage 使用准确的 block 大小。

## 7. 保存与提交

保存流程：

```text
逻辑 key → 原有 TP 变换 → stage 物理 key
全局 layer ID → 本地 shard index
当前 stage tensor → Cache → 当前 stage 的 Posix 文件
```

MHA/GQA 保持每个 TP rank 保存自己的文件；MLA 保持按绝对 token block 位置对 TP 取模分工，同一个 block 的所有层归同一个 worker 保存。

Posix 的 psync/AIO 完成位图算法不变。区别是它现在只需要收齐本 stage 的 shard，能够在单个 Store 进程内完成并 rename 成正式文件。

不同 stage 不等待彼此提交，也不共享完成位图。PP0 可以先完成；PP1 未完成时全局 lookup 仍为 miss。

## 8. 查询与加载

scheduler 只通过纯 Posix 查询正式文件，不用 Cache 中某个 shard 的存在代表完整 block。

必要文件集合：

| 模型类型 | 必须存在的物理文件 |
|---|---|
| MHA/GQA | 每个 stage、每个现有 TP rank key 的文件 |
| MLA | 每个 stage 的一份公共 key 文件 |

各必要分片返回连续前缀最后一个命中下标，`-1` 表示没有命中。全局结果取最小值：

```text
PP0 命中前 100 块，PP1 命中前 80 块 → 全局命中前 80 块
```

查询中任何必要分片报错，都不能跳过它报告命中。调用方继续按原有错误策略处理。任意点查询按交集汇总，反向查询取共同命中的最后一块。

TP > KV heads 时仍检查所有实际加载方的 rank key。副本去重必须同时改变 key、dump owner 和 load 路由，本次不单独减少查询 rank 数。

worker 加载仍走 `Cache|Posix`，物理 key 和 shard 映射与保存相同。现有 Cache 的逐 shard Ready 等待保持不变：可以读取已 Ready 的 shard，再从正式文件补齐其余 shard；不能读取未 Ready 的字节。

scheduler 不创建 Cache observer，也不执行 Cache prefetch；worker 在实际 load 时填充 Cache。这会改变预取时机，性能需要实机测量。

## 9. 请求生命周期和一致性

逻辑 key 始终用于请求元数据、缺失记录和原有 RankConsistencyManager。只有提交到 Store 时才转换物理 key；Task 等待和错误回传仍使用原来的生命周期。

本次不增加“跨 stage dump 完成事件”的聚合协议，也不把 D2H 成功当成全局磁盘提交成功。现有成功元数据可以放开一次重新查询；真正报告命中时，StageQueryStore 仍必须确认所有正式文件，因此单个 stage 的成功不会绕过完整性检查。

加载期间发生文件删除、GC 或存储错误时，沿用既有加载失败/重算策略。不另行重设计 DP 多 writer 锁、Cache Ready 超时或通用错误恢复。

## 10. 修改文件

| 文件 | 职责 |
|---|---|
| `ucm/integration/vllm/ucm_connector.py` | 启用条件、stage 布局、启动描述交换、握手转发、Store 参数和本地层遍历 |
| `ucm/integration/vllm/pp_stage.py` | 布局校验、签名、物理 key、worker Store 适配器、scheduler 查询汇总、磁盘容量分配 |
| `ucm/store/cache/v2/global_config.h` | 显式 Cache rank 和 observer 数配置 |
| `ucm/store/cache/v2/cache_buffer.h` | 按域内 rank 选择共享分区 |
| `ucm/store/cache/v2/load_queue.h` | 使用 Cache rank 安排后端加载顺序 |
| `ucm/store/cache/v2/ctrl_strategy.h` | 按实际 observer 数接受共享控制区成员 |
| `test/suites/Unit/test_pp_stage.py` | CPU 布局、key、提交边界和查询回归 |
| `ucm/store/test/case/cache/v2/cache2_config_test.cc` | 原生配置兼容性与 rank 范围验证 |
| `ucm/store/test/case/cache/v2/ctrl_strategy_test.cc` | 无 scheduler observer 时的共享控制区成员验证 |

本次没有修改 `hla_connector.py`、`hma_connector.py`，也没有修改 Posix 位图和 AIO/psync 提交实现。

## 11. 验证与上线验收

CPU 测试覆盖不同 stage 布局、非连续 layer/MTP 行、无独立 indexer 的 stage、物理 key 隔离、缺少 TP/stage 文件时截断前缀、各 stage 独立收满位图、scheduler 使用纯 Posix、Cache 成员和容量隔离。既有 KV 布局、Cache 配置和 block-size 发布回归也需通过。

本机没有 Ascend/CUDA 运行环境，也没有可用的原生 CMake 编译环境。CPU mock/AST 测试不能替代真实多进程 Posix 提交和 KV 字节校验。

本次 CPU 回归结果：224 项通过，21 项因缺少 CMake 跳过，另有 31 项子测试通过；新增 C++ 测试尚未编译运行。Python 编译检查和 `git diff --check` 通过。

可在仓库根目录运行独立 CPU 回归（`--noconftest` 用于跳过全局 GPU 测试环境初始化）：

```bash
python -m pytest --noconftest -o addopts= -q \
  test/test_cache2_integration.py \
  test/suites/Unit/test_kv_cache_layout.py \
  test/suites/Unit/test_pp_stage.py \
  test/suites/Unit/test_block_size_publish.py \
  test/suites/Unit/test_ucm_v41_compat.py \
  test/suites/Unit/test_hma_block_geometry.py
```

部署前重新编译并安装本次 UCM，包括 Cache 原生库；不能只同步 Python 文件。然后按以下步骤验收：

1. 保持当前 `Cache|Posix` YAML 和 PP/TP 启动参数，确认日志出现各 stage 的层列表、布局签名及不同的 block 大小。
2. 完整执行一次足够长的请求并等待异步落盘，确认 `pp_0/`、`pp_1/` 出现正式文件，而不只是 `.tmp`。
3. reset HBM prefix cache，重放完全相同的请求，检查外部命中。
4. 停止服务后重启，再次查询，确认正式文件可复用。
5. 在独立测试目录故意缺少一个 stage 或 TP 文件，确认命中前缀在缺失位置截断。
6. 对保存/加载前后的有效 KV 字节做校验，重点包含末尾 MTP 行、indexer 和 scale；仅比较生成文本不足以确认正确性。
7. 分别验证 PP 跨机共享文件系统、TP=1/TP>1、psync/AIO 以及 Cache 部分命中。
8. 记录 lookup 耗时、首 token 延迟、保存吞吐和实际内存占用。本方案不承诺未经压测的性能数值。

旧数据保留在原目录，新代码不会混读。回滚代码不会删除新数据，但旧代码仍存在原先的 PP 提交问题。
