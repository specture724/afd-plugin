---
title: "Prefill 桶化 FULL 协同捕获方案（DBO on graph · prefill 篇）交接"
kind: design
status: draft
date: 2026-08-31
branch: dbo_on_graph
---

# Prefill 桶化 FULL 协同捕获方案

**结论先行：可行，且是当前部署下让 DBO 转为正收益的唯一路径。** 把 AFDUBatchWrapper
既有的 FULL 协同捕获从 decode 桶扩展到 prefill token 桶：每个桶一次两线程协同捕获
（含 MoE dispatch/recv op 与 yield 顺序），运行时把 prefill 步补齐到最近桶直接
replay。每步主机成本从 ~1250 次 launch（~13ms）降为一次 `cudaGraphLaunch`
（~10µs），DBO 的主机翻倍成本归零，交叠第一次净赚。**第一个落地物是一个探针**
（§5）：验证 FLASH_ATTN_MLA 的 varlen prefill 在固化 grid 下按桶重放的正确性，
它决定整个方案成立与否。

适用边界：本方案只针对 async GPU connector 的 1A1F/2A1F 部署、OSL=1（纯
prefill）或短 decode 的负载形态；decode 的 FULL_DECODE_ONLY 图与 FFN padded
graph 保持既有机制不变。

## 1. 动机与实测依据

本轮在 L20X（48G）+ DeepSeek-V2-Lite 1A1F 上测了三种负载形态的 DBO 开/关对比
（全部 async connector、attention eager、FFN padded graph、OSL=1、64~150 请求，
trace 在 `/tmp/dbo_prof/`，汇总脚本 `/tmp/dbo_tail_client.py`）：

| 负载形态 | 部署参数 | DBO 关 | DBO 开 | 差异 |
| --- | --- | --- | --- | --- |
| 等长 prefill（~100tok×并发） | batched=512 | 7.01s/150步 | 8.96s/150步 | 慢 28% |
| 重尾 v1（52短/9中/3长，并发 8） | batched=4096 | total 2.16s，p50 244ms | total 2.99s，p50 327ms | 慢 38% |
| 重尾 v2 调优（43短/12中/9长，并发 12） | batched=1024 | total 1.80s，p50 312ms，p90 459ms | total 2.67s，p90 1002ms | 慢 48% |

三个共同的实测事实：

1. **host-bound**：所有 trace 的 GPU busy 只有 4~5%。tail2 每步 ~26ms 中
   GPU ~10ms、主机 ~1250 次 launch 占 ~13ms。
2. **chunked prefill 已把步级 TTFT 做平**：DBO 关时短/中/长三桶延迟几乎相等
   （311/317/315ms）——调度层的公平切片把队头阻塞消掉了，DBO 没有空窗可赚。
3. **DBO 翻倍主机成本**：内核数 +50%（劈半+padding），发射线程从 1 变 2
   （2200 次交替，机制本身完全正常）。

推论：在 eager attention 下 DBO 必然变慢；只有把每步主机成本降为常数，
交叠的收益（重尾下短请求穿流长 FFN 计算）才能净赚——这就是本方案。

## 2. 原理：prefill 入图到底要解决什么

prefill 步的"变长"只有一维是**真的**需要图来管的：

- **步内总 token 数任意**（chunked prefill 封顶于 `max_num_batched_tokens`，
  其余任意）→ 用桶化+补齐解决，与 decode FULL 的 padding 同构。
- **请求间 ragged**（每请求 seq_len / query 边界 / slot_mapping 不同）→
  这些全部是**设备侧张量**（`query_start_loc` / `cu_seqlens` / `seq_lens` /
  `block_table`），varlen 内核从显存读取。图捕获固化的是 **grid 尺寸**，
  只要捕获时的 grid 上界 ≥ 重放时的真实值，多余 block 读 cu_seqlens 早退。
  这与 decode FULL 图 pad 出的空请求早退是同一机制。

因此 prefill 入图不需要把 raggedness 变成图形状，只需要：

1. 步总 token 数 **桶化**（补齐到桶）；
2. 捕获时元数据按**桶上界**构造（grid 固化在安全上界）；
3. 步内请求成员数（num_reqs）也钉死为桶值（原因见 §4.3，FA 的
   scheduler_metadata 依赖精确 num_reqs）。

## 3. 现状盘点

### 3.1 已就位（不需要重做）

| 组件 | 位置 | 状态 |
| --- | --- | --- |
| 协同捕获骨架 | `afd_plugin/v1/worker/ubatch_wrapper.py:80-98`（AFD 子类）+ 上游 `_capture_ubatches`（`gpu_ubatch_wrapper.py:255-310`） | 只认 `CUDAGraphMode.FULL`，桶化后直接复用 |
| dispatch/recv 不透明 op | `afd_plugin/connectors/gpu/async_moe_op.py`（132dca4） | AOT 编译在 op 处切分；pending payload 按 stage 键控 |
| 重放安全 flag 协议 | `async_gpu.py` 模块头 + e5e1514（seq 设备计数器、`FLAG_REPLY_READY` 常量、图内复位） | 已四次重放正确性验证 |
| header warm map | `async_gpu.py` `_headers_for_shape`，按 `(layer, num_tokens)` 键控 | 桶化后 key 有限，warmup 每桶覆盖 |
| FFN padded graph | `cuda_graph.py:137 padded_ffn_graph_shape`，尺寸 = `max_num_batched_tokens` | 桶 ≤ 它，必然覆盖；无需改 |
| 捕获元数据接口 | 上游 `_build_attention_metadata(for_cudagraph_capture=True)`（`gpu_model_runner.py:2276-2285`）→ builder `build_for_cudagraph_capture` | decode 捕获在用；prefill 桶需补/验证（§4.4） |
| 请求对齐劈批 | b40c8bc（`ubatch_split.py` patch） | prefill/decode 通用；需按桶语义校准守卫（§4.5） |
| AOT 编译穿透 MoE proxy | 已由 custom op 解决（编译 62s 一次，桶内动态形状） | 已验证 |

### 3.2 缺口（本方案要补的）

1. **dispatcher 不给 prefill 步发 FULL**：`vllm/v1/cudagraph_dispatcher.py`
   `_initialize_keys`（:180-232）只为 uniform decode 建 FULL key
   （`decode_mode() == FULL` 分支，`uniform=True`）；`dispatch`（:235-313）
   的 FULL 分支要求 key 精确匹配（含 num_reqs，"FA3 scheduler_metadata
   依赖精确 num_reqs"）。→ 需要为 prefill 建带 `uniform=False` 的 FULL key
   （§4.3）。
2. **AFD 校验只放行 FULL_DECODE_ONLY**：`cuda_graph.py:45-90
   validate_cuda_graph_mode` → 新策略。
3. **prefill 桶的 dummy run 与捕获序列**：`_dummy_run`（`gpu_model_runner.py:
   5826+`）的合成批分支（mixed/uniform）不含"prefill 桶"形态 → 增加。
4. **`is_last_ubatch_empty` 守卫按桶重新校准**：桶补齐后"最后 ubatch 全是
   padding"是常态而非异常（§4.5）。
5. **MLA builder 的捕获分支待确认**：当前 backend 是 `FLASH_ATTN_MLA`
   （`vllm/v1/attention/backends/mla/flashattn_mla.py:115
   FlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder)`）；其 prefill 调用
   传 `max_seqlen_q = max_query_len`（`mla/prefill/flash_attn.py:244-245`），
   是捕获时固化的 host 值——需要桶上界语义（§4.4 / P0 探针）。

另一个方向说明：上游对 prefill 入图的原生答案是 PIECEWISE，但 0.26 的
`UBatchWrapper` 捕获/重放分支只认 FULL（`gpu_ubatch_wrapper.py:455-461,
485-505`），PIECEWISE 与 DBO 双线程协同捕获在本版本不组合；且 PIECEWISE
每层保留 eager attention launch，主机削减幅度减半以上。故不选。

## 4. 方案

### 4.1 桶设计

- 桶集合建议 `{1024, 2048, 4096}`（= 步进 ×2，≤ `MAX_NUM_BATCHED_TOKENS`）。
  上限由两个硬约束决定：
    - **显存**：每个桶一张整前向协同图，激活显存 ∝ 桶大小；KV cache 占
    90% 预算下 3 桶的 FULL 图 + 8 个 decode 桶预计 < 2 GiB（tail2 实测
    FULL 单桶图仅 0.15 GiB@8tok，线性外推需实测确认）；
    - **补齐浪费**：真实步被补到桶，多出的行是真算（非空转）——桶步进越大
    浪费越高。tail 分布（短多长少）下建议小桶密、大桶疏。
- 两个桶值之间的步一律补到上桶。判据：补齐浪费的 GPU 时间 < 被消掉的
  主机时间（host-bound 下必然成立，直到 GPU busy 显著上升）。

### 4.2 捕获流程（启动期，每桶一次）

沿用既有顺序，逐桶执行：

1. `_dummy_run(num_tokens=桶, uniform_decode=False, should_ubatch=True)`：
   合成批用**两个不等长请求**（如 7:3 桶值），覆盖请求对齐劈批与延后 recv
   路径；`_build_attention_metadata(for_cudagraph_capture=True)` 用桶上界
   构造捕获元数据；
2. AFDUBatchWrapper `_capture_ubatches`：两线程协同捕获，MoE dispatch/recv
   op 与 yield 的交替顺序烘进图；
3. 捕获前 warm：connector `_headers_for_shape` 的 `(layer, 桶)` key 由该次
   dummy run 自然覆盖（它就是真实执行）；
4. `cudagraph_num_of_warmups ≥ 1` 保持既有防呆。

工程注意：捕获发生在 FFN rendezvous 之前（ffn_worker.py `start_ffn_server_loop`
的既有约定），探针流量会被 FFN 丢弃态吞掉——依赖现状即可，无需新增握手。

### 4.3 运行时裁决与补齐

- `CudagraphDispatcher._initialize_keys` 增补 prefill FULL key：
  `(num_tokens=桶, num_reqs=桶内合成请求数, uniform=False)`。**num_reqs 必须
  钉死**（上游注释：FA3 scheduler_metadata 依赖精确 num_reqs）——运行时不足
  桶值的请求用空请求补齐（decode FULL 的 padding 同理：空请求零 query token，
  或把补齐 token 归属到最后一个请求的 chunk，二者取一并在探针中定案）。
- 裁决入口在 AFD 单 rank 旁路（`attention_model_runner.py
  _should_ubatch_single_rank`，:632-676）与 `_determine_batch_execution_and_padding`：
  prefill 步 `num_tokens ≥ prefill 阈值` 时补到最近桶并裁决 FULL。
- DBO 劈批仍由 `ubatch_split.py` 请求对齐 patch 负责；**劈批决策先于补齐**
  （先在真实 token 上选请求边界，再对两个 ubatch 分别补齐到桶/半桶）。

### 4.4 ragged 注意力与内核网格（P0 探针，见 §5）

- `FlashAttnMLA` prefill 传 `max_seqlen_q = max_query_len`
  （`mla/prefill/flash_attn.py:244-245`）——捕获时必须用**桶上界**（任一请求
  ≤ 桶，天然安全上界），重放时 FA varlen 对超出实际长度的 block 依
  `cu_seqlens` 早退。
- 需逐项验证：MLACommonMetadataBuilder 是否实现
  `build_for_cudagraph_capture`（flashattn_mla.py:115 引用）；softmax_lse
  输出形状在补齐下的处理；KV 写入（`slot_mapping` 补齐行须指向 null block
  或 -1，decode 已有先例 `_get_slot_mappings` 的 -1 填充）；plugin 侧
  `routed_experts_slot_mapping_device` 的补齐快照（gpu_model_runner.py:2336
  附近的既有拷贝逻辑）。

### 4.5 DBO 语义与守卫校准

- 劈批先于补齐：请求对齐边界在**真实 token** 上选取；补齐 tail 归最后一个
  ubatch（上游 `_pad_out_ubatch_slices` 语义不变）。
- `is_last_ubatch_empty`（`ubatch_utils.py:49`）在桶化下会系统性误判
  （"后半桶全是 padding"是常态）：桶化模式下该守卫应改按
  `num_tokens vs 桶/2` 判定，或直接信任劈批决策（探针中定案）。
- 两条 ubatch 线程共享 connector，pending payload 按 stage 键控
  （async_gpu.py `pending_cam_dispatches`）——桶化不改变该机制。

### 4.6 校验与开关

- `validate_cuda_graph_mode`（cuda_graph.py:45-90）新增策略
  `FULL_PREFILL_BUCKETS`（attention 侧；与 FFN padded graph 组合），
  保持 DBO 限 2 ubatch 的断言。
- recipe `ATTN_EAGER` 开关扩展为三态：`1`=eager（现状默认）、`0`=decode 图
  （现有行为）、`2`=prefill 桶化 FULL（本方案）；桶列表由
  `PREFILL_BUCKETS` 环境变量注入。

## 5. P0 探针（第一个落地物，~1 天）

**目的**：单点验证"FLASH_ATTN_MLA varlen prefill 在固化 grid 下按桶重放"的正确性，
它决定方案成立与否。

**做法**：最小 Python 脚本（不起 vLLM 栈）：

1. 构造桶 1024 的 dummy prefill 元数据（uniform=True、max_query_len=1024），
   捕获一次 `FlashAttnMLA` prefill 调用（含 kv cache 写入）；
2. 以 5 组不同的 ragged 分布（不同请求数、不同 seq_len 组合，总和 ≤ 桶）
   重放，与 eager 直接调用逐元素对比（容差按 `async_gpu_moe_equivalence`
   的既有判据，禁止逐字符对比）；
3. 记录：输出最大误差、LSE 处理、越界 block 早退的耗时特征。

**出口判据**：5 组重放全部通过容差 → 方案成立，进入 §7 实施；任一组失败 →
评估退半图方案（attention eager + dense 段入图，收益减半）或换后端。

## 6. 风险表

| 风险 | 处置 |
| --- | --- |
| FLASH_ATTN_MLA 固化 grid 重放不正确 | P0 探针先行；失败则半图或后端替换 |
| 桶图显存挤占 KV cache | 桶数 ≤3 起步；`--kv-cache-memory` 显式预算；实测 FULL 图 0.15GiB@8tok 外推后复核 |
| 补齐浪费在长尾下放大 | 小桶密/大桶疏；补齐行是真实计算（非空转），统计进收益模型 |
| num_reqs 钉死与 chunked prefill 片段语义冲突 | chunk 片段视为一个整体成员参与劈批；空请求补齐方案在探针中定案 |
| `is_last_ubatch_empty` 误判 | 桶模式改写守卫（§4.5），单测覆盖 |
| decode 桶与 prefill 桶共存的路由混乱 | dispatcher key 已含 uniform 位；运行时按 uniform_decode 天然分流，单测覆盖 |
| host 成本下降后 GPU 成为新瓶颈、DBO 仍不赚 | 先量 GPU busy（tail 系列已有 4-5% 基线），桶化后再测；若仍 host-bound 则收益模型重新评估 |

## 7. 分阶段落地与出口判据

| 阶段 | 内容 | 出口判据 |
| --- | --- | --- |
| P0 | §5 探针 | 5 组重放通过容差；定案空请求补齐方式 |
| P1（~1 周） | dispatcher prefill FULL key + 捕获序列 + 守卫校准 + 校验策略 + recipe 三态 | 栈以 `ATTN_EAGER=2` 启动；OSL=1 负载全程走 replay（trace 中 cudaGraphLaunch > 0 且 eager kernel 归零） |
| P2 | 重尾 TTFT 对比重测（同 tail2 负载） + GPU busy 复测 | DBO 开的 total/p50/p90 ≤ DBO 关（首次正收益），或给出量化的剩余瓶颈 |
| P3 | E2E 接入（`afd-graph-dbo` 或新场景）、`moe_equivalence` 回归、文档转正式 | 4 场景门禁通过 |

## 8. 收益模型（何时赚、赚多少）

设每步主机成本 H（eager ≈ 13ms）、GPU 计算成本 G（attention+FFN+dispatch，
tail2 ≈ 10ms）、DBO 主机倍率 ×2、交叠节省 ≤ min(FFN 时间, 另一半 attention 时间)。

- eager：DBO 净值 ≈ +H − 交叠节省 → host-bound 下恒负（实测 −28%~−48%）。
- 桶化入图：H ≈ 0，DBO 净值 ≈ 补齐浪费 − 交叠节省。补齐浪费 = (桶 − 真实
  tokens) × 每 token 计算成本；桶设计得好时远小于交叠节省。
- 收益上限 = FFN 时间占步时间的比例；FFN 计算 ∝ tokens × topk × 专家宽，
  随模型/并发变大而变大——模型越大本方案收益越大。

## 9. 参考

- 实测数据与 trace：`/tmp/dbo_prof/{pre,tail,tail2}_{dbo,nobo}/`，
  Perfetto view：`/data/ajhou/dbo_traces/`；负载与客户端：
  `/tmp/dbo_make_tail_prompts.py`、`/tmp/dbo_tail_client.py`、
  `/tmp/dbo_tail_capture.sh`、`/tmp/dbo_profile_capture*.sh`
- 既有方案文档：`docs/design/async_gpu_cudagraph.md`（decode FULL 与 flag
  协议设计）、`docs/design/async_gpu_handoff.md`（eager DBO 的教训与数据）
- 关键提交：e5e1514（dispatch 捕获+重放安全）、6feb32d（拒绝门由来）、
  9320bc1（dbo.py compile 守卫）、b40c8bc（请求对齐劈批）、132dca4
  （dispatch/recv custom op + recipe `ATTN_EAGER`）
- 上游触点：`vllm/v1/cudagraph_dispatcher.py`（key 初始化 :180-232、
  dispatch :235-313）、`vllm/v1/worker/gpu_ubatch_wrapper.py`（FULL-only
  :455-505）、`vllm/v1/worker/gpu_model_runner.py`（`_dummy_run` 合成批
  :5826+、捕获元数据 :2276-2285）、`vllm/v1/attention/backends/mla/
  flashattn_mla.py`（FLASH_ATTN_MLA builder :115、prefill 网格 :244-245）
