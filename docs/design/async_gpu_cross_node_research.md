---
title: Async GPU Connector 跨机器（跨节点）调研
kind: research
status: draft
primary_code_paths:
  - "afd_plugin/connectors/gpu/nvshmem_rt.py"
  - "afd_plugin/connectors/gpu/symm_window.py"
  - "afd_plugin/connectors/gpu/async_gpu.py"
  - "afd_plugin/connectors/async_topology.py"
depends_on:
  - "rfc_async_gpu_connector.md"
  - "async_gpu_connector.md"
  - "async_gpu_handoff.md"
---

# Async GPU Connector 跨机器调研

`GpuAsyncAFDConnector` 当前**只支持单机**：RFC 和交接文档都把"跨节点"列在限制项里，
但都没有展开要改什么。本文顺着代码把阻塞点钉到具体位置，并给出一个改造顺序。结论
先行：**阻塞点比想象的少，都集中在传输层的两处；拓扑推导和 bootstrap 已经是网络中立
的，不用碰。真正的未知数是性能——现有的瓶颈分析建立在 NVLink 级延迟上，换成跨机网络
后可能整体作废，需要重新画像。**

**2026-08-25 更新：§2.1、§2.2 已经实现**（传输层换成 NVSHMEM put + 显式 order 点），
单测全绿，但**还没有在真实多机环境验证过**——细节和踩到的坑见对应小节末尾的更新。

## 1. 现状：为什么现在是单机

三处代码，从表层到根因：

1. **配方脚本强制关闭远程传输。** 所有 `recipe/gpu/GpuAsyncAFDConnector/**/*.sh` 都
   `export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}`——这是为了不让
   NVSHMEM 去初始化 IBRC，因为容器没有 RDMA 权限会直接 abort（`async_gpu_handoff.md`
   §6）。这只是运维层面的规避，不是设计限制。
2. **`peer_ptr` 设计上要求 P2P 直达。**
   [nvshmem_rt.py:205-214](../../afd_plugin/connectors/gpu/nvshmem_rt.py:205) 的
   `peer_ptr()` 调用 `nvshmem_ptr()`，对不可达的 PE 返回 NULL 时立即抛错——这是
   NVSHMEM 的正常语义（`nvshmem_ptr` 只在有对称内存 handle 直接映射时才返回非空指针），
   不是 bug。
3. **`SymmWindow.__init__` 对所有 peer 都急切调用 `peer_ptr`。**
   [symm_window.py:328-331](../../afd_plugin/connectors/gpu/symm_window.py:328)：
   ```python
   self._peer_base = {
       pe: (self._base if pe == rank else nvshmem_rt.peer_ptr(self._base, pe))
       for pe in range(world_size)
   }
   ```
   这是**当前"跨节点直接崩"的确切位置**——不是运行时某次传输才报错，而是连接器初始化
   阶段，只要有一个 peer 不在 NVLink/PCIe P2P 范围内，`SymmWindow` 就建不起来。

## 2. 需要改的东西

### 2.1 写入路径：从"裸指针 + copy_"换成 NVSHMEM 的单边 put —— **已实现**

原写法是 P2P 映射拿到对端裸指针，再用普通 `torch.Tensor.copy_` 写过去。这条路径的
前提就是 P2P 直达；跨机器上 `nvshmem_ptr` 不会（也不应该）返回非空指针。

已经改成调用 NVSHMEM 自己的单边写原语 `nvshmemx_putmem_on_stream`
（[nvshmem_rt.py](../../afd_plugin/connectors/gpu/nvshmem_rt.py) 新增的
`put_on_stream`），让传输走 NVSHMEM 内部的 transport 选择——同一份 PE 编号，可达时走
P2P/NVLink，不可达时自动走 IBRC/GPUDirect RDMA，调用方不用关心是哪一种。

关键是 put 的 `dest` 参数不是对端裸指针，而是**本 rank 自己的 base 地址**
（`self._base + offset`）——每个 PE 都用同一套公式算出同一个值，NVSHMEM 内部把它翻译
成 `peer` 上的物理地址。这意味着 `SymmWindow.__init__`
（[symm_window.py:325](../../afd_plugin/connectors/gpu/symm_window.py:325)）不再需要
对每个 peer 预先解析 `peer_ptr()`——§1 的那个初始化崩溃点作为这次改动的副作用被顺带
消除了：`peer_ptr` 还留在 `nvshmem_rt.py` 里（没删，未来若要做"机内 P2P 快路径 + 跨机
put 兜底"的混合方案还用得上，见 §7），但现在没有任何生产代码调用它。

**接收侧确实不用改**，和预想的一致。`poll()`（symm_window.py）和
[cuda_rt.py](../../afd_plugin/connectors/gpu/cuda_rt.py) 的 `stream_wait_value32`
读写的都是接收方**自己本地**符号内存里的 flag/header，不管发送方是直写还是走 RDMA put
落地，接收方永远在读本地内存。

**一个必须付的代价：put 不做隐式类型转换。** 原来的 `copy_` 在写进映射视图时顺手做了
窄化转换（`send_ffn_output` 里 float32 的 `reduced` 写进 bf16 的 slot），put 是裸字节
搬运，做不到这一点。现在 `write_slot` 在 dtype 不匹配时显式 `.to(dtype)` 一次，这是
真实多出来的一趟 kernel，不是免费的——`async_gpu_handoff.md` 记录过类似"看似免费的转换
其实要单独量一次"的教训，这里先老实付掉，要不要优化留给后续 profile。

**Header 的 CPU 分支多了一跳。** 原来"pinned host → 对端映射视图"是一次 H2D 就能直接
落到对端；put 的 source 必须是设备内存，所以现在是"pinned host → 本 rank 一个小的设备
暂存 tensor → put"。这个设备暂存 tensor 在 `__init__` 里分配一次、所有发送复用，安全性
建立在"这个连接器的所有发送目前都在同一条 stream 上按序发出"这个既有假设上（§8 提过
"发送独立 stream" 试过并回退了）——如果以后真的给发送引入多流，这个复用假设要重新
审视，代码里留了注释标出这一点。

**顺手发现并修的一个已有 bug：** `announce_shutdown`（async_gpu.py）原来的
`write_slot` 调用传了不存在的 `shared_idx` 关键字参数、漏传必填的 `expand_idx` /
`weights`，只要真的走到关停路径就会 `TypeError`。改 `write_slot` 签名时顺带看到，
已修正——这和跨机器无关，是踩点时顺手拾到的。

### 2.2 写序：同 stream 顺序完成的假设需要显式 order 点 —— **已实现（细节和最初设想不同）**

`symm_window.py` 模块文档原来点破的问题：payload 和 flag 在同一个 stream 上顺序发出，
靠"同 stream 顺序完成"隐含"看到 flag 就等于 payload 写完"。这个假设**只在 NVLink
映射内存上成立**——两次 D2D copy 走同一条物理链路，天然保序；换成 put 之后不再成立。

**实现时发现一个和最初设想不同的地方：** 已安装的 NVSHMEM host API
（`nvidia-nvshmem` wheel 自带的头文件，见
`nvshmemx_api.h`）里只有 `nvshmemx_quiet_on_stream`，**没有** `nvshmem_fence` 的
stream-enqueue 版本——`fence` 只有阻塞式的主机直调版本，排不进 CUDA stream。既然
`write_slot` 的所有写入都是通过 `_on_stream` 系列排进 stream 的，能拿来在 payload put
和 flag put 之间插一刀的只有 `nvshmemx_quiet_on_stream`。

`quiet` 比 `fence` 语义更重——`fence` 只保证顺序（不等完成），`quiet` 会等这个 PE
之前发出的所有 put 都完成。正确性上没问题（是更强的保证），但可能比严格意义上的
"只要保证顺序"更贵。已经在 [nvshmem_rt.fence_on_stream](../../afd_plugin/connectors/gpu/nvshmem_rt.py) 的 docstring
里记下这个取舍，作为一个待验证的性能项——先保证正确，等真的有跨机部署可以 profile 了
再决定要不要换更轻量的方案（比如设备侧 kernel 里调 `nvshmem_fence`）。

### 尚未验证

以上两点改完之后，`uv run python -m pytest tests/unit -q` 全绿（573 passed），三个新
绑定的符号名和签名也在真实的 `libnvshmem_host.so.3` 里核对过、能正常 `ctypes` 绑定。
但**没有在真实的多 GPU / 多机环境跑过** `tests/e2e/async_gpu_window_roundtrip.py` 或
`tests/e2e/async_gpu_connector_e2e.py`——这两个测试都需要至少两张真实 GPU，本次改动
没有这样的环境可用。put 的语义假设（非 nbi 变体在 stream 上排队时，source buffer 在
该 stream 位置的操作完成后即可安全复用）是从 OpenSHMEM/NVSHMEM 一贯的
blocking-vs-nbi 命名约定推断的，**建立在真实文档惯例上，但没有在硬件上跑过验证**，
这是接手人第一件要做的事。

### 2.3 Bootstrap：已经是网络中立的，不用改

`nvshmem_rt.init()`（nvshmem_rt.py:118-181）通过 AFD 进程组的 store 交换 uniqueid，
这一步走的是 TCPStore，本来就能跨机；`async_topology.py` 里完全没有 node/local_rank
概念，拓扑推导（Attention 在前、FFN 在后的 world rank 分配）不用碰。这是这次改造最大
的杠杆点——和 RFC 里"异步 DP 引擎补丁已平台中立"是同一个模式。

### 2.4 运维/配置项（代价小，照抄现成先例）

- **放开 `NVSHMEM_REMOTE_TRANSPORT`**，让 IBRC/UCX 真正初始化。需要宿主机有 RDMA
  网卡权限，以及 `NVSHMEM_HCA_LIST` 之类的环境变量——这块目前完全没人验证过，
  容器化环境下大概率还要过一轮权限/驱动的坑。
- **rendezvous host 目前硬编码 `127.0.0.1`**（所有配方脚本的 AFD 世界 rendezvous
  `"host"` 字段、vLLM `--host`）。换成真实可路由地址即可；
  `P2pNcclAFDConnector` 的 `prefill_decode_disaggregation` 配方已经在用
  `--prefiller-hosts` / `--decoder-hosts` 做跨机部署，可以照抄这个模式。

## 3. 不用改，但要重新验证的东西

**槽位不变式本身跨机器仍然成立。** `ring_depth` 由"每个 (peer, stage) 最多一个在途
请求"的不变式推导（`async_gpu_connector.md` §1），这条协议层的性质不依赖 NVLink，跨
机器一样成立。变的是"一个在途请求"对应的等待时间——NVLink 级 RTT 是亚微秒到微秒级，
IB/RoCE 的 RTT 通常是几十微秒起，量级差远不止一个数量级。

**这意味着 `async_gpu_handoff.md` §5-§11 的瓶颈分析可能整体要重新做，不能照搬结论：**

- 该文档定位的核心问题是"attention 主机线程 + 每层一次 A→F→A 往返把前向拍扁成
  `Σ max(host, gpu)`"，这个结论是在往返延迟是"主机能不能喂满 GPU"这个量级下成立的。
  跨机器往返变长一个数量级后，谁是瓶颈（主机线程 vs. 网络 RTT vs. GPU 空闲）需要重新
  profile，不能假设"host-bound"这个结论还成立。
- `recv_poll_timeout_ms`、`ring_depth` 这些按机内延迟拍的默认值，在新的 RTT 下大概率
  需要重新调；"是否要放宽单在途请求的不变式、上多槽位吃掉更高 RTT"是一个真实的性能
  权衡，但这是优化项，不是正确性前提——**先保证功能正确，再决定要不要做这层优化**，
  和 §5 一样的分层策略。

## 4. 测试基础设施缺口

现有验证手段全部假设单机：

- `tests/e2e/async_gpu_connector_e2e.py` 显式设 `NVSHMEM_REMOTE_TRANSPORT=none`，两个
  进程在同一台机器上跑。
- `gpu run`（本机 canhazgpu 预约系统）看不出跨机预留的支持——这本身是验证上面第 2 节
  改动的前提：需要两台通过 RDMA/IB 互联、且能同时被预约到的机器，这件事目前不确定
  能不能做到，需要先确认。

## 5. 顺带的技术债，值得一起处理

`close()` 目前不释放对称内存（RFC 已知风险，`nvshmem_free` 是 collective 调用，要求
两个角色同步退出才安全）。跨机场景下网络分区、单侧崩溃这类故障天然更容易触发，这条
技术债在跨机器上兑现的概率比单机高得多，建议跟着这次改造一起补，而不是继续留着。

## 6. 建议的落地顺序

1. **先确认能拿到两台通过 RDMA 互联的机器**（§4）——这是一切验证的前提，如果只能
   拿到 loopback 环境，后面的改动没法端到端验起来。**代码已经先落地**（§2.1、§2.2），
   但真正卡住验证的还是这一条：没有两机环境，`tests/e2e/async_gpu_connector_e2e.py`
   这类测试就跑不起来。
2. ~~传输层换成 NVSHMEM put + 显式 fence~~ **已完成**（§2.1、§2.2）——接收侧确实没碰；
   实现时发现 host API 只有 `quiet_on_stream`、没有 `fence` 的 stream 版本，用前者顶上，
   细节见 §2.2。**尚未在真实硬件上跑过，是下一步验证的第一优先级。**
3. **配置项**（§2.4）：放开 remote transport、host 从回环换成真实地址——还没做，可以
   和第 1 步的多机环境搭建并行准备。
4. **端到端跑通后重新画像**（§3）——不要复用 `async_gpu_handoff.md` 的瓶颈结论和
   参数默认值，跨机器的性能特征大概率是另一套故事。
5. **`close()` 补上对称内存释放**（§5）——功能验证阶段就会需要反复起停，顺手修掉。

## 7. 开放问题

- NVSHMEM 的 IBRC transport 在目标部署环境（RDMA 网卡型号、容器权限模型）下是否真的
  可用，还是需要换 UCX 或者别的 bootstrap 方式？这个只能上机器试，调研阶段回答不了。
- 是否要接受"机内走 P2P、跨机走 put"这种混合路径（NVSHMEM 本来就是这么设计的，
  `nvshmem_ptr` 可达时可以继续用现在的裸指针快路径，不可达时才落到 put），还是干脆
  统一走 put 简化代码、放弃机内的裸指针优化？前者性能更好但要维护两条路径，后者更
  简单但可能在纯单机场景上出现回退——值得在开始写代码前拍板。
