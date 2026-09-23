最新的 ClosD + UBD 共18行汇总见 [深度与带宽报告](depth_bandwidth.md)、[CSV](depth_bandwidth.csv) 和 [JSON](depth_bandwidth.json)。下文保留第一轮 ClosD 实验记录。

# ForestColl / MCNF ClosD 首批比较

已完成 16 卡小规模验证，以及标准 ClosD 64/128/256 × 坏 0/1/2 卡共 9 行 ForestColl 完整树生成与物理容量校验。全部修改和结果在本 fork 内；原 MCNF_tree 无修改。上游三阶段算法源码未修改。

上游：<https://github.com/liangyuRain/ForestColl>，revision `d7bd6a94e443e218aa26f319cbe22c70fe16921f`。
MCNF 代码快照 revision `91717ac39bc043fac7490e13beeb082b9cc82fc9`，12 个原样复制的 Python 文件及逐文件 SHA256 见 `vendor/manifest.json`。只新增隔离包入口。

## 比较口径

- 标准输入直接取归档中的 network/task/comm/modeling 字段；失败 ID 为 64:[13,2]、128:[26,5]、256:[52,11]。启用备用 GPU，保持活动卡数不变；带宽均为 1。
- 任务 data_volume=1 在 canonical generator 中扩成 M=N。ForestColl 每根 k=1，每棵树广播量 M/(Nk)=1；AllGather 带宽时间 U/k，RS+AG 两阶段带宽时间 2U/k。
- ForestColl 原生没有 T 参数。`fixed_K` 控制每根树数，不控制深度。代码 `convertToRuntime` 返回 U/(Nk)，乘 M 得带宽模型的时间；未加入树深、启动延迟或有限分块的流水线填充/排空时间。
- `forest_t12` 固定原树和原权重，在复制的 canonical MCNF 模型中检查可行性。`pruned_t12` 才是本项目的真实截断规则：剔除 depth>T 的完整候选树，并让 LP 重优化剩余树权重。这是 ForestColl 候选树 + MCNF 混合诊断，不能称为 ForestColl 原算法结果。
- 单向深度≤T 只是初筛。AllReduce 还要在同一 T 内完成 reduce 与 broadcast 的依赖。ClosD64/f0 剩一棵深度 12 的树，但两阶段仍不可行；其余 8 个标准 case 的所有候选树都被剔除。
- 所有树按 GPU 间逻辑发送计算深度，交换机不另算计算节点。检查了根、全部参与者、路径连续性、真实物理边、每根 chunk 总量、边容量和与 MCNF 唯一物理路由的一致性。
- MCNF 标准大规模结果来自现有 main-only 归档，没有重新跑 GPU。128/f2、256/f1、256/f2 不满足 1e-4 原空间门槛；保留标识，不能当作可靠最优值。64 与128其余行约等于 ForestColl 带宽值；256/f0 为518.045，相比理想带宽值510约高1.58%，但这不是可实现的 ForestColl T12 加速比。
- ForestColl 合成在本机 Apple M4 / Java25 / common pool parallelism=4 / Xmx4g 上执行。合成秒包含 JVM 启动与映射回物理路径；MCNF 历史求解在另一台 CPU/GPU 机器上，不能据此宣称同机求解加速。时间为单次观测，部分小规模诊断与大规模合成有重叠。
- 没有进行 NCCL/RCCL GPU 执行或真实网络测试；这里比较的是合成和数学调度。

## 标准矩阵

| GPU | 坏卡 | ForestColl 带宽时间 | MCNF 历史 T12 | 历史原空间合格 | 树深 min–max | 合成秒 | T12 剩余树 | 剔除后重优化 |
|---:|---:|---:|---:|:---:|:---:|---:|---:|:---|
| 64 | 0 | 126 | 125.999992 | 是 | 12–63 | 1.867 | 1 | infeasible |
| 64 | 1 | 126 | 126.000025 | 是 | 13–63 | 1.320 | 0 | no_tree_after_depth_filter |
| 64 | 2 | 126 | 125.999993 | 是 | 17–63 | 1.691 | 0 | no_tree_after_depth_filter |
| 128 | 0 | 254 | 254.000004 | 是 | 38–127 | 7.300 | 0 | no_tree_after_depth_filter |
| 128 | 1 | 254 | 253.999874 | 是 | 38–127 | 7.541 | 0 | no_tree_after_depth_filter |
| 128 | 2 | 254 | 254.021537 | 否 | 33–126 | 7.886 | 0 | no_tree_after_depth_filter |
| 256 | 0 | 510 | 518.045259 | 是 | 88–255 | 64.812 | 0 | no_tree_after_depth_filter |
| 256 | 1 | 510 | 517.780204 | 否 | 89–255 | 82.361 | 0 | no_tree_after_depth_filter |
| 256 | 2 | 510 | 518.237883 | 否 | 87–255 | 71.105 | 0 | no_tree_after_depth_filter |

## 有限轮数证据

ClosD16 的 HWC f0/f1/f2 本机复跑均在 T=12 得到约30，原空间验证通过。
对于 Java 原版生成的 ClosD16/f0 树，固定其权重、只优化时间安排，T=32/64/128 的完成时间分别为 33.693504 / 30.161532 / 30.002436，均通过原空间检查。稳态带宽值为30。
这说明在这个已测例子中，充分增加轮数可以逼近带宽值，而低轮数下 HWC 已达到它。
小规模按截断规则重优化：f0/f2 不可行；f1 可行但目标224。此结果针对保存的这组树，不是对所有 ForestColl 树选择的下界。

![comparison](comparison.png)

## 对树深的解释

论文 §3 明确选择优先吞吐；附录 E.3 承认低树高有利于低延迟及接近最优吞吐，并引用最小高度边不相交生成树构造的 NP-complete 结果。依据：<https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf#page=25>。
准确的比较主张是：其吞吐最优保证不包含树高或有限轮次完成时间保证。本次生成树确实出现近线性深度。不能把最小树高的困难性扩大为“任何多项式算法都无法控制树高”，也不能把本模型的 T12 不可行写成真实硬件上无法执行。

## 重跑

Python 使用 conda math；Java 使用支持源码 API 的 JDK21+（本次JDK25）。脚本拒绝覆盖已有结果目录。

```sh
conda run -n math python -B benchmarks/setup_java.py
conda run -n math python -B benchmarks/clos_benchmark.py --n 16 --failure 0 --label new-smoke --backend java --finite --baseline
conda run -n math python -B benchmarks/clos_benchmark.py --n 64 --failure 1 --label new-standard --backend java --finite
conda run -n math python -B benchmarks/check_clos.py
```

`finite_sweep.py <case_dir> --horizons 32 64 128` 保持 ForestColl 原权重，优化有限步调度。`evaluate_pruned.py` 重放本次12个Java case的 depth≤12 截断并自由优化权重；已有目录不会被覆盖。`build_report.py` 只从本次完成结果重建报告。

原始输入、树、Java阶段输出、日志、求解向量均保存在 `results/`。NPZ 向量以及Java依赖和构建产物被 Git 忽略；依赖的固定URL和SHA256在 `java_dependencies.json`。Java的集合迭代和并行实现可能改变树的具体选择；每次的原始树及其哈希均保存，不能假定重跑逐字节相同。
