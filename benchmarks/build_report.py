"""Build a self-contained, source-bounded report from completed benchmark rows."""
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
rows = [json.loads(path.read_text()) for path in (HERE / "results/java-standard-v1").glob("*/result.json")]
rows.sort(key=lambda r: (r["n"], r["failure_count"]))
assert len(rows) == 9 and all(r["forest"].get("validated") for r in rows)
sweep = json.loads((HERE / "results/java-smoke-v1/closd16_f0/finite_sweep/result.json").read_text())
assert all(r["validated"] for r in sweep)
summary = []
for r in rows:
    folder = HERE / "results/java-standard-v1" / f"closd{r['n']}_f{r['failure_count']}"
    pruned = json.loads((folder / "pruned_t12/result.json").read_text())
    f, ref = r["forest"], r["reference"]
    summary.append({"n": r["n"], "failure": r["failure_count"],
        "forest_rs_ag_bandwidth_time": f["rs_ag_optimum_time"],
        "mcnf_archived_t12_objective": ref["objective"],
        "mcnf_archived_feasible": ref["standard_feasible"],
        "mcnf_archived_max_violation": ref["max_violation"],
        "forest_min_depth": f["min_depth"], "forest_max_depth": f["max_depth"],
        "forest_generation_sec": f["java_wall_sec"] + f["mapback_sec"],
        "mcnf_historical_worker_sec": ref["timings"]["worker_wall_time_sec"],
        "retained_trees_t12": pruned["retained_out_trees"], "pruned_t12_status": pruned["status"],
        "forest_java_peak_rss_mib": f["java_peak_rss_mib"],
        "input_sha256": r["input_sha256"],
        "trees_sha256": hashlib.sha256((folder / "trees.json").read_bytes()).hexdigest()})
(HERE / "summary.json").write_text(json.dumps({"standard": summary, "finite_sweep": sweep}, indent=2) + "\n")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
ax = axes[0]
for f in range(3):
    part = [r for r in summary if r["failure"] == f]
    ax.plot([r["n"] for r in part], [r["forest_max_depth"] for r in part], marker="o", label=f"ForestColl f{f}")
ax.axhline(12, color="#c55b30", linestyle="--", label="MCNF horizon T=12")
ax.set(xlabel="Active GPUs", ylabel="Maximum logical tree depth", ylim=(0, 275), xticks=[64, 128, 256])
ax.legend(fontsize=9)
ax = axes[1]
ax.plot([r["horizon"] for r in sweep], [r["objective"] for r in sweep], "o-", label="Fixed ForestColl trees + LP timing")
ax.axhline(30, color="#c55b30", linestyle="--", label="HWC at T=12 / bandwidth limit")
ax.set(xlabel="Allowed rounds T (ClosD16, f0)", ylabel="Normalized completion time", ylim=(0, 38), xticks=[12, 32, 64, 128])
ax.legend(fontsize=8)
fig.savefig(HERE / "comparison.png", dpi=180)
plt.close(fig)

header = "| GPU | 坏卡 | ForestColl 带宽时间 | MCNF 历史 T12 | 历史原空间合格 | 树深 min–max | 合成秒 | T12 剩余树 | 剔除后重优化 |\n|---:|---:|---:|---:|:---:|:---:|---:|---:|:---|\n"
lines = []
for r in summary:
    lines.append(f"| {r['n']} | {r['failure']} | {r['forest_rs_ag_bandwidth_time']:.0f} | {r['mcnf_archived_t12_objective']:.6f} | {'是' if r['mcnf_archived_feasible'] else '否'} | {r['forest_min_depth']}–{r['forest_max_depth']} | {r['forest_generation_sec']:.3f} | {r['retained_trees_t12']} | {r['pruned_t12_status']} |")
table = header + "\n".join(lines)
readme = f"""# ForestColl / MCNF ClosD 首批比较

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

{table}

## 有限轮数证据

ClosD16 的 HWC f0/f1/f2 本机复跑均在 T=12 得到约30，原空间验证通过。
对于 Java 原版生成的 ClosD16/f0 树，固定其权重、只优化时间安排，T=32/64/128 的完成时间分别为 {sweep[0]['objective']:.6f} / {sweep[1]['objective']:.6f} / {sweep[2]['objective']:.6f}，均通过原空间检查。稳态带宽值为30。
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
"""
(HERE / "README.md").write_text(readme)
cells = "".join("<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in line.strip("|").split("|")) + "</tr>" for line in lines)
paragraphs = "".join(f"<p>{html.escape(p.strip('- '))}</p>" for p in readme.split("\n") if p.startswith("- "))
page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ForestColl · ClosD 比较</title><style>body{{font:16px/1.7 system-ui,sans-serif;background:#f4f6f8;color:#203048;margin:0}}main{{max-width:1280px;margin:40px auto;padding:32px;background:white;border-radius:12px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dde3ea;text-align:right}}th{{background:#edf2f6}}p{{max-width:1050px}}img{{width:100%}}.note{{background:#eaf3ef;padding:18px;border-left:4px solid #438a68}}a{{color:#23629c}}</style><main><h1>ForestColl 在标准 ClosD 上的表现</h1><p>2026-09-23 · 16卡验证 + 64/128/256卡 × 坏0/1/2卡 · 原 MCNF_tree 无修改</p><p class="note"><b>9/9 完整森林通过物理容量检查。</b>稳态 RS+AG 带宽值为126/254/510；它们不是 T=12 完成时间。本次树深最高63/127/255。按我们的实际截断规则剔除过深树后，9个标准case均无法在T=12完成。</p><img src="comparison.png" alt="树深随规模增长，以及16卡固定树完成时间随轮数逼近带宽值"><h2>标准结果</h2><div style="overflow:auto"><table><thead><tr>{''.join('<th>'+v+'</th>' for v in ['GPU','坏卡','ForestColl带宽时间','MCNF历史T12','原空间合格','树深范围','合成秒','T12剩余树','截断重优化'])}</tr></thead><tbody>{cells}</tbody></table></div><h2>比较口径</h2>{paragraphs}<h2>有限轮数验证</h2><p>ClosD16/f0 原森林配额不变，MCNF仅优化时间安排：T=32 → {sweep[0]['objective']:.6f}；T=64 → {sweep[1]['objective']:.6f}；T=128 → {sweep[2]['objective']:.6f}。HWC在T=12已达到30。以上均经过原空间检查，尚未进行真实GPU通信测试。</p><p>论文附录E.3承认低树高有利于延迟和吞吐收敛，但最小高度树打包的困难性不能推出所有多项式算法都无法控制树深。</p><p><a href="README.md">详细说明与重跑命令</a> · <a href="summary.json">汇总JSON</a> · <a href="https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf#page=25">论文附录E.3</a></p></main></html>"""
(HERE / "report.html").write_text(page)
provenance = {"forest_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "platform": platform.platform(), "python": sys.version,
              "java": subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True),
              "mcnf_snapshot": json.loads((HERE / "vendor/manifest.json").read_text()),
              "adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}}
(HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print("Wrote README.md, summary.json, report.html, comparison.png, provenance.json")
