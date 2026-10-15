"""Report separate depth and bandwidth coefficients; do not substitute alpha/beta."""
from collections import Counter
import csv
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
from clos_benchmark import HERE, case, create_topology, graph_from_topology, validate_trees


def load_rows():
    rows = []
    for network, label in (("closd", "java-standard-v1"), ("ubd", "ubd-standard-v1")):
        for n in (64, 128, 256):
            for failure in range(3):
                folder = HERE / "results" / label / f"{network}{n}_f{failure}"
                attempts = []
                if network == "ubd":
                    for candidate in sorted((HERE / "results").glob(f"ubd-standard-v*/ubd{n}_f{failure}/result.json")):
                        attempt = json.loads(candidate.read_text())
                        attempts.append({"path": str(candidate.relative_to(HERE)),
                                         "validated": attempt["forest"].get("validated", False),
                                         "error": attempt["forest"].get("generation_error"),
                                         "wall_sec": attempt.get("wall_sec")})
                        if attempt["forest"].get("validated"):
                            folder = candidate.parent
                result = json.loads((folder / "result.json").read_text())
                f = result["forest"]
                if not f.get("validated"):
                    raise RuntimeError(f"Forest not complete: {folder.name}")
                config, plan, reference = case(n, failure, network)
                topology = create_topology(network, config["network"]["params"])
                graph = graph_from_topology(topology)
                serialized = json.loads((folder / "trees.json").read_text())
                trees = {(t["root"], t["index"]): [[tuple(e) for e in p] for p in t["paths"]] for t in serialized}
                counts = {(t["root"], t["index"]): t["chunks"] for t in serialized}
                metrics, _ = validate_trees(graph, set(topology.gpu_nodes), trees, counts, f["U"], f["k"])
                loads = Counter()
                for key, paths in trees.items():
                    for path in paths:
                        for edge in path:
                            loads[edge] += counts[key] / f["k"]
                objective = metrics["allreduce_fixed_forest_bandwidth_time"]
                bottlenecks = [{"edge": [u, v], "capacity": d["capacity"],
                               "ag_load": loads[u, v], "rs_load": loads[v, u]}
                              for u, v, d in graph.edges(data=True)
                              if abs((loads[u, v] + loads[v, u]) / d["capacity"] - objective) < 1e-8]
                rows.append({"network": network, "num_gpus": n, "failure_count": failure,
                    "single_phase_depth": metrics["max_depth"],
                    "allreduce_dependency_depth": metrics["allreduce_dependency_depth"],
                    "bandwidth_objective": objective,
                    "sequential_rs_ag_objective": metrics["rs_ag_congestion_time"],
                    "allgather_objective": metrics["allgather_congestion_time"],
                    "tree_count": metrics["tree_count"], "trees_per_root_k": f["k"],
                    "minimum_tree_depth": metrics["min_depth"],
                    "mcnf_t12_historical_objective": reference["objective"],
                    "mcnf_original_max_violation": reference["max_violation"],
                    "mcnf_passes_current_1e_4_gate": reference["standard_feasible"],
                    "mcnf_original_status": reference["raw_status"],
                    "generation_seconds": f["java_wall_sec"] + f["mapback_sec"],
                    "java_peak_rss_mib": f["java_peak_rss_mib"],
                    "validated": True, "failure_plan": plan, "bottleneck_edges": bottlenecks, "attempts": attempts,
                    "deadline_override": json.loads((folder / "deadline_override.json").read_text()) if (folder / "deadline_override.json").exists() else None,
                    "artifacts": str(folder.relative_to(HERE)),
                    "input_sha256": result["input_sha256"],
                    "trees_sha256": hashlib.sha256((folder / "trees.json").read_bytes()).hexdigest()})
    return rows


def main():
    rows = load_rows()
    definitions = {
        "single_phase_depth": "Maximum GPU-to-GPU logical tree depth D; switch hops do not add logical rounds.",
        "allreduce_dependency_depth": "2D: longest reduce-then-broadcast dependency chain for the saved paired trees.",
        "bandwidth_objective": "B_inf=max_e((L_AG(e)+L_AG(reverse(e)))/capacity(e)); fixed saved tree weights, overlapped phases, M=N.",
        "sequential_rs_ag_objective": "Sum of separate reduce-scatter and allgather congestion times; may exceed B_inf.",
        "alpha_beta": "No numerical alpha/beta substituted. 2D and B_inf are separate coefficients; their sum with alpha/beta is a latency+congestion estimate, not a verified finite-packet makespan.",
        "scope": "Upstream equal-shard allgather trees and their reversal for allreduce; not the paper Appendix G allreduce-optimization LP."
    }
    provenance = {"forest_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip(),
                  "mcnf_snapshot_revision": json.loads((HERE / "vendor/manifest.json").read_text())["source_revision"],
                  "python": sys.version, "platform": platform.platform(),
                  "adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}}
    (HERE / "depth_bandwidth.json").write_text(json.dumps({"definitions": definitions, "provenance": provenance, "rows": rows}, indent=2) + "\n")
    fields = [k for k in rows[0] if k not in ("failure_plan", "bottleneck_edges", "attempts", "deadline_override")]
    with (HERE / "depth_bandwidth.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for index, network in enumerate(("closd", "ubd")):
        for failure, color in enumerate(("#277da8", "#df8f28", "#569467")):
            part = [r for r in rows if r["network"] == network and r["failure_count"] == failure]
            x = [r["num_gpus"] for r in part]
            axes[index, 0].plot(x, [r["single_phase_depth"] for r in part], "o-", color=color, label=f"failure {failure}")
            axes[index, 1].plot(x, [r["bandwidth_objective"] for r in part], "o-", color=color, label=f"ForestColl f{failure}")
            axes[index, 1].plot(x, [r["mcnf_t12_historical_objective"] for r in part], "x--", color=color, alpha=.65, label=f"MCNF historical f{failure}")
        axes[index, 0].set_title(network.upper() + " logical depth D")
        axes[index, 1].set_title(network.upper() + " normalized bandwidth objective")
        for ax in axes[index]:
            ax.set_ylim(bottom=0)
            ax.set_xticks([64, 128, 256])
            ax.set_xlabel("Active GPUs")
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "depth_bandwidth.png", dpi=170)
    plt.close(fig)

    notes = [
        "本轮按用户修订，只记录深度与带宽 objective，不代入论文的 α、β，不运行参数扫描。ClosD 和 UBD 各9行，共18行均保存完整树并通过容量校验。",
        "D 是单阶段逻辑树高；2D 是同一棵树先归约再广播的最长依赖链。逻辑树的一条边可以包含多个交换机物理跳，沿用 MCNF 的 GPU 间通信步定义。",
        "M=N：task.data_volume=1 经标准生成器扩大到N，每个根负责1单位，每个 elementary tree 负责1/k单位；压缩后的树记录chunks/k单位。容量和objective均沿用标准归一化单位。",
        "B∞按归约与广播合并后的逐物理边总负载计算，允许两个阶段流水重叠；不会强制全局阶段屏障。顺序执行的两个阶段objective也单独保留。",
        "后续可用(2D)α+B∞β作为延迟项加拥塞项的估计；这两个独立极限不一定能在同一有限分块调度里同时达到，不能直接标作已验证执行时间。",
        "这次测量的是开源默认的等分片AllGather生成树及其反向组合。论文§5.7明确指出，该组合不保证AllReduce全局最优；论文附录G的专门AllReduce优化LP未在此次适配中实现。",
        "UBD输入复用issue237归档，并显式填写当前标准带宽1/4/4/32/64。固定故障64:[13,2]、128:[26,5]、256:[52,11]均在原rack0；f2按既有替代策略启用rack0和rack1备用卡，不扩容。原标准文档中“两故障位于不同rack”的文字与实际输入不符，本实验以代码和归档的确切映射为准，未修改MCNF仓库。",
        "MCNF结果复用历史T12 main-only归档。UBD原始状态全部为optimal_inaccurate，保留原状态；仅64/f0与64/f1的原空间最大违反≤当前1e-4门槛。其余历史UBD数值仅供观察，不能据此宣称严格胜负。",
        "Java并行实现、JDK25（启用-ea断言）、common pool parallelism=4、Xmx4g。树的具体选择可能随集合遍历和并行运行变化；本次原始树及SHA256均保存。没有真实GPU网络执行，也没有把异机历史求解时间当成同机加速比。",
        "UBD256/f0首轮在900秒时限内未完成，失败尝试保留在ubd-standard-v1；重跑使用独立v2目录，初始3600秒观察时限在不重启Java工作进程的情况下延至5400秒；具体监督记录在deadline_override.json。表中合成时间指最终完成的一次尝试，失败尝试另见JSON中的attempts。",
        "原ForestColl三阶段算法文件及MCNF源仓库均未修改。适配、报告、完整输出只在ForestColl fork内。"
    ]
    columns = ["网络", "GPU", "坏卡", "D", "2D", "B∞", "顺序RS+AG", "MCNF历史T12", "历史≤1e-4", "树数", "合成秒"]
    def cells(r):
        return [r["network"], str(r["num_gpus"]), str(r["failure_count"]), str(r["single_phase_depth"]),
                str(r["allreduce_dependency_depth"]), f"{r['bandwidth_objective']:.6f}",
                f"{r['sequential_rs_ag_objective']:.6f}", f"{r['mcnf_t12_historical_objective']:.6f}",
                "是" if r["mcnf_passes_current_1e_4_gate"] else "否", str(r["tree_count"]), f"{r['generation_seconds']:.2f}"]
    mdtable = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    mdtable += "\n".join("| " + " | ".join(cells(r)) + " |" for r in rows)
    text = "# ForestColl：ClosD / UBD 标准实验的深度与带宽\n\n" + "\n\n".join(notes) + "\n\n" + mdtable
    text += "\n\n![深度和带宽](depth_bandwidth.png)\n\n"
    text += "带宽计算：令 L_e 为所有保存的 out-tree 在物理有向边e上的总负载，则 B∞=max_e (L_e+L_reverse(e))/W_e。归约用反向路径，广播用原路径。\n\n"
    text += "数据：[CSV](depth_bandwidth.csv)、[JSON与逐边瓶颈](depth_bandwidth.json)。上轮ClosD有限T诊断仍在[旧报告](report.html)。\n\n"
    text += "重跑：`conda run -n math python -B benchmarks/clos_benchmark.py --network ubd --n 64 --failure 0 --label new-ubd --backend java`。结果目录不得已存在。验证：`conda run -n math python -B benchmarks/check_clos.py`。生成本报告：`conda run -n math python -B benchmarks/build_depth_report.py`。\n\n"
    text += "论文：[§5.7](https://www.usenix.org/system/files/nsdi26-zhao-liangyu.pdf#page=9)。\n"
    (HERE / "depth_bandwidth.md").write_text(text)
    tables = ""
    for network in ("ubd", "closd"):
        tables += f"<h2>{network.upper()}</h2><div class='scroll'><table><tr>" + "".join(f"<th>{x}</th>" for x in columns[1:9]) + "</tr>"
        tables += "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells(r)[1:9]) + "</tr>" for r in rows if r["network"] == network) + "</table></div>"
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ForestColl · 深度与带宽</title><style>body{font:15px/1.7 system-ui,sans-serif;background:#f3f5f7;color:#23334a;margin:0}main{max-width:1250px;padding:28px;margin:28px auto;background:white;border-radius:10px}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #dce3e9;text-align:right}th{background:#eef3f7}.scroll{overflow:auto}img{width:100%}.note{padding:16px;background:#e8f1ee;border-left:4px solid #438368}a{color:#246790}li{margin:12px 0}</style><main><h1>ForestColl：深度与带宽分别记录</h1><p>ClosD + UBD · 64/128/256卡 × 坏0/1/2卡 · 未代入α、β</p><p class='note'>18/18 完整森林已验证。D：单阶段最大树深；2D：AllReduce最长依赖链；B∞：固定树与权重下、归约和广播重叠的带宽objective。</p>"""
    page += tables + "<h2>对比图</h2><p>虚线为历史MCNF数值，其中部分未通过原空间验收；具体状态见表。</p><img src='depth_bandwidth.png' alt='ClosD和UBD的深度及带宽objective随规模的变化'><h2>口径与复现</h2><ul>"
    page += "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul><p><a href='depth_bandwidth.csv'>CSV</a> · <a href='depth_bandwidth.json'>JSON与逐边瓶颈</a> · <a href='depth_bandwidth.md'>说明与重跑命令</a> · <a href='report.html'>上轮ClosD有限T报告</a></p></main></html>"
    (HERE / "depth_bandwidth.html").write_text(page)
    print("Wrote 18-case depth/bandwidth JSON, CSV, Markdown, chart and HTML")


if __name__ == "__main__":
    main()
