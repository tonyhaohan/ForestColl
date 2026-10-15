"""Revalidate saved forests and report WH1234 results beside current MCNF evidence."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clos_benchmark import HERE, graph_from_data, validate_trees


def load_rows(root):
    root = Path(root).resolve()
    queue = json.loads((root / "queue.json").read_text())
    inputs = HERE / "current_inputs.json"
    assert queue["source_sha256"]["benchmarks/current_inputs.json"] == hashlib.sha256(inputs.read_bytes()).hexdigest()
    references = json.loads(inputs.read_text())["rows"]
    rows = []
    for attempt in queue["rows"]:
        folder = root / attempt["case"]
        result = json.loads((folder / "result.json").read_text()) if (folder / "result.json").exists() else {}
        forest = result.get("forest", {})
        expected = next(r for r in references if f"{r['network']}{r['num_gpus']}_f{r['failure_count']}" == attempt["case"])
        ref = result.get("reference", expected)
        assert ref == expected
        mcnf = ref.get("mcnf", {})
        metrics = {}
        trees_hash = None
        if attempt["validated"]:
            path = folder / "trees.json"
            saved = json.loads(path.read_text())
            trees = {(t["root"], t["index"]): [[tuple(e) for e in p] for p in t["paths"]] for t in saved}
            counts = {(t["root"], t["index"]): t["chunks"] for t in saved}
            metrics, _ = validate_trees(graph_from_data(ref["graph"]), set(ref["graph"]["compute_nodes"]),
                                        trees, counts, forest["U"], forest["k"])
            assert abs(metrics["allgather_congestion_time"] - forest["allgather_optimum_time"]) < 1e-8
            trees_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"case": attempt["case"], "network": ref["network"], "num_gpus": ref["num_gpus"],
                     "failures": ref["failure_count"], "validated": attempt["validated"],
                     "error": attempt["error"], "depth_D": metrics.get("max_depth"),
                     "dependency_2D": metrics.get("allreduce_dependency_depth"),
                     "forest_B_inf": metrics.get("allreduce_fixed_forest_bandwidth_time"),
                     "forest_sequential_RS_AG": metrics.get("rs_ag_congestion_time"),
                     "forest_tree_count": metrics.get("tree_count"), "forest_k": forest.get("k"),
                     "java_search_sec": forest.get("search_sec") if forest.get("backend") == "java-parallel" else None,
                     "java_split_sec": forest.get("split_sec"), "java_branch_sec": forest.get("branch_sec"),
                     "java_wall_sec": forest.get("java_wall_sec"), "mapback_sec": forest.get("mapback_sec"),
                     "python_bound_check_sec": result.get("python_bound_check_sec"),
                     "validation_sec": result.get("validation_sec"),
                     "case_wall_sec": result.get("wall_sec"), "process_wall_sec": attempt["process_wall_sec"],
                     "java_peak_rss_mib": forest.get("java_peak_rss_mib"),
                     "mcnf_t12_objective": mcnf.get("t12_objective"),
                     "mcnf_t12_status": mcnf.get("t12_status"),
                     "mcnf_t12_max_violation": mcnf.get("t12_max_violation"),
                     "mcnf_t12_feasible_1e4": mcnf.get("t12_feasible_1e4"),
                     "mcnf_t12_historical_worker_sec": ref.get("mcnf_t12_timings", {}).get("worker_wall_time_sec"),
                     "mcnf_fixed_pattern_bound": mcnf.get("fixed_lower_bound"),
                     "mcnf_all_pattern_compact_bound": mcnf.get("compact_objective"),
                     "mcnf_compact_historical_call_sec": mcnf.get("compact_seconds"),
                     "mcnf_compact_status": mcnf.get("compact_status"),
                     "mcnf_compact_max_violation": mcnf.get("compact_max_violation"),
                     "input_sha256": result.get("input_sha256"), "trees_sha256": trees_hash,
                     "artifacts": os.path.relpath(folder, HERE)})
    return queue, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = HERE / "results" / args.label
    queue, rows = load_rows(root)
    assert len({r["case"] for r in rows}) == len(rows)
    if queue["complete"]:
        assert len(rows) == 27
    definitions = {
        "D": "Maximum GPU-to-GPU tree depth; switch hops do not add logical rounds.",
        "2D": "Longest reduce-then-broadcast dependency of these saved paired trees.",
        "B_inf": "max_e((L_AG(e)+L_AG(reverse(e)))/capacity(e)); M=N, each root contributes one unit.",
        "java_wall_sec": "JVM launch, input, search, switch splitting, tree generation, output; no Python validation.",
        "case_wall_sec": "Input loading, Python bound cross-check, JVM, mapback, validation, barrier diagnostic and writes; excludes imports.",
        "process_wall_sec": "Parent-observed Python process wall time including imports and all case work.",
        "mcnf_t12_historical_worker_sec": "Archived WH1234 worker time; HPN objective scaled from 200/50 to 4/1, time unchanged.",
        "mcnf_compact_historical_call_sec": "Archived WH1234 compact model build + solve + extraction; excludes task generation/imports/writes.",
    }
    (root / "comparison.json").write_text(json.dumps({"definitions": definitions,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "queue": queue, "rows": rows}, indent=2) + "\n")
    with (root / "comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for index, network in enumerate(("closd", "ubd", "hpn")):
        for f, color in enumerate(("#277da8", "#df8f28", "#569467")):
            part = sorted([r for r in rows if r["network"] == network and r["failures"] == f and r["validated"]], key=lambda r: r["num_gpus"])
            x = [r["num_gpus"] for r in part]
            axes[index, 0].plot(x, [r["depth_D"] for r in part], "o-", color=color, label=f"f{f}")
            axes[index, 1].plot(x, [r["forest_B_inf"] for r in part], "o-", color=color, label=f"Forest f{f}")
            axes[index, 1].plot(x, [r["mcnf_t12_objective"] for r in part], "x--", color=color, alpha=.6, label=f"T12 archive f{f}")
            axes[index, 2].plot(x, [r["process_wall_sec"] for r in part], "o-", color=color, label=f"f{f}")
        for column, title in enumerate(("ForestColl depth D", "B_inf and historical T12", "ForestColl process seconds")):
            ax = axes[index, column]
            ax.set_title(network.upper() + " · " + title)
            ax.set_ylim(bottom=0)
            ax.set_xticks([64, 128, 256])
            ax.set_xlabel("Active GPUs")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    passed = sum(r["validated"] for r in rows)
    notes = [
        f"WH1234（{queue['hostname']}）本轮 {passed}/{len(rows)} 个完整森林通过独立重放。上游 Java 三阶段算法未修改，common pool=4、Xmx4g；HPN 单向边容量 4/1。每例单次实测。",
        "ForestColl 等分片 AllGather 树与反向 ReduceScatter 配对：分别记录 D、2D、固定森林带宽极限 B∞ 和顺序 RS+AG。B∞不含启动延迟、有限分块的填充/排空；不等价于 MCNF T=12 的可行完成时间。",
        "当前主库的 HWC T12、固定模式下界、全模式紧凑LP均列入 JSON/CSV。主库结果来自已有 WH1234 归档；原状态与原空间最大违反保留，超过1e-4的T12数值不能用于严格胜负判断。",
        "HPN历史T12按200/50求解，目标乘50对齐当前4/1；耗时和残差保持历史实测。ForestColl直接用当前4/1物理图。两者物理资源和M=N一致，但候选树、路由和时间约束不同。",
        "Java秒包含JVM启动、读图、搜索、拆交换机、生成和输出；进程秒另含Python导入、独立带宽交叉检查、映射、校验和诊断。表中MCNF时间为历史worker时间；紧凑LP调用时间的范围更小，保留在CSV，不计算求解加速比。",
        "失败和超时也保留实际进程秒与日志。原始树及其哈希保留；集合迭代/并行调度可能让不同运行生成不同森林。未运行真实网络或GPU collective。",
    ]
    columns = [("case", "案例"), ("depth_D", "D"), ("dependency_2D", "2D"), ("forest_B_inf", "B∞"),
               ("mcnf_t12_objective", "HWC T12历史"), ("mcnf_t12_feasible_1e4", "T12≤1e-4"),
               ("mcnf_all_pattern_compact_bound", "全模式下界"), ("java_wall_sec", "Java秒"),
               ("process_wall_sec", "进程秒"), ("mcnf_t12_historical_worker_sec", "T12历史秒")]
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "是" if value else "否"
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    table = "| " + " | ".join(title for _, title in columns) + " |\n|" + "---|" * len(columns) + "\n"
    table += "\n".join("| " + " | ".join(cell(r[key]) for key, _ in columns) + " |" for r in rows)
    title = "ForestColl / MCNF：WH1234 的 ClosD、UBD、HPN 对比"
    markdown = "# " + title + "\n\n" + "\n\n".join(notes) + "\n\n" + table + "\n\n![对比](comparison.png)\n"
    (root / "README.md").write_text(markdown)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ForestColl WH1234</title><style>body{font:15px/1.6 system-ui;background:#f3f5f7;color:#24334a;margin:0}main{max-width:1450px;margin:auto;padding:28px;background:white}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:13px}td,th{padding:8px;text-align:right;border-bottom:1px solid #dce3e9}th{background:#eef3f7}.scroll{overflow:auto}img{width:100%}li{margin:10px 0}a{color:#246790}</style><main>"
    page += f"<h1>{html.escape(title)}</h1><p>已完成 {len(rows)}/27 例 · {passed}/{len(rows)} 完整森林通过校验 · 64/128/256 GPU × 坏0/1/2卡</p>"
    for network in ("hpn", "closd", "ubd"):
        page += f"<h2>{network.upper()}</h2><div class='scroll'><table><tr>" + "".join(f"<th>{title}</th>" for _, title in columns) + "</tr>"
        page += "".join("<tr>" + "".join(f"<td>{html.escape(cell(r[key]))}</td>" for key, _ in columns) + "</tr>" for r in rows if r["network"] == network)
        page += "</table></div>"
    page += "<h2>深度、带宽与计算时间</h2><img src='comparison.png' alt='三种拓扑的深度、带宽及计算时间'><h2>口径</h2><ul>"
    page += "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    page += "</ul><p><a href='comparison.csv'>CSV</a> · <a href='comparison.json'>JSON与来源</a> · <a href='README.md'>Markdown</a> · <a href='queue.json'>逐例命令和状态</a></p></main></html>"
    (root / "comparison.html").write_text(page)
    print(f"Report: {passed}/{len(rows)} validated; queue_complete={queue['complete']}")


if __name__ == "__main__":
    main()
