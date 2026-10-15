"""Build a report from fixed-forest scheduling checkpoints, without solving."""
import argparse
import csv
import html
import json
from pathlib import Path


def build_report(root):
    root = Path(root)
    queue = json.loads((root / "queue.json").read_text())
    rows = []
    for attempt in queue["rows"]:
        path = root / attempt["result"] if attempt["result"] else None
        data = json.loads(path.read_text()) if path and path.exists() else {}
        ref = data.get("reference", {})
        historical = ref.get("mcnf", {})
        horizon = attempt["time_steps"]
        rows.append({"case": attempt["case"], "T": horizon,
            "status": data.get("status", attempt.get("error") or "worker_failed"),
            "validated": data.get("validated", False), "objective": data.get("objective"),
            "wave_objective": data.get("wave_objective"),
            "B_inf": data.get("forest", {}).get("allreduce_fixed_forest_bandwidth_time"),
            "paired_min_T": data.get("minimum_paired_horizon"),
            "lp_status": data.get("lp", {}).get("status"),
            "lp_accepted_1e4": data.get("lp", {}).get("accepted_1e4"),
            "mcnf_T12_objective": historical.get("t12_objective"),
            "mcnf_T12_accepted_1e4": historical.get("t12_feasible_1e4"),
            "same_T_reference": horizon == 12,
            "process_wall_sec": attempt["process_wall_sec"],
            "timings": data.get("timings", {}),
            "source_trees_sha256": data.get("source_trees_sha256"),
            "plan": str(path.parent.relative_to(root) / data["plan"]) if data.get("plan") else None,
            "error": attempt.get("error") or ("worker_exit_" + str(attempt["exit_code"]) if attempt["exit_code"] not in (0, None) else None)})
    notes = [
        "M=N；单向物理边容量与当前输入一致（HPN 4/1），beta=1、alpha=0；目标是各轮持续时间之和。交换机沿固定路径流式转发，不另加逻辑时间步。",
        "每个 ForestColl 分片在原树反向归约，再沿同一棵树广播；分片比例和物理路径完全冻结。每轮数据仅能在下一轮复用，允许不同树、分片及归约/广播阶段重叠。",
        "T=12 为严格同轮数参考；2D/4D 为扩展时间步敏感性实验。MCNF历史值仍为T12，扩展行不计算同T胜负。历史原空间违反超过1e-4的值不作为可行上界。",
        "成对固定分片在深度D的树上需要至少2D轮。T不足时保留不可行证据，不剔除树、不重新分配权重。",
        "wave方案按相邻轮启动分片包；LP使用冻结的canonical node-arc模型，只优化固定方案的时序。模型超出变量守卫时仍导出并验证wave方案，不能称为LP最优。",
        "plan.json.gz包含精确操作、数据区间、依赖、路径与轮时长。wave包循环是完整的压缩操作描述；LP使用逐轮FIFO区间。尚未部署为MSCCL/NCCL硬件程序。",
        "计算秒为WH1234本轮生成/求解/校验/导出耗时，与通信目标分开。原始z、solver.log及大计划文件留在服务器。",
    ]
    if any(row["validated"] for row in rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, network in zip(axes, ("hpn", "closd", "ubd")):
            cases = sorted({row["case"] for row in rows if row["case"].startswith(network)})
            for case in cases:
                part = sorted([r for r in rows if r["case"] == case and r["validated"]], key=lambda r: r["T"])
                if part:
                    ax.plot([r["T"] for r in part], [r["objective"] for r in part], "o-", label=case)
            ax.set(title=network.upper(), xlabel="Time steps T", ylabel="Communication objective", ylim=(0, None))
            if ax.lines:
                ax.legend(fontsize=6)
        fig.savefig(root / "comparison.png", dpi=150)
        plt.close(fig)
    payload = {"queue": queue, "notes": notes, "rows": rows}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    if rows:
        with (root / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**row, "timings": json.dumps(row["timings"], sort_keys=True)} for row in rows)
    columns = [("case", "案例"), ("T", "T"), ("status", "状态"), ("validated", "方案校验"),
               ("objective", "通信目标"), ("wave_objective", "wave目标"), ("B_inf", "带宽极限"),
               ("lp_status", "LP状态"), ("mcnf_T12_objective", "MCNF历史T12"),
               ("mcnf_T12_accepted_1e4", "历史≤1e-4"), ("same_T_reference", "相同T"),
               ("process_wall_sec", "本轮计算秒")]
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "是" if value else "否"
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ForestColl固定森林调度</title><style>body{font:15px/1.65 system-ui;margin:24px;color:#203248}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f7}.scroll{overflow:auto}a{color:#176f9b}</style>"
    page += f"<h1>ForestColl：固定森林的有限轮调度</h1><p>队列状态：{html.escape(queue['status'])} · 已记录 {len(rows)} 个时间步实验 · 待处理 {len(queue['pending_cases'])} 例</p>"
    page += "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul>"
    if (root / "comparison.png").exists():
        page += "<img src='comparison.png' style='max-width:100%' alt='Communication objective versus time steps for fixed ForestColl forests'>"
    page += "<div class='scroll'><table><tr>" + "".join(f"<th>{title}</th>" for _, title in columns) + "<th>导出方案</th></tr>"
    for row in rows:
        page += "<tr>" + "".join(f"<td>{html.escape(cell(row[key]))}</td>" for key, _ in columns)
        page += f"<td><a href='{html.escape(row['plan'])}'>JSON.gz</a></td></tr>" if row["plan"] else "<td>—</td></tr>"
    page += "</table></div><p><a href='comparison.json'>JSON及各阶段计时</a> · <a href='comparison.csv'>CSV</a></p></html>"
    (root / "comparison.html").write_text(page)
    (root / "README.md").write_text("# ForestColl fixed-forest schedules\n\n" + "\n\n".join(notes) + "\n\nSee comparison.html, comparison.json and comparison.csv.\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build_report(parser.parse_args().root)
