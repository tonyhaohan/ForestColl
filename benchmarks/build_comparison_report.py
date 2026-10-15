"""Join this run's forest/schedule checkpoints with frozen main timing evidence."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
import math
from pathlib import Path

from build_wh1234_report import HERE, load_rows

REPORT_DEFAULTS = {"data_per_gpu_MB": .25, "bytes_per_MB": 1_000_000,
                   "unit_link_Gbps": 400, "alpha_us_per_active_step": .1}


def bandwidth_us(objective, num_gpus):
    if objective is None:
        return None
    per_gpu_us = REPORT_DEFAULTS["data_per_gpu_MB"] * REPORT_DEFAULTS["bytes_per_MB"] * 8 / (REPORT_DEFAULTS["unit_link_Gbps"] * 1000)
    return objective * per_gpu_us / num_gpus


def runtime_us(objective, durations, num_gpus):
    if objective is None or durations is None:
        return None, None
    assert durations and all(math.isfinite(v) and v >= 0 for v in durations)
    assert abs(sum(durations) - objective) <= 1e-7 * max(1, abs(objective))
    active = sum(v > 0 for v in durations)
    # M=N，先按每卡数据量缩放带宽项，再按实际非空轮计启动开销。
    return bandwidth_us(objective, num_gpus) + REPORT_DEFAULTS["alpha_us_per_active_step"] * active, active


def add_runtime_values(rows, schedule):
    reference_path = HERE / "main_runtime_reference.json"
    reference = json.loads(reference_path.read_text())
    assert reference["current_inputs_sha256"] == hashlib.sha256((HERE / "current_inputs.json").read_bytes()).hexdigest()
    main = {r["case"]: r for r in reference["rows"]}
    for row in rows:
        old = main[row["case"]]
        assert old["source_sha256"] == row["main_T12_source_sha256"]
        a = old["A"]
        assert len(a) == 12
        row["main_T12_A"] = a
        row["main_T12_runtime_us"], row["main_T12_active_steps"] = runtime_us(row["main_T12_objective"], a, row["num_gpus"])
        row["forest_B_inf_bandwidth_us"] = bandwidth_us(row["B_inf"], row["num_gpus"])
        a, plan_hash = None, None
        if row["result_4D"] and row["validated_4D"]:
            result_path = Path(schedule) / row["result_4D"]
            result = json.loads(result_path.read_text())
            plan_path = result_path.parent / result["plan"]
            if plan_path.exists():
                with plan_path.open("rb") as handle:
                    plan_hash = hashlib.file_digest(handle, "sha256").hexdigest()
                expected = result["lp"]["plan_sha256"] if result["status"] == "lp_feasible" else result["wave_plan_sha256"]
                assert plan_hash == expected, f"Changed exported plan for {row['case']}"
                with gzip.open(plan_path, "rt") as handle:
                    plan = json.load(handle)
                assert plan["time_steps"] == row["T_4D"]
                a = plan["durations"]
                assert len(a) == row["T_4D"]
        row["forest_4D_A"] = a
        row["forest_4D_plan_sha256"] = plan_hash
        row["forest_4D_runtime_us"], row["forest_4D_active_steps"] = runtime_us(row["objective_4D"], a, row["num_gpus"])
    return hashlib.sha256(reference_path.read_bytes()).hexdigest()


def plot_runtime(axes, rows):
    for column, network in enumerate(("hpn", "closd", "ubd")):
        part = [r for r in rows if r["network"] == network]
        labels = [f"{r['num_gpus']}/f{r['failures']}" for r in part]
        ax = axes[column]
        for key, label, style, color in (
            ("forest_B_inf_bandwidth_us", "Forest B_inf (bandwidth only)", "o--", "0.55"),
            ("forest_4D_runtime_us", "Forest T=4D (feasible upper bound)", "o-", "tab:green"),
            ("main_T12_runtime_us", "Ours T=12 (archived schedule)", "-", "tab:blue")):
            ax.plot(labels, [r[key] if r[key] is not None else float("nan") for r in part], style, color=color, label=label)
        accepted = [i for i, r in enumerate(part) if r["main_T12_accepted_1e4"]]
        rejected = [i for i, r in enumerate(part) if not r["main_T12_accepted_1e4"]]
        ax.scatter(accepted, [part[i]["main_T12_runtime_us"] for i in accepted], marker="s", color="tab:blue", s=25)
        if rejected:
            ax.scatter(rejected, [part[i]["main_T12_runtime_us"] for i in rejected], marker="x", color="tab:red", s=40, label="Ours: fails 1e-4 feasibility gate")
        ax.set(title=network.upper(), ylabel="End-to-end communication time (us)", yscale="log")
        ax.grid(axis="y", which="both", alpha=.2)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=7)


def collect_rows(source, schedule):
    source, schedule = Path(source), Path(schedule)
    synthesis_queue, forests = load_rows(source)
    queue = json.loads((schedule / "queue.json").read_text())
    inputs = HERE / "current_inputs.json"
    assert queue["source_sha256"]["benchmarks/current_inputs.json"] == hashlib.sha256(inputs.read_bytes()).hexdigest()
    refs = json.loads(inputs.read_text())["rows"]
    forests = {row["case"]: row for row in forests}
    attempts = {(a["case"], a["time_steps"]): a for a in queue["rows"]}
    source_failures = {a["case"]: a.get("error") for a in queue["rows"] if a["time_steps"] is None}
    rows = []
    for ref in sorted(refs, key=lambda r: (r["network"], r["num_gpus"], r["failure_count"])):
        name = f"{ref['network']}{ref['num_gpus']}_f{ref['failure_count']}"
        forest = forests.get(name, {})
        depth = forest.get("depth_D")
        main = ref["mcnf"]
        main_times = ref.get("mcnf_t12_timings", {})
        row = {"case": name, "network": ref["network"], "num_gpus": ref["num_gpus"],
               "failures": ref["failure_count"], "D": depth, "B_inf": forest.get("forest_B_inf"),
               "synthesis_status": ("validated" if forest.get("validated") else forest.get("error") or "pending"),
               "synthesis_process_sec": forest.get("process_wall_sec"),
               "java_search_sec": forest.get("java_search_sec"), "java_split_sec": forest.get("java_split_sec"),
               "java_branch_sec": forest.get("java_branch_sec"), "java_wall_sec": forest.get("java_wall_sec"),
               "mapback_sec": forest.get("mapback_sec"), "python_bound_check_sec": forest.get("python_bound_check_sec"),
               "synthesis_validation_sec": forest.get("validation_sec"), "synthesis_case_sec": forest.get("case_wall_sec"),
               "trees_sha256": forest.get("trees_sha256"),
               "main_T12_objective": main.get("t12_objective"), "main_T12_status": main.get("t12_status"),
               "main_T12_accepted_1e4": main.get("t12_feasible_1e4"),
               "main_T12_worker_sec": main_times.get("worker_wall_time_sec"),
               "main_T12_problem_build_sec": main_times.get("problem_build_wall_time_sec"),
               "main_T12_model_build_sec": main_times.get("model_build_wall_time_sec"),
               "main_T12_solve_call_sec": main_times.get("main_wall_time_sec"),
               "main_T12_timings": main_times, "main_T12_source": main.get("t12_source"),
               "main_T12_source_sha256": ref.get("t12_source_sha256"),
               "main_compact_call_sec": main.get("compact_seconds"),
               "main_compact_build_sec": main.get("compact_model_build_seconds"),
               "main_compact_objective": main.get("compact_objective"),
               "main_compact_status": main.get("compact_status"),
               "main_compact_max_violation": main.get("compact_max_violation"),
               "main_compact_source": main.get("compact_source")}
        for multiplier in (2, 4):
            suffix = f"{multiplier}D"
            horizon = multiplier * depth if depth is not None else None
            attempt = attempts.get((name, horizon), {}) if horizon is not None else {}
            path = schedule / attempt["result"] if attempt.get("result") else None
            data = json.loads(path.read_text()) if path and path.exists() else {}
            if data:
                assert data["source_trees_sha256"] == forest["trees_sha256"], f"Different forest for {name}"
                assert data["time_steps"] == horizon
                assert data["reference"] == ref
                assert abs(data["forest"]["allreduce_fixed_forest_bandwidth_time"] - row["B_inf"]) < 1e-8
            lp = data.get("lp", {})
            objective = data.get("objective") if data.get("validated") else None
            lp_valid = bool(lp.get("accepted_1e4") and lp.get("export_validation", {}).get("validated"))
            if data.get("status") == "lp_feasible" and not lp_valid:
                objective = None
            kind = "missing"
            if objective is not None:
                kind = ("lp_optimal" if lp.get("status") == "optimal" else "lp_feasible_upper_bound") if data.get("status") == "lp_feasible" and lp_valid else "wave_upper_bound"
            active = queue.get("current_case") == name and queue.get("current_horizon") == horizon
            exit_error = f"worker_exit_{attempt['exit_code']}" if attempt.get("exit_code") not in (None, 0) else None
            status = (attempt.get("error") or exit_error or data.get("status") or source_failures.get(name)
                      or ("running" if active else "pending"))
            process_sec = attempt.get("process_wall_sec")
            total = row["synthesis_process_sec"] + process_sec if row["synthesis_process_sec"] is not None and process_sec is not None else None
            row.update({f"T_{suffix}": horizon, f"objective_{suffix}": objective,
                        f"objective_kind_{suffix}": kind, f"status_{suffix}": status,
                        f"validated_{suffix}": bool(data.get("validated")),
                        f"wave_objective_{suffix}": data.get("wave_objective"),
                        f"lp_objective_{suffix}": lp.get("objective") if lp_valid else None,
                        f"lp_status_{suffix}": lp.get("status"), f"lp_{suffix}": lp,
                        f"schedule_process_sec_{suffix}": process_sec,
                        f"total_process_sec_{suffix}": total, f"timings_{suffix}": data.get("timings", {}),
                        f"result_{suffix}": attempt.get("result")})
            worker = row["main_T12_worker_sec"]
            row[f"total_to_main_worker_ratio_{suffix}"] = total / worker if total is not None and worker else None
        worker = row["main_T12_worker_sec"]
        row["synthesis_to_main_worker_ratio"] = row["synthesis_process_sec"] / worker if row["synthesis_process_sec"] is not None and worker else None
        rows.append(row)
    return synthesis_queue, queue, rows


def build_report(source, schedule, output):
    synthesis, queue, rows = collect_rows(source, schedule)
    runtime_reference_hash = add_runtime_values(rows, schedule)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    notes = [
        "端到端通信时间与计算耗时图的纵轴统一使用log坐标（底数10）。缺失项保留空值，B∞仍仅为带宽项。",
        f"默认端到端报告口径：每卡{REPORT_DEFAULTS['data_per_gpu_MB']:g} MB（十进制{REPORT_DEFAULTS['data_per_gpu_MB'] * REPORT_DEFAULTS['bytes_per_MB']:g} bytes），单位链路{REPORT_DEFAULTS['unit_link_Gbps']:g} Gbps，每活跃步启动开销{REPORT_DEFAULTS['alpha_us_per_active_step']:g} μs；原始M=N，故时间(μs)=objective×{bandwidth_us(1, 1):g}/N+{REPORT_DEFAULTS['alpha_us_per_active_step']:g}×活跃步数。活跃步按保存的A_t>0计数，空轮不收费；保留拓扑原有容量比例，例如HPN的4/1对应{4 * REPORT_DEFAULTS['unit_link_Gbps']:g}/{REPORT_DEFAULTS['unit_link_Gbps']:g} Gbps。",
        "图中隐藏2D，展示ForestColl的4D具体方案与我们的历史T12方案；仅在已有调度上换算，未针对新alpha重新优化。B∞虚线仅含带宽项，不是完整端到端时间。红叉标记main历史结果未通过原空间1e-4门槛，不能当作已验收的可执行方案。",
        "每例并列报告 B∞、T=2D 和 T=4D 的线性 objective；D 是固定森林最大单阶段 GPU 逻辑树深，交换机不另计层数。树、分片比例和物理路径固定；M=N、beta=1、alpha=0，HPN单向容量4/1。",
        "B∞ 是固定森林归约与广播合计物理边负载/容量的最大值。有限T目标是各轮持续时间之和。LP最优与wave可行上界明确分开；规模守卫、超时、未完成保留状态和空值，不能把wave写成LP最优值。",
        "ForestColl合成秒为包含Python/JVM启动、树生成、映射与校验的进程时间；2D/4D新增秒各自包含调度生成、可选LP求解、校验与导出。单个T的总秒=合成秒+该T新增秒；合成仅计一次，两个T的总秒不能相加。",
        "main T12秒是历史WH1234 worker总时间；main compact是去掉时序约束的全模式恒定流紧凑LP，用于带宽下界，其计算秒为建模+求解+提取调用时间，已包含建模秒，不是通信执行时间。候选树、T、后端和计时范围不同，耗时比只描述这批历史记录，不表示同算法同口径求解加速比。",
        "main记录仍保留T12原空间1e-4验收状态。HPN旧200/50结果目标乘50对齐4/1，历史计算耗时不缩放。旧T12 objective是附加参考，不与2D/4D作同T胜负判断。",
        "main阶段时间可能嵌套，不能逐项全部相加；缺失阶段保持空值。失败/超时的实际进程时间仍计入。所有时间为本轮或历史单次测量，单位秒；详细阶段计时、原始LP状态及来源在JSON/CSV。",
    ]
    stats = {}
    for network in ("hpn", "closd", "ubd"):
        part = [r for r in rows if r["network"] == network and r["synthesis_to_main_worker_ratio"] is not None]
        stats[network] = {"timed_cases": len(part),
                          "synthesis_less_than_main_worker": sum(r["synthesis_to_main_worker_ratio"] < 1 for r in part),
                          "synthesis_greater_than_main_worker": sum(r["synthesis_to_main_worker_ratio"] > 1 for r in part)}
    metadata = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_reference_sha256": runtime_reference_hash,
                "time_axis_scale": "log",
                "main_reference_revision": json.loads((HERE / "current_inputs.json").read_text())["source_revision"],
                "source_label": Path(source).name, "schedule_label": Path(schedule).name,
                "synthesis_complete": synthesis["complete"], "schedule_complete": queue["complete"],
                "synthesis_validated": sum(r["synthesis_status"] == "validated" for r in rows),
                "finite_values": sum(r[f"objective_{s}"] is not None for r in rows for s in ("2D", "4D")),
                "forest_4D_runtime_values": sum(r["forest_4D_runtime_us"] is not None for r in rows),
                "lp_optimal_values": sum(r[f"objective_kind_{s}"] == "lp_optimal" for r in rows for s in ("2D", "4D"))}
    payload = {"metadata": metadata, "notes": notes, "timing_summary": stats,
               "runtime_contract": {**REPORT_DEFAULTS,
                                    "active_step_rule": "A_t > 0", "reoptimized_for_alpha": False},
               "synthesis_queue": synthesis, "schedule_queue": queue, "rows": rows}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()} for row in rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runtime_title = f"{REPORT_DEFAULTS['data_per_gpu_MB']:g} MB per GPU | unit link: {REPORT_DEFAULTS['unit_link_Gbps']:g} Gbps | startup: {REPORT_DEFAULTS['alpha_us_per_active_step']:g} us per active step"
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    plot_runtime(axes[0], rows)
    fig.suptitle(runtime_title, fontsize=13)
    for column, network in enumerate(("hpn", "closd", "ubd")):
        part = [r for r in rows if r["network"] == network]
        labels = [f"{r['num_gpus']}/f{r['failures']}" for r in part]
        for key, label in (("synthesis_process_sec", "Forest synthesis"), ("total_process_sec_4D", "Forest + T=4D"), ("main_T12_worker_sec", "Ours T12 worker (archive)"), ("main_compact_call_sec", "All-pattern bandwidth LP (archive)")):
            axes[1, column].plot(labels, [r[key] if r[key] is not None else float("nan") for r in part], ".-", label=label)
        ax = axes[1, column]
        ax.set(title=network.upper(), ylabel="Observed calculation seconds", yscale="log")
        ax.grid(axis="y", which="both", alpha=.2)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=7)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    plot_runtime(axes, rows)
    fig.suptitle(runtime_title, fontsize=13)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"end_to_end.{extension}", dpi=200)
    plt.close(fig)

    runtime_columns = [("case", "案例"), ("forest_B_inf_bandwidth_us", "Forest B∞纯带宽μs"),
        ("T_4D", "Forest T=4D"), ("forest_4D_active_steps", "Forest活跃步"), ("forest_4D_runtime_us", "Forest端到端μs"),
        ("main_T12_active_steps", "我们T12活跃步"), ("main_T12_runtime_us", "我们T12端到端μs"), ("main_T12_accepted_1e4", "我们历史≤1e-4")]
    objective_columns = [("case", "案例"), ("D", "D"), ("B_inf", "B∞"),
        ("T_2D", "T=2D"), ("objective_2D", "objective(2D)"), ("objective_kind_2D", "2D值性质"), ("lp_status_2D", "2D LP状态"),
        ("T_4D", "T=4D"), ("objective_4D", "objective(4D)"), ("objective_kind_4D", "4D值性质"), ("lp_status_4D", "4D LP状态"),
        ("status_2D", "2D运行状态"), ("status_4D", "4D运行状态"), ("main_T12_objective", "main历史T12"), ("main_T12_accepted_1e4", "历史≤1e-4")]
    timing_columns = [("case", "案例"), ("synthesis_process_sec", "Forest合成秒"),
        ("schedule_process_sec_2D", "2D新增秒"), ("total_process_sec_2D", "合成+2D秒"),
        ("schedule_process_sec_4D", "4D新增秒"), ("total_process_sec_4D", "合成+4D秒"),
        ("main_T12_worker_sec", "main T12 worker秒"), ("main_compact_call_sec", "main compact调用秒"),
        ("synthesis_to_main_worker_ratio", "合成/main比"), ("total_to_main_worker_ratio_2D", "合成+2D/main比"), ("total_to_main_worker_ratio_4D", "合成+4D/main比")]
    stage_columns = [("case", "案例"), ("java_search_sec", "Java搜索秒"), ("java_split_sec", "拆交换机秒"),
        ("java_branch_sec", "构树秒"), ("java_wall_sec", "Java总秒"), ("mapback_sec", "映射秒"),
        ("python_bound_check_sec", "带宽校验秒"), ("synthesis_validation_sec", "树校验秒"),
        ("main_T12_problem_build_sec", "main任务构造秒"), ("main_T12_model_build_sec", "main建模秒"),
        ("main_T12_solve_call_sec", "main求解调用秒"), ("main_compact_build_sec", "compact建模秒")]
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, float):
            return f"{value:.6g}"
        return {"lp_optimal": "LP最优", "lp_feasible_upper_bound": "LP可行上界", "wave_upper_bound": "wave可行上界", "missing": "缺失"}.get(str(value), str(value))
    def table(part, columns):
        return "<div class='scroll'><table><tr>" + "".join(f"<th>{title}</th>" for _, title in columns) + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(cell(r[key]))}</td>" for key, _ in columns) + "</tr>" for r in part) + "</table></div>"
    title = "ForestColl / ours：端到端时间与计算耗时"
    status = f"27例全部列出 · 森林校验 {metadata['synthesis_validated']}/27 · 有限T有效值 {metadata['finite_values']}/54（LP最优 {metadata['lp_optimal_values']} 项，其余有效值为可行上界）"
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ForestColl / main 比较</title><style>body{font:15px/1.6 system-ui;margin:24px;color:#203248}h1{font-size:28px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f7;position:sticky;top:0}img{max-width:100%}li{margin:6px 0}a{color:#176f9b}</style>"
    page += f"<h1>{title}</h1><p>{status}</p><p>快照：{metadata['generated_at']} · 合成队列完成：{cell(synthesis['complete'])} · 调度队列完成：{cell(queue['complete'])}</p>"
    page += "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul><img src='comparison.png' alt='Three topologies: end-to-end microseconds and calculation seconds'><p>单独导出端到端图：<a href='end_to_end.png'>PNG</a> · <a href='end_to_end.pdf'>PDF</a> · <a href='end_to_end.svg'>SVG</a></p>"
    for network in ("hpn", "closd", "ubd"):
        part = [r for r in rows if r["network"] == network]
        page += f"<h2>{network.upper()}：端到端时间（μs）</h2>" + table(part, runtime_columns)
        page += f"<h2>{network.upper()}：通信目标</h2>" + table(part, objective_columns)
        page += f"<h2>{network.upper()}：计算耗时对比（秒）</h2>" + table(part, timing_columns)
    page += "<details><summary>全部案例的阶段耗时（嵌套阶段不重复相加）</summary>" + table(rows, stage_columns) + "</details>"
    page += "<p><a href='comparison.csv'>完整CSV</a> · <a href='comparison.json'>JSON、逐阶段计时与来源</a> · <a href='README.md'>Markdown</a></p></html>"
    (output / "comparison.html").write_text(page)
    markdown = f"# {title}\n\n{status}\n\n" + "\n\n".join(notes) + "\n\n"
    for caption, columns in (("端到端时间（μs）", runtime_columns), ("通信目标", objective_columns), ("计算耗时（秒）", timing_columns)):
        markdown += f"## {caption}\n\n| " + " | ".join(t for _, t in columns) + " |\n|" + "---|" * len(columns) + "\n"
        markdown += "\n".join("| " + " | ".join(cell(r[k]) for k, _ in columns) + " |" for r in rows) + "\n\n"
    (output / "README.md").write_text(markdown + "![对比图](comparison.png)\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.source, args.schedule, args.out)
    print(json.dumps(report["metadata"]))
