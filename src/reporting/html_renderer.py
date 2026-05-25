from __future__ import annotations

from typing import Any


_CSS = """
<style>
  :root { --bg:#f8f9fa; --card:#fff; --text:#212529; --muted:#6c757d; --green:#28a745; --red:#dc3545; --orange:#fd7e14; --blue:#007bff; }
  body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; background:var(--bg); color:var(--text); margin:0; padding:2rem; }
  h1 { font-size:1.5rem; margin-bottom:.5rem; }
  .subtitle { color:var(--muted); margin-bottom:1.5rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:1rem; margin-bottom:1.5rem; }
  .card { background:var(--card); border-radius:8px; padding:1rem; box-shadow:0 1px 3px rgba(0,0,0,.08); }
  .card .label { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .card .value { font-size:1.5rem; font-weight:600; margin-top:.25rem; }
  .section { background:var(--card); border-radius:8px; padding:1.25rem; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,.08); }
  .section h2 { font-size:1.1rem; margin:0 0 1rem; }
  table { width:100%; border-collapse:collapse; font-size:.9rem; }
  th,td { text-align:left; padding:.5rem .75rem; border-bottom:1px solid #e9ecef; }
  th { color:var(--muted); font-weight:500; }
  .bar-wrap { display:flex; align-items:center; gap:.75rem; }
  .bar-track { flex:1; height:18px; background:#e9ecef; border-radius:4px; overflow:hidden; display:flex; }
  .bar-seg { height:100%; }
  .bar-tp { background:var(--green); }
  .bar-fp { background:var(--red); }
  .bar-fn { background:var(--orange); }
  .bar-tn { background:var(--blue); }
  .legend { display:flex; gap:1rem; font-size:.8rem; color:var(--muted); margin-top:.5rem; }
  .legend span { display:inline-flex; align-items:center; gap:.25rem; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
</style>
"""


def render_analysis_report(results: list[dict[str, Any]], title: str = "Penalty Monitor 分析报告") -> str:
    parts = ["<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"/><title>", title, "</title>", _CSS, "</head><body>"]
    parts.append(f"<h1>{title}</h1>")
    parts.append(f"<div class='subtitle'>共分析 {len(results)} 次运行</div>")

    for r in results:
        parts.extend(_render_single_run(r))

    parts.append("</body></html>")
    return "".join(parts)


def _render_single_run(r: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    run_name = r["run_dir"].split("/")[-1] if "/" in r["run_dir"] else r["run_dir"]
    lines.append(f"<div class='section'><h2>运行: {run_name}</h2>")

    # 指标卡片
    lines.append("<div class='grid'>")
    lines.append(_card("事件总数", str(r["total_events"])))
    lines.append(_card("有标签事件", str(r["labeled_events"])))
    lines.append(_card("纸面订单", str(r["paper_orders"])))
    lines.append(_card("Precision", f"{(r['precision'] or 0):.2%}" if r["precision"] is not None else "-"))
    lines.append(_card("Recall", f"{(r['recall'] or 0):.2%}" if r["recall"] is not None else "-"))
    lines.append(_card("F1", f"{r['f1']:.4f}" if r["f1"] is not None else "-"))
    lines.append(_card("总 PnL", f"{r['total_pnl']:.4f}"))
    lines.append(_card("TP/FP/TN/FN", f"{r['true_positive']}/{r['false_positive']}/{r['true_negative']}/{r['false_negative']}"))
    lines.append("</div>")

    # 信号分布条形图
    lines.append("<h3>信号分布均值</h3>")
    lines.append("<table><thead><tr><th>信号</th><th>分布</th></tr></thead><tbody>")
    for field, vals in r["signal_analysis"].items():
        max_val = max(vals["tp_mean"], vals["fp_mean"], vals["fn_mean"], vals["tn_mean"], 0.01)
        tp_pct = vals["tp_mean"] / max_val * 100
        fp_pct = vals["fp_mean"] / max_val * 100
        fn_pct = vals["fn_mean"] / max_val * 100
        tn_pct = vals["tn_mean"] / max_val * 100
        lines.append("<tr>")
        lines.append(f"<td>{field}</td>")
        lines.append("<td>")
        lines.append("<div class='bar-track'>")
        if tp_pct > 0:
            lines.append(f"<div class='bar-seg bar-tp' style='width:{tp_pct:.1f}%'></div>")
        if fp_pct > 0:
            lines.append(f"<div class='bar-seg bar-fp' style='width:{fp_pct:.1f}%'></div>")
        if fn_pct > 0:
            lines.append(f"<div class='bar-seg bar-fn' style='width:{fn_pct:.1f}%'></div>")
        if tn_pct > 0:
            lines.append(f"<div class='bar-seg bar-tn' style='width:{tn_pct:.1f}%'></div>")
        lines.append("</div>")
        lines.append("</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    lines.append(
        "<div class='legend'>"
        "<span><i class='dot' style='background:var(--green)'></i> TP</span>"
        "<span><i class='dot' style='background:var(--red)'></i> FP</span>"
        "<span><i class='dot' style='background:var(--orange)'></i> FN</span>"
        "<span><i class='dot' style='background:var(--blue)'></i> TN</span>"
        "</div>"
    )

    # 预测原因分布
    if r.get("reason_distribution"):
        lines.append("<h3>预测原因分布</h3>")
        lines.append("<table><thead><tr><th>原因</th><th>次数</th></tr></thead><tbody>")
        for reason, count in sorted(r["reason_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"<tr><td>{reason}</td><td>{count}</td></tr>")
        lines.append("</tbody></table>")

    # 失败原因分布
    if r.get("failure_distribution"):
        lines.append("<h3>失败原因分布</h3>")
        lines.append("<table><thead><tr><th>原因</th><th>次数</th></tr></thead><tbody>")
        for failure, count in sorted(r["failure_distribution"].items(), key=lambda x: -x[1]):
            lines.append(f"<tr><td>{failure}</td><td>{count}</td></tr>")
        lines.append("</tbody></table>")

    lines.append("</div>")
    return lines


def _card(label: str, value: str) -> str:
    return f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div></div>"
