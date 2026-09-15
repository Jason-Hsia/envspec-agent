# -*- coding: utf-8 -*-
"""评测执行入口。输出 out/eval_report.json + 控制台摘要。"""
import io
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.loop import EnvSpecAgent
from src.config import DATA_DIR, GATES, OUT_DIR
from src.tools.registry import EnvSpecTools
from . import metrics as M


def load_golden() -> List[Dict]:
    raw = json.loads((DATA_DIR / "golden_set.json").read_text(encoding="utf-8"))
    return raw["items"]


def run(verbose: bool = True) -> Dict:
    items = load_golden()
    tools = EnvSpecTools()
    agent = EnvSpecAgent(tools)

    stale_ids = {s["std_id"] for s in tools.index["standards"] if s["status"] == "superseded"}
    per_case: List[Dict] = []

    for it in items:
        res = agent.run(it["query"])
        retrieved_ids = [h["clause_id"] for h in
                         next((o["output"]["hits"] for o in res["observations"]
                               if o["tool"] == "search_standard"), [])]
        # 检索层同时看"工具返回的限值来源"——限值路径不走 search_standard
        for lim in res["limits"]:
            retrieved_ids.append(f"{lim['std_id']}#{lim['table']}")

        relevant = set(it.get("expected_clause_ids") or [])
        rm = M.retrieval_metrics(retrieved_ids, relevant)

        row = {
            "id": it["id"], "category": it["category"], "query": it["query"],
            "refused": res["refused"], "should_refuse": it["should_refuse"],
            "n_citations": len(res["citations"]),
            "plan": [p["tool"] for p in res["plan"]],
            "expected_tool": it.get("expected_tool", ""),
            "stale_warnings": res["stale_warnings"],
            # L1
            **rm,
            # L2
            "numerical_accuracy": M.numerical_accuracy(res["answer"], it.get("expected_numbers", [])),
            "key_point_coverage": M.key_point_coverage(res["answer"], it.get("key_points", [])),
            "conclusion_accuracy": M.conclusion_accuracy(res["answer"], it.get("expected_numbers", [])),
            # 正确拒答时本就不该有引用 —— 记 N/A 而非 0 分，否则拒答题会系统性拉低引用精度。
            # 判据是"是否应当作答"，而不是"有没有引用"，避免把漏引误判成 N/A。
            "citation_precision": (
                M.citation_precision(res["citations"], it.get("must_not_cite", []),
                                     set(it.get("expected_std_ids") or []))
                if res["citations"] else (None if it["should_refuse"] else 0.0)),
            "citation_recall": M.citation_recall(res["citations"], set(it.get("expected_std_ids") or [])),
            # L3
            "tool_selection": M.tool_selection_accuracy(res["plan"], it.get("expected_tool", "")),
            "tool_success_rate": M.tool_success_rate(res["tool_calls"]),
            "tool_efficiency": M.tool_efficiency(res["tool_calls"]),
            "n_tool_calls": len(res["tool_calls"]),
            # L4
            "stale_citation": M.stale_citation_rate(res["citations"], stale_ids),
            "refusal_correct": M.refusal_correct(res["refused"], it["should_refuse"]),
        }
        per_case.append(row)

        if verbose:
            mark = "OK " if row["refusal_correct"] else "NG "
            st = "STALE!" if row["stale_citation"] else ""
            print(f"{mark}{it['id']} [{it['category']:<18}] "
                  f"na={row['numerical_accuracy']} kp={row['key_point_coverage']} {st}")

    agg = _aggregate(per_case)
    gate = M.end_to_end_pass(agg, GATES)

    report = {"aggregate": agg, "gates": GATES, "gate_result": gate, "cases": per_case}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        _print_summary(agg, gate)
    return report


def _aggregate(cases: List[Dict]) -> Dict:
    keys = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@10",
            "numerical_accuracy", "key_point_coverage", "conclusion_accuracy",
            "citation_precision", "citation_recall", "tool_selection",
            "tool_success_rate", "tool_efficiency", "refusal_correct"]
    agg = {k: M.mean([c.get(k) for c in cases]) for k in keys}
    agg["stale_citation_rate"] = (
        sum(c["stale_citation"] for c in cases) / len(cases) if cases else 0.0)
    agg["avg_tool_calls"] = M.mean([c["n_tool_calls"] for c in cases])
    agg["n_cases"] = len(cases)

    # 分类别切片：这是定位问题的关键视图
    by_cat: Dict[str, Dict] = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)
    agg["by_category"] = {
        cat: {
            "n": len(v),
            "numerical_accuracy": M.mean([x.get("numerical_accuracy") for x in v]),
            "key_point_coverage": M.mean([x.get("key_point_coverage") for x in v]),
            "conclusion_accuracy": M.mean([x.get("conclusion_accuracy") for x in v]),
            "stale_citation_rate": sum(x["stale_citation"] for x in v) / len(v),
            "refusal_correct": M.mean([x.get("refusal_correct") for x in v]),
            "recall@5": M.mean([x.get("recall@5") for x in v]),
        } for cat, v in by_cat.items()
    }
    return agg


def _disp_width(s: str) -> int:
    """字符串的显示宽度：CJK 与全角符号占两列。

    f-string 的 `:<22` 按字符数补空格，中英文混排的指标名会歪掉。
    这份报告是给人看的交付物，对齐得自己算。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _print_summary(agg: Dict, gate: Dict):
    lines: List[str] = []
    add = lines.append

    add("=" * 62)
    add("汇总指标")
    add("=" * 62)
    labels = {
        "recall@5": "L1 检索 Recall@5", "mrr": "L1 检索 MRR",
        "ndcg@10": "L1 检索 nDCG@10",
        "key_point_coverage": "L2 要点覆盖率", "numerical_accuracy": "L2 数值准确率",
        "conclusion_accuracy": "L2 结论一致率",
        "citation_precision": "L2 引用精度", "citation_recall": "L2 引用召回",
        "tool_selection": "L3 工具选择准确率", "tool_success_rate": "L3 工具成功率",
        "tool_efficiency": "L3 工具调用效率",
        "stale_citation_rate": "L4 时效性违规率 ↓", "refusal_correct": "L4 拒答正确率",
    }
    for k, lab in labels.items():
        v = agg.get(k)
        if v is None:
            continue
        add(f"  {_pad(lab, 26)} {v:.3f}")

    add("")
    add("分类别切片")
    for cat, st in sorted(agg["by_category"].items()):
        add(f"  {_pad(cat, 20)} n={st['n']:<3} 数值准确率={_f(st['numerical_accuracy'])} "
            f"要点覆盖={_f(st['key_point_coverage'])} 时效违规={st['stale_citation_rate']:.2f} "
            f"拒答正确={_f(st['refusal_correct'])}")

    add("")
    add("发布门禁: " + ("✅ 通过" if gate["passed"] else "❌ 未通过")
        + f"（实际校验 {gate.get('n_evaluated', '?')}/{gate.get('n_gates', len(GATES))} 项）")
    for f in gate["failures"]:
        add("   ✗ " + f)

    report = "\n".join(lines)
    print("\n" + report)

    # 直接写文件而不是靠 shell 重定向：Windows 控制台默认代码页会把手写
    # 重定向出来的报告变成乱码（早期 out/eval_report.txt 就是这样坏掉的）。
    (OUT_DIR / "eval_report.txt").write_text(report + "\n", encoding="utf-8")


def _f(v):
    return " n/a " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    run()
