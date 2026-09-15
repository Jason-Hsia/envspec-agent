# -*- coding: utf-8 -*-
"""Bad case 归因与定向优化建议。

闭环的核心不是"跑一次评测"，而是"评测结果能自动映射到具体动作"。
本模块把失败样例归入 7 类，每类绑定确定的修复手段，输出可执行清单。
"""
import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import OUT_DIR

# 归因规则：(代号, 名称, 判定函数, 根因, 定向动作, 责任层)
RULES = [
    ("R1", "检索未召回",
     lambda c: c.get("recall@5") == 0.0,
     "查询词与条文表述无重叠，或元数据过滤过严把正确条文剪掉了",
     "① 补领域同义词表（污染物别名/俗称）② 检查 _passes_filter 的 medium 映射是否过窄 "
     "③ 降低 MIN_SCORE_THRESHOLD 后复测",
     "retrieval"),

    ("R2", "召回含噪声",
     lambda c: (c.get("recall@5") or 0) > 0 and (c.get("citation_precision") or 1) < 0.7,
     "正确条文召回了，但错误条文排在前面，模型被带偏",
     "① 提高重排权重，尤其是标准号精确匹配的 boost ② 引入 cross-encoder 重排 "
     "③ 收紧 clause_type 过滤",
     "retrieval"),

    ("R3", "召回不足",
     lambda c: 0 < (c.get("recall@5") or 0) < 1.0,
     "多证据题只召回了部分条文（典型场景：跨标准对比题）",
     "① 提升 RRF 融合的 top_k ② 对多污染物查询做查询分解（query decomposition）"
     "③ 增加按标准号定向检索的分支",
     "retrieval"),

    ("T1", "工具选择错误",
     lambda c: c.get("tool_selection") == 0,
     "规划器未选择预期工具，走了错误路径",
     "① 补规划器规则分支 ② 若是 llm 规划器，在 system prompt 增加该场景的 few-shot "
     "③ 检查槽位抽取是否漏识别（如等级/属地）",
     "planning"),

    ("T2", "工具执行失败",
     lambda c: c.get("tool_success_rate") is not None and c["tool_success_rate"] < 1.0,
     "工具被调用但返回 error，多为参数不匹配或库中无记录",
     "① 校验参数 schema 与归一化逻辑 ② 为 lookup_limit 增加模糊匹配降级 "
     "③ 记录缺失的标准/污染物并补充数据",
     "tools"),

    ("G1", "数值/要点错误",
     lambda c: (c.get("numerical_accuracy") is not None and c["numerical_accuracy"] < 1.0)
               or (c.get("key_point_coverage") is not None and c["key_point_coverage"] < 0.6),
     "答案中的关键数值或要点缺失/错误",
     "① 确认该题是否真的调用了 lookup_limit（未调用则不可能是数值错误，而是规划问题）"
     "② 检查限值表的 grade/condition 是否覆盖了该提问方式 ③ 补评测标注",
     "generation"),

    ("G2", "时效性违规",
     lambda c: c.get("stale_citation") == 1,
     "答案引用了已废止标准 —— 项目最高危缺陷",
     "① 强制 search_standard 返回的 is_stale 条文不得进入 citations "
     "② 在 _compose 中把 stale_warnings 做硬过滤而非仅提示 "
     "③ 溯源替代链并自动替换为标准的新版本",
     "generation"),

    ("U1", "拒答行为错误",
     lambda c: c.get("refusal_correct") == 0,
     "该拒答时作答（幻觉），或该作答时拒答（过度保守）",
     "作答方向：收紧 MIN_SCORE_THRESHOLD、加强 prompt 的'无依据即拒答'约束；"
     "拒答方向：检查过滤条件是否把证据剪掉了",
     "generation"),
]


def diagnose(path: Path = None) -> Dict:
    path = path or (OUT_DIR / "eval_report.json")
    if not path.exists():
        raise SystemExit(f"未找到评测报告 {path}，请先运行 python -m src.eval.run_eval")

    report = json.loads(path.read_text(encoding="utf-8"))
    cases = report["cases"]

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for c in cases:
        tags = [code for code, _n, pred, _r, _a, _l in RULES if pred(c)]
        for t in tags:
            buckets[t].append(c)

    # 优先级：时效性 > 拒答 > 数值 > 检索 > 规划 > 工具
    PRIORITY = {"G2": 0, "U1": 1, "G1": 2, "R1": 3, "R3": 4, "R2": 5, "T1": 6, "T2": 7}

    # 判定"硬失败"：整体通过判据（答对 + 无时效性违规 + 要点覆盖达标）
    def hard_failed(c: Dict) -> bool:
        return not (c["refusal_correct"] == 1
                    and not c.get("stale_citation")
                    and (c.get("key_point_coverage") is None or c["key_point_coverage"] >= 0.6))

    failed_ids = {c["id"] for c in cases if hard_failed(c)}

    findings = []
    for code, name, _pred, root, action, layer in RULES:
        hit = buckets.get(code, [])
        if not hit:
            continue
        sev = "fail" if any(c["id"] in failed_ids for c in hit) else "weak"
        findings.append({
            "code": code, "name": name, "layer": layer,
            "severity": sev,
            "count": len(hit),
            "rate": round(len(hit) / len(cases), 3),
            "case_ids": [c["id"] for c in hit],
            "root_cause": root,
            "actions": action,
            "priority": PRIORITY.get(code, 99) + (10 if sev == "weak" else 0),
        })
    findings.sort(key=lambda f: (f["priority"], -f["count"]))

    passed = [c for c in cases if c["refusal_correct"] and not c.get("stale_citation")
              and (c.get("key_point_coverage") is None or c["key_point_coverage"] >= 0.6)]

    result = {
        "n_cases": len(cases),
        "n_passed": len(passed),
        "pass_rate": round(len(passed) / len(cases), 3),
        "attribution": findings,
        "layer_distribution": dict(Counter(f["layer"] for f in findings
                                           for _ in range(f["count"]))),
        "gate_result": report.get("gate_result"),
        "next_sprint": [f"{f['code']} {f['name']}（{f['count']}例，{f['layer']}层）：{f['actions']}"
                        for f in findings[:3]],
    }
    return result


def format_report(r: Dict) -> str:
    """拼装报告文本并返回，不直接 print。

    直接 print + shell 重定向在 Windows 上会落盘成乱码（控制台默认按 GBK 解码子进程
    stdout），out/diagnosis.txt 之前就是这么坏掉的，而 README 还指着这个文件。
    改成显式 UTF-8 写文件，与 run_eval 保持一致。
    """
    out: List[str] = []
    add = out.append
    add("=" * 66)
    add(f"Bad case 归因报告   硬通过 {r['n_passed']}/{r['n_cases']}  通过率 {r['pass_rate']:.1%}")
    add("=" * 66)
    add("说明：以下条目同时包含『硬失败』与『子指标薄弱』两类。")
    add("      硬失败 = 该样例整体判错；薄弱 = 样例已通过，但某项子指标仍可改进。")
    if not r["attribution"]:
        add("无失败样例。")
    for f in r["attribution"]:
        tag = "硬失败" if f["severity"] == "fail" else "薄弱项"
        add(f"\n[{f['code']}] {f['name']}   {f['count']} 例（{f['rate']:.1%}）  "
            f"{tag}  责任层: {f['layer']}")
        add(f"     样例: {', '.join(f['case_ids'][:6])}")
        add(f"     根因: {f['root_cause']}")
        add(f"     动作: {f['actions']}")
    add("\n" + "-" * 66)
    add(f"问题分布（按责任层）: {r['layer_distribution']}")
    add("\n下一轮优先修复：")
    for s in r["next_sprint"]:
        add("  • " + s)
    if r.get("gate_result"):
        g = r["gate_result"]
        add("\n门禁: " + ("✅ 通过" if g["passed"] else "❌ 未通过 " + "; ".join(g["failures"])))
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()
    res = diagnose(Path(args.report) if args.report else None)
    text = format_report(res)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diagnosis.txt").write_text(text + "\n", encoding="utf-8")
    (OUT_DIR / "diagnosis.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT_DIR / 'diagnosis.txt'} 与 {OUT_DIR / 'diagnosis.json'}")

