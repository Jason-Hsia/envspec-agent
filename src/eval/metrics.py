# -*- coding: utf-8 -*-
"""四层评测指标。

L1 检索层  —— 召回质量，与生成质量解耦，便于定位问题在"找错"还是"说错"
L2 生成层  —— 引用与数值的准确性（环保领域最硬的两个指标）
L3 工具层  —— Tool Use 是否被正确触发、参数是否正确
L4 端到端  —— 业务可交付性，含时效性合规与拒答行为

指标设计原则：能用规则判定的，绝不用 LLM 打分。
数值类指标（numerical_accuracy / stale_citation_rate）全部是确定性判定。
"""
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set


# ================================================= L1 检索层

def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> Optional[float]:
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> Optional[float]:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return len(set(topk) & relevant) / len(topk)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> Optional[float]:
    if not relevant:
        return None
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> Optional[float]:
    if not relevant:
        return None
    dcg = sum((1.0 / math.log2(i + 1)) for i, d in enumerate(retrieved[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else None


def retrieval_metrics(retrieved_ids: List[str], relevant: Set[str]) -> Dict[str, Optional[float]]:
    return {
        "recall@5": recall_at_k(retrieved_ids, relevant, 5),
        "recall@10": recall_at_k(retrieved_ids, relevant, 10),
        "precision@5": precision_at_k(retrieved_ids, relevant, 5),
        "mrr": reciprocal_rank(retrieved_ids, relevant),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant, 10),
    }


# ================================================= L2 生成层

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numerical_accuracy(answer: str, expected_numbers: List[Dict]) -> Optional[float]:
    """数值准确率：标注的关键数值是否都出现在答案中。

    环保场景里 20 和 200 的差别就是合规与违法，必须用确定性规则判定，
    不能交给 LLM 做相似度打分。
    """
    if not expected_numbers:
        return None
    hit = 0
    for en in expected_numbers:
        v = en.get("value")
        if v is None:
            continue
        # 允许 35 / 35.0 两种写法；同时避免 "150" 命中 "1500"
        pat = re.compile(rf"(?<!\d){re.escape(str(v).rstrip('0').rstrip('.') if isinstance(v, float) else str(v))}(?!\d)")
        if pat.search(answer) or str(v) in answer:
            hit += 1
    return hit / len(expected_numbers)


def key_point_coverage(answer: str, key_points: List[str]) -> Optional[float]:
    if not key_points:
        return None
    hit = sum(1 for k in key_points if k in answer)
    return hit / len(key_points)


_VERDICT_RE = re.compile(r"适用优先级裁决[：:]([^\n]*)")


def conclusion_accuracy(answer: str, expected_numbers: List[Dict]) -> Optional[float]:
    """结论一致率：裁决题给出的**唯一结论**是否等于标注答案。

    为什么必须有这一项 —— numerical_accuracy 的问法是「答案里出现过 100 吗」，
    而裁决题问的是「结论是不是 100」。把三个候选等级成套列进正文
    （一级 100 / 二级 150 / 三级 500）时，前者满分、后者零分，
    而这恰好就是答对与答错的分界。

    这不是假想：Q006（排入地表水Ⅲ类水域的工业废水执行几级标准）
    结论写的是「三级，500 mg/L」，标注答案是「一级，100 mg/L」，
    却因为正文里同时列了三个等级，numerical_accuracy 与 key_point_coverage
    双双给 1.0，端到端门禁判为通过。**指标本身在替错误答案背书。**
    加了这一项之后，Q006 在修复前是不及格、修复后才通过。

    只在答案里存在明确裁决结论句时判定；纯查限值题返回 None。
    """
    m = _VERDICT_RE.search(answer)
    if not m:
        return None
    if not expected_numbers:
        return None
    return numerical_accuracy(m.group(1), expected_numbers)


def citation_precision(citations: List[Dict], must_not_cite: List[str],
                       relevant_std: Set[str]) -> float:
    """引用精度：引用的标准里，有多少是站得住的（非禁用、且与问题相关）。"""
    if not citations:
        return 0.0
    bad = 0
    for c in citations:
        sid = c.get("std_id", "")
        if any(m and (m in sid or sid in m) for m in must_not_cite):
            bad += 1
        elif relevant_std and not any(r in sid or sid in r for r in relevant_std):
            bad += 0.5   # 相关但非必需，扣半分
    return max(0.0, 1.0 - bad / len(citations))


def citation_recall(citations: List[Dict], expected_std: Set[str]) -> Optional[float]:
    if not expected_std:
        return None
    cited = {c.get("std_id", "") for c in citations}
    hit = sum(1 for e in expected_std if any(e in c or c in e for c in cited))
    return hit / len(expected_std)


# ================================================= L3 工具层

def tool_selection_accuracy(plan: List[Dict], expected_tool: str) -> int:
    return int(any(p["tool"] == expected_tool for p in plan))


def tool_success_rate(call_log: List[Dict]) -> Optional[float]:
    if not call_log:
        return None
    return sum(1 for c in call_log if c.get("ok")) / len(call_log)


def tool_efficiency(call_log: List[Dict], budget: int = 6) -> float:
    """调用次数效率：超过预算线性衰减，防止靠暴力多调工具刷指标。"""
    n = len(call_log)
    return 1.0 if n <= budget else max(0.0, 1.0 - (n - budget) / budget)


# ================================================= L4 端到端

def stale_citation_rate(citations: List[Dict], stale_ids: Set[str]) -> int:
    """时效性违规：是否引用了已废止标准。0/1 计数，用于一票否决门禁。"""
    for c in citations:
        sid = c.get("std_id", "")
        if any(s in sid or sid in s for s in stale_ids):
            return 1
    return 0


def refusal_correct(refused: bool, should_refuse: bool) -> int:
    return int(refused == should_refuse)


# 门禁键名 → 聚合指标键名。
# 两套命名是历史上分开长的（GATES 写 recall_at_5，聚合写 recall@5），
# 而 metrics.get(k) 取不到时被 `continue` 静默跳过 —— 七个门禁里三个从未生效：
# recall_at_5 / tool_selection_accuracy / refusal_accuracy 一直是摆设。
# 这三个恰好当时都是达标的，所以"24/24 通过"的结论没错，但门禁本身是坏的。
# 教训：门禁引用了不存在的指标属于配置错误，必须显式失败，绝不能静默跳过。
_GATE_KEY_ALIASES = {
    "recall_at_5": "recall@5",
    "recall_at_10": "recall@10",
    "precision_at_5": "precision@5",
    "ndcg_at_10": "ndcg@10",
    "tool_selection_accuracy": "tool_selection",
    "refusal_accuracy": "refusal_correct",
}


def end_to_end_pass(metrics: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """发布门禁：任一硬指标不达标则整体不通过，并列出失败项。

    与旧版的区别只有一处，但很关键：指标缺失时不再跳过。
    一个取不到值的门禁等于没有门禁，而报告上它和"通过"长得一模一样。
    """
    failures = []
    evaluated = 0
    for k, threshold in gates.items():
        mk = _GATE_KEY_ALIASES.get(k, k)
        v = metrics.get(mk)
        if v is None:
            failures.append(f"{k}: 无对应指标（聚合键 {mk}）—— 门禁配置错误，按不通过处理")
            continue
        evaluated += 1
        # stale_citation_rate 越低越好，其余越高越好
        if k == "stale_citation_rate":
            if v > threshold:
                failures.append(f"{k}={v} 超过阈值 {threshold}")
        elif v < threshold:
            failures.append(f"{k}={v:.3f} 低于阈值 {threshold}")
    # 把"实际校验了几项"一并返回：只看 passed=True 无法区分"全部门禁达标"
    # 与"门禁因为取不到指标而被跳过"。
    return {"passed": not failures, "failures": failures,
            "n_gates": len(gates), "n_evaluated": evaluated}


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
