# -*- coding: utf-8 -*-
"""Agent 编排层：规划 -> 调用工具 -> 汇总证据 -> 生成带引用的答案。

两种规划器：
  offline   —— 规则驱动，确定性，零 API 成本。用于建立可复现的评测基线。
  llm       —— OpenAI 兼容 Function Calling。接入后与 offline 做 A/B 对比。

设计约束（对应评测指标）：
  · 任何标准号出现前，必须已调用 check_standard_status  -> 保证 stale_citation_rate
  · 任何条文引用，必须来自 get_clause 的返回原文          -> 保证 citation_precision
  · 证据不足时输出拒答而非猜测                            -> 保证 refusal_accuracy
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, MAX_TOOL_ROUNDS
from src.retrieval.hybrid import understand
from src.tools.registry import EnvSpecTools


WATER_CLASSES = set("ⅠⅡⅢⅣⅤ")


def _num(v) -> str:
    """数值展示的统一约定：整数值不带小数点。

    同一份 JSON 在 Python 里 5.0 解析为 float、在 JS 里 5 解析为 number，
    直接插值会分别渲染成 "5.0" 和 "5"。前端内置引擎与后端做逐字节比对时
    暴露了这处差异，因此两侧统一走同一个格式化约定，避免展示层出现两套写法。
    """
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _grade_phrase(grade: str) -> str:
    """把等级槽位拼成读得通的措辞。

    早先用的是「以'级'结尾就补'标准'，否则补'类'」，结果两处出洋相：
    自定义标签「浙江地标」被写成「浙江地标类」，而噪声等级「2类」被写成「2类类」。
    等级槽位的取值本来就跨三套体系（Ⅲ / 一级 / 2类 / 浙江地标），
    不能靠一个二分判断猜后缀，要按实际形态分派。
    """
    if not grade:
        return ""
    if grade.endswith("级"):
        return f"{grade}标准"
    if grade.endswith("类"):
        return grade
    if grade in ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ"):
        return f"{grade}类"
    return grade


def _fmt_citation(c: Dict[str, Any]) -> str:
    """引用条目的展示格式。

    单独抽成函数是因为这里曾经有一处隐蔽的格式瑕疵：f-string 隐式拼接时
    `table` 后面多留了一个空格，而 `.rstrip()` 只能去掉行尾空格，导致
    输出成「表1 ，Ⅲ」这种夹在中间的空格。前端内置引擎复刻同一套逻辑时
    立刻把这一字节的差异暴露了出来 —— 两份实现只有在完全相同的前提下，
    前端展示的才可能是评测报告里的那个结果。
    """
    s = f"[{c['idx']}] {c['std_id']}"
    if c.get("table"):
        s += f" {c['table']}"
    if c.get("clause_no") and not c.get("table"):
        s += f" 第{c['clause_no']}条"
    if c.get("grade"):
        s += f"（{c['grade']}）"
    return s


def _grade_for_medium(medium: str, grades: List[str]) -> str:
    """查询分解时，为每种介质挑出它自己那套等级体系的取值。

    地表水说"Ⅲ类"，废水说"一级"，噪声说"2类" —— 三套体系互不相通。
    混用会直接取错档位，所以按介质分别挑。
    """
    if not grades:
        return ""
    if medium == "water_surface":
        return next((g for g in grades if g in WATER_CLASSES), "")
    if medium in ("water_wastewater", "air_emission"):
        return next((g for g in grades if g.endswith("级")), "")
    if medium == "noise":
        return next((g for g in grades if g.endswith("类")), "")
    return ""


class EnvSpecAgent:
    def __init__(self, tools: Optional[EnvSpecTools] = None, provider: str = None):
        self.tools = tools or EnvSpecTools()
        self.provider = provider or LLM_PROVIDER
        self.decls = {d["name"]: d for d in EnvSpecTools.declarations()}

    # ------------------------------------------------------ 规划
    def plan(self, query: str, slots) -> List[Dict[str, Any]]:
        if self.provider == "llm":
            return self._plan_llm(query, slots)
        return self._plan_offline(query, slots)

    def _plan_offline(self, query: str, slots) -> List[Dict[str, Any]]:
        """规则规划器：把槽位映射成工具序列。

        关键决策——**目标介质**的确定：
          出现"排放/废水/废气"语境时，问的是排放标准，目标是 water_wastewater/air_emission；
          只出现"地表水/环境空气"时，目标是环境质量标准；
          多介质并列（对比题）时反而不锁介质，交给检索层聚合多路证据。
        """
        plan: List[Dict[str, Any]] = []
        media = slots.media
        region = slots.regions[0] if slots.regions else ""
        industry = slots.industry[0] if getattr(slots, "industry", None) else ""

        # 0) 硬约束前置：任何点名了标准号的查询，都必须先复核时效性。
        #    这条规则不交给模型"判断要不要"，而是由规划器无条件插入 —— 时效性违规是
        #    本项目唯一的一票否决指标，不能有概率性。
        for r in slots.std_refs[:3]:
            plan.append({"tool": "check_standard_status", "args": {"std_id": r}})
        if len(slots.std_refs) >= 2:
            plan.append({"tool": "trace_standard_chain", "args": {"std_id": slots.std_refs[0]}})

        # 纯时效问询：前置步骤已经答完
        if slots.wants_status and slots.std_refs and not slots.wants_limit:
            return plan

        # 环评类别判定类
        if slots.wants_eia_class:
            plan.append({"tool": "classify_eia_category",
                         "args": {"project_type": query, "scale": query}})
            return plan

        # 注意：这里不再单独拦截"时效性问询"。
        # 前置的 check_standard_status 已经覆盖该意图，如果在这里再 return 一次，
        # 会把"某现行标准的某污染物限值是多少"这类复合问题误判成纯时效问题、跳过查限值。

        # 多介质对比题 —— 做查询分解，每种介质各查一次。
        # 典型："地表水III类COD限值和废水一级排放限值有什么区别"。
        # 这类问题不存在唯一适用标准，若强行走单路查限值+裁决，必然丢掉一半答案。
        if len(media) > 1 and slots.pollutants and slots.is_comparison:
            for p in slots.pollutants:
                for m in media:
                    g = _grade_for_medium(m, slots.grades)
                    plan.append({"tool": "lookup_limit",
                                 "args": {"pollutant": p, "medium": m, "grade": g,
                                          "region": region}})
            plan.append({"tool": "search_standard",
                         "args": {"query": query, "top_n": 4}})
            return plan

        discharge_ctx = any(m in media for m in ("water_wastewater", "air_emission"))
        if discharge_ctx:
            target_medium = "water_wastewater" if "water_wastewater" in media else "air_emission"
        elif len(media) == 1:
            target_medium = media[0]
        else:
            target_medium = ""

        grade = ""
        if discharge_ctx:
            # 排放等级必须优先取"受纳水体翻译结果"，再退回字面等级。
            # 顺序不能反："排入地表水Ⅲ类水域的工业废水执行几级标准"里槽位抽到的是
            # Ⅲ（受纳水体类别，来自 slots.grades），它不等于排放等级；真正的等级
            # 要由 infer_discharge_grade 翻译成"一级"。取反了就会退化为按限值
            # 严格程度猜等级，给出 500 mg/L 而不是 100 mg/L。
            grade = slots.discharge_grade or next(
                (g for g in slots.grades if g.endswith("级")), "")
        elif slots.grades:
            grade = slots.grades[0]
        if len(media) > 1 and not discharge_ctx:
            target_medium, grade = "", ""     # 对比题不锁条件

        # 限值 / 达标判定类（主线）
        if slots.wants_limit and slots.pollutants:
            for p in slots.pollutants:
                plan.append({"tool": "lookup_limit",
                             "args": {"pollutant": p, "medium": target_medium,
                                      "grade": grade, "region": region,
                                      "prefer_period": slots.period_stated}})
            # 只在"目标介质已确定"时才做适用性裁决。
            # 多介质对比题（如"地表水III类 vs 废水一级"）本来就没有唯一适用标准，
            # 强行裁决只会把其中一半的正确答案抹掉。
            if target_medium and (region or discharge_ctx or industry or len(media) > 1):
                plan.append({"tool": "resolve_applicable_standard",
                             "args": {"pollutant": slots.pollutants[0],
                                      "medium": target_medium, "region": region,
                                      "industry": industry, "grade": grade,
                                      "prefer_period": slots.period_stated}})
            plan.append({"tool": "search_standard",
                         "args": {"query": query, "medium": target_medium, "top_n": 4}})
            return plan

        # 兜底：通用检索 + 时效复核
        plan.append({"tool": "search_standard", "args": {"query": query, "top_n": 6}})
        return plan

    def _plan_llm(self, query: str, slots) -> List[Dict[str, Any]]:
        """真实 Function Calling 规划。

        保留工具白名单与硬约束提示；注意 prompt 中显式要求
        「引用任何标准号前必须先调用 check_standard_status」，
        这是把业务规则写进系统提示，而非寄望模型自觉。
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("provider=llm 需要安装 openai：pip install openai")

        client = OpenAI(base_url=LLM_BASE_URL or None, api_key=LLM_API_KEY or "sk-none")
        sys_prompt = (
            "你是环保标准检索助手的环境工程专家。可用工具见 function 定义。\n"
            "硬性规则：\n"
            "1) 引用任何标准号前，必须先调用 check_standard_status 确认其为现行有效版本。\n"
            "2) 引用条文原文前，必须调用 get_clause 获取，禁止凭记忆复述。\n"
            "3) 涉及国标/行标/地标同时适用时，必须调用 resolve_applicable_standard 裁决。\n"
            "4) 证据不足时立即停止调用并说明无法回答，禁止臆测限值。\n"
            "请在完成必要调用后，用一句 JSON 数组输出你计划调用的工具序列。"
        )
        funcs = [{"type": "function",
                  "function": {"name": d["name"], "description": d["description"],
                               "parameters": {"type": "object",
                                              "properties": {k: {"type": "string"}
                                                             for k in d["parameters"]},
                                              "required": []}}}
                 for d in self.decls.values()]
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=[{"role": "system", "content": sys_prompt},
                                       {"role": "user", "content": query}],
            tools=funcs, tool_choice="auto", temperature=0,
        )
        calls = []
        msg = resp.choices[0].message
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"tool": tc.function.name, "args": args})
        return calls or self._plan_offline(query, slots)

    # ------------------------------------------------------ 执行 + 汇总
    def run(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        slots = understand(query)
        plan = self.plan(query, slots)[:MAX_TOOL_ROUNDS]
        self.tools.call_log = []

        observations: List[Dict[str, Any]] = []
        evidence_clauses: Dict[str, Dict] = {}
        stale_warnings: List[str] = []
        limits_found: List[Dict] = []

        for step in plan:
            fn = getattr(self.tools, step["tool"], None)
            if fn is None:
                continue
            try:
                out = fn(**step["args"])
            except TypeError as e:
                out = {"error": f"参数错误: {e}"}
            observations.append({"tool": step["tool"], "args": step["args"], "output": out})

            if step["tool"] == "search_standard":
                for h in out.get("hits", []):
                    evidence_clauses[h["clause_id"]] = h
                    if h.get("is_stale") and h.get("replacement"):
                        stale_warnings.append(
                            f"{h['std_id']} 已被替代，应改用 {h['replacement']}")
            if step["tool"] == "check_standard_status":
                if out.get("warning"):
                    stale_warnings.append(out["warning"])
                # 标准号不完整导致多版本候选时，同样要把失效版本挑出来提示
                for cand in out.get("candidates") or []:
                    rep = cand.get("replaced_by") or []
                    if cand.get("status") == "superseded" and rep:
                        stale_warnings.append(f"{cand['std_id']} 已被替代，应改用 {rep[0]}")
            if step["tool"] == "lookup_limit":
                limits_found.extend(out.get("records", []))
            if step["tool"] == "get_clause" and "text" in out:
                evidence_clauses[out["clause_id"]] = out
            if verbose:
                print(f"  -> {step['tool']}({step['args']}) ok={bool(out) and 'error' not in out}")

        answer, citations, refused = self._compose(
            query, slots, observations, evidence_clauses, limits_found, stale_warnings)

        return {
            "query": query,
            "slots": slots.to_dict(),
            "plan": [{"tool": p["tool"], "args": p["args"]} for p in plan],
            "tool_calls": list(self.tools.call_log),
            "observations": observations,
            "answer": answer,
            "citations": citations,
            "stale_warnings": sorted(set(stale_warnings)),
            "refused": refused,
            "limits": limits_found,
        }

    def _compose(self, query, slots, observations, evidence_clauses,
                 limits_found, stale_warnings):
        citations: List[Dict] = []
        parts: List[str] = []

        # --- 时效性提示必须置顶
        if stale_warnings:
            parts.append("【时效性提示】" + "；".join(sorted(set(stale_warnings)))
                         + "。以下如涉及该标准，请以替代版本为准。")

        # --- 适用优先级裁决（若有，作为主结论，并据此收敛限值展示）
        verdict = next((o["output"] for o in observations
                        if o["tool"] == "resolve_applicable_standard"
                        and "applicable" in o["output"]), None)
        if verdict:
            a = verdict["applicable"]
            idx = len(citations) + 1
            citations.append({"idx": idx, "std_id": a["std_id"], "table": "",
                              "clause_no": "", "grade": a.get("grade", ""),
                              "value": a.get("value"), "unit": a.get("unit", "")})
            parts.append(f"适用优先级裁决：应适用 {a.get('title','')}（{a['std_id']}）"
                         f"{a.get('grade','')}，限值 {a.get('value_display') or _num(a.get('value'))} "
                         f"{a.get('unit','')} [{idx}]。"
                         f"理由：{'；'.join(verdict.get('reasons', []))}")

        # --- 限值证据收敛三段式：
        #   ① 已废止版本的记录一律不出现在引用中（只是不作为引用，不影响替代提示）
        #   ② 有裁决结论时，只保留胜出标准自己的记录 —— 否则会把被否掉的候选一并倒给用户
        #   ③ 无裁决时，按查询中的等级收敛，避免把 Ⅲ/Ⅳ/Ⅴ 三档全列出来
        shown = [l for l in limits_found if l.get("is_current")] or limits_found
        if verdict:
            winner_id = verdict["applicable"]["std_id"]
            same = [l for l in shown if l["std_id"] == winner_id]
            shown = same or shown
        elif slots.grades:
            narrowed = [l for l in shown if l.get("grade", "") in slots.grades]
            shown = narrowed or shown
        for l in shown[:4]:
            idx = len(citations) + 1
            citations.append({"idx": idx, "std_id": l["std_id"], "table": l["table"],
                              "clause_no": l["table"], "grade": l["grade"],
                              "value": l["value"], "unit": l["unit"]})
            flag = "" if l.get("is_current") else "（该版本已非现行）"
            grade_txt = _grade_phrase(l["grade"])
            parts.append(
                f"{l.get('std_title') or ''}（{l['std_id']}）{l['table']} 规定，"
                f"{grade_txt}下限值为 {l.get('value_display') or _num(l['value'])} {l['unit']}"
                f"（适用条件：{l['condition']}）{flag} [{idx}]")

        # --- 达标判定结论
        for o in observations:
            if o["tool"] == "judge_compliance" and "verdict" in o["output"]:
                j = o["output"]
                parts.append(f"达标判定：实测折算 {_num(j['measured_normalized'])} {j['limit_unit']}，"
                             f"限值 {j.get('limit_display') or _num(j['limit'])} {j['limit_unit']}，"
                             f"结论为 **{j['verdict']}**"
                             f"（超标倍数 {j['exceedance_ratio']:+.2%}）。{j.get('caveat','')}")

        # --- 环评类别
        for o in observations:
            if o["tool"] == "classify_eia_category":
                out = o["output"]
                if "suggested_type" in out:
                    idx = len(citations) + 1
                    citations.append({"idx": idx, "std_id": out.get("source_std_id", "名录-2021"),
                                      "table": "分类管理名录", "clause_no": "",
                                      "grade": "", "value": None, "unit": ""})
                    parts.append(f"环评类别判定：{out['project_type']} 对应「{out['matched_category']}」，"
                                 f"建议编制 **{out['suggested_type']}** [{idx}]。"
                                 f"注意：{out['must_verify']}")
                else:
                    parts.append(f"环评类别判定：{out.get('error')}。{out.get('hint','')}")

        # --- 条文证据
        # 只在"纯规则问询"（无结构化限值证据、也无裁决结论）时补充条文原文。
        # 限值型答案若再塞进规则条文，只会拉低 citation_precision。
        extra: List[Dict] = []
        if not limits_found and verdict is None:
            extra = [h for h in evidence_clauses.values()
                     if h.get("clause_type") in ("management", "method", "definition")
                     # 硬过滤：已废止标准的条文一律不得进入引用列表。
                     # 只在顶部提示替代关系，绝不引用其原文 —— 这是时效性合规率能归零的关键，
                     # 仅靠提示语是拦不住的。
                     and not h.get("is_stale")]
            extra.sort(key=lambda h: -h.get("score", 0))
        for h in extra[:2]:
            idx = len(citations) + 1
            citations.append({"idx": idx, "std_id": h["std_id"],
                              "clause_no": h.get("clause_no", ""), "table": "",
                              "grade": "", "value": None, "unit": ""})
            txt = h["text"][:170] + ("…" if len(h["text"]) > 170 else "")
            parts.append(f"规则依据 —— {h['std_id']} 第{h.get('clause_no','')}条：{txt} [{idx}]")

        # --- 拒答判定：必须有实质性依据才作答。
        # 注意：工具返回了明确结论（如环评类别）也算依据，不能因为"没有条文引用"就误判为拒答。
        if not citations and not stale_warnings:
            return (f"知识库中未检索到足以回答「{query}」的依据。"
                    "为避免给出错误的限值或条文，此处不作推测性回答。"
                    "建议补充污染物名称、适用介质（地表水/废水/环境空气等）与标准等级后重试。",
                    [], True)

        body = "\n".join(f"- {p}" for p in parts)
        refs = "\n".join(_fmt_citation(c) for c in citations)
        answer = f"{body}\n\n**依据**\n{refs}\n\n" \
                 f"_提示：以上限值来自项目知识库快照，正式引用前请以官方发布文本复核。_"
        return answer, citations, False


def ask(query: str, verbose: bool = False) -> Dict[str, Any]:
    return EnvSpecAgent().run(query, verbose=verbose)


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    q = " ".join(sys.argv[1:]) or "地表水III类标准的COD限值是多少"
    res = ask(q, verbose=True)
    print("\n" + "=" * 60)
    print(res["answer"])
    print("=" * 60)
    print(f"拒答: {res['refused']} | 引用数: {len(res['citations'])} "
          f"| 工具调用: {[c['tool'] for c in res['tool_calls']]}")
