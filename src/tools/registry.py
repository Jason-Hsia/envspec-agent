# -*- coding: utf-8 -*-
"""Tool Use 层：8 个工具的声明与实现。

工具存在的唯一理由，是把"模型不擅长 / 不该靠记忆回答"的部分外包给确定性代码。
本项目的三个关键工具：
  check_standard_status      —— 时效性校验（纯 RAG 的死穴）
  resolve_applicable_std     —— 标准适用优先级裁决（国标/行标/地标冲突）
  judge_compliance           —— 达标判定与单位换算（数值计算不可交给 LLM）
"""
import re
from datetime import date
from typing import Any, Dict, List, Optional

from src.retrieval.hybrid import HybridRetriever, understand

_RE_COND_PERIOD = re.compile(r"(\d{1,2}\s*月|冬季|冬期|汛期|非汛期|枯水期|丰水期)")


def _cond_has_period(condition: str) -> bool:
    """限值记录的适用条件是否**限定在特定时期**（区别于"常规时段"）。

    这里刻意不把"常规时段"算进来。它与"11月1日至次年3月31日"是一对互斥取值，
    真正要判断的是"这条限值只在某段时期生效吗"：
      查询限定了时期（冬季）→ 优先取 True 的那条
      查询没限定        → 优先取 False 的常规那条
    把"常规时段"也算成 True，两类记录就分不开了，筛选退化为按原顺序取第一条。
    """
    return bool(_RE_COND_PERIOD.search(condition or ""))


# ---------------------------------------------------------------- 单位换算表
UNIT_TO_BASE = {
    ("concentration_water", "mg/l"): 1.0,
    ("concentration_water", "ug/l"): 0.001,
    ("concentration_water", "g/l"): 1000.0,
    ("concentration_air", "mg/m3"): 1000.0,
    ("concentration_air", "ug/m3"): 1.0,
}
UNIT_FAMILY = {
    "mg/l": "concentration_water", "ug/l": "concentration_water", "g/l": "concentration_water",
    "mg/m3": "concentration_air", "ug/m3": "concentration_air",
    "db(a)": "level", "db": "level",
}


def _unify_unit(u: str) -> str:
    return (u or "").replace("μ", "u").replace("³", "3").lower().strip()


def convert(value: float, from_u: str, to_u: str) -> float:
    f, t = _unify_unit(from_u), _unify_unit(to_u)
    if f == t:
        return value
    fam = UNIT_FAMILY.get(f)
    if not fam or UNIT_FAMILY.get(t) != fam or fam == "level":
        raise ValueError(f"单位不可换算: {from_u} -> {to_u}（分贝为对数单位，不可线性换算）")
    return value * UNIT_TO_BASE[(fam, f)] / UNIT_TO_BASE[(fam, t)]


# ---------------------------------------------------------------- 工具实现

class EnvSpecTools:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.r = retriever or HybridRetriever()
        self.call_log: List[Dict[str, Any]] = []
        self.index = self.r.index
        self._std_map = {s["std_id"]: s for s in self.index["standards"]}
        self._chain = self.index.get("chain", [])
        self._limits = self.index["limits"]

    # ---- 内部：调用日志（工具层指标的数据来源）
    def _log(self, name: str, args: Dict, ok: bool, n_result: int = 0):
        self.call_log.append({"tool": name, "args": args, "ok": ok, "n_result": n_result})

    # ---------------- 1. 标准/条文检索
    def search_standard(self, query: str, medium: str = "", grade: str = "",
                        region: str = "", top_n: int = 6) -> Dict:
        slots = understand(query)
        if medium:
            slots.media = [medium]
        if region:
            slots.regions = [region]
        if grade:
            slots.grades = [grade]
        results, _ = self.r.retrieve(query, top_n=top_n, slots=slots)
        self._log("search_standard", {"query": query, "medium": medium}, bool(results), len(results))
        return {
            "query": query,
            "slots": slots.to_dict(),
            "hits": [{
                "clause_id": c["clause_id"], "std_id": c["std_id"], "clause_no": c["clause_no"],
                "std_title": c["std_title"], "clause_type": c["clause_type"],
                "text": c["text"], "score": c["score"],
                "std_status": c["std_status"], "is_stale": c["is_stale"],
                "replacement": c.get("replacement"),
                "effective_date": c.get("std_effective_date"),
            } for c in results],
        }

    # ---------------- 2. 条文精查
    def get_clause(self, clause_id: str) -> Dict:
        c = self.r.get_clause(clause_id)
        self._log("get_clause", {"clause_id": clause_id}, c is not None, 1 if c else 0)
        if not c:
            return {"error": f"条文不存在: {clause_id}", "hint": "先用 search_standard 确认条文号"}
        return {
            "clause_id": c["clause_id"], "std_id": c["std_id"], "clause_no": c["clause_no"],
            "path": c["path"], "clause_type": c["clause_type"],
            "text": c["text"], "refs": c["refs"],
            "std_status": c["std_status"], "std_replaced_by": c["std_replaced_by"],
        }

    # ---------------- 3. 时效性校验 ★核心
    def check_standard_status(self, std_id: str, as_of: str = "") -> Dict:
        as_of = as_of or date.today().isoformat()
        s = self._std_map.get(std_id)
        if not s:
            # 模糊匹配：用户常写 "3838-2002" 或省略年份的 "GB 18597"。
            # 唯一命中时直接按该标准回答；多命中时把各版本时效性一并列出 ——
            # 这恰好就是用户想要的答案（"哪一版还在用"）。
            digits = re.sub(r"[^0-9]", "", std_id)
            cands = [k for k in self._std_map if digits and digits in re.sub(r"[^0-9]", "", k)]
            if len(cands) == 1:
                out = self.check_standard_status(cands[0], as_of)
                out["resolved_from"] = std_id
                return out
            self._log("check_standard_status", {"std_id": std_id}, bool(cands), len(cands))
            if not cands:
                return {"error": f"未收录标准号 {std_id}", "candidates": [],
                        "hint": "确认标准号拼写，或先用 search_standard 检索"}
            return {
                "std_id": std_id, "resolved": False, "ambiguous": True,
                "candidates": [{
                    "std_id": c, "title": self._std_map[c].get("title", ""),
                    "status": self._std_map[c]["status"],
                    "effective_date": self._std_map[c].get("effective_date", ""),
                    "replaced_by": self._std_map[c].get("replaced_by", []),
                } for c in sorted(cands)],
                "hint": "该标准号存在多个版本，请认准 status=active 的现行版本",
            }
        self._log("check_standard_status", {"std_id": std_id}, True, 1)
        return {
            "std_id": s["std_id"], "title": s["title"],
            "status": s["status"],
            "is_current": s["status"] == "active",
            "publish_date": s["publish_date"], "effective_date": s["effective_date"],
            "as_of": as_of,
            "replaced_by": s["replaced_by"],
            "replaces": s["replaces"],
            "warning": ("该标准已被替代，禁止用于新建项目环评/验收，请改用 "
                        + "、".join(s["replaced_by"])) if s["status"] == "superseded" else None,
        }

    # ---------------- 4. 限值结构化查询
    def lookup_limit(self, pollutant: str, medium: str = "", grade: str = "",
                     condition: str = "", region: str = "", prefer_period: bool = False) -> Dict:
        rows = [l for l in self._limits if l["medium"] == medium] if medium else list(self._limits)

        def p_match(l):
            key = pollutant.lower()
            return (key in l["pollutant"].lower()
                    or any(key in a.lower() for a in l.get("pollutant_alias", [])))

        rows = [l for l in rows if p_match(l)]
        if grade:
            graded = [l for l in rows if grade in l["grade"]]
            rows = graded or rows
        if condition:
            cond = [l for l in rows if condition in l["condition"]]
            rows = cond or rows

        # 时期条件的取舍：问题限定了时期（"11月到次年3月"/冬季/汛期）就取带时段的记录，
        # 没限定就取常规时段记录。这一步必须排在"属地优先"和后面的"同分从严"之前 ——
        # 用限值严格程度去替代时期匹配，只是一种蒙对率不稳定的启发式。
        rows = sorted(rows, key=lambda l: (_cond_has_period(l["condition"]) != bool(prefer_period),))

        # 属地标准优先：地方标准严于国标时，属地在场则地标置顶
        if region:
            def has_region(std_id):
                s = self._std_map.get(std_id, {})
                return region in (s.get("applicability", {}).get("region") or [])
            rows = sorted(rows, key=lambda l: (not has_region(l["std_id"]),))

        for l in rows:
            l["_std_title"] = self._std_map.get(l["std_id"], {}).get("title", "")
            l["_is_current"] = self._std_map.get(l["std_id"], {}).get("status") == "active"
        self._log("lookup_limit", {"pollutant": pollutant, "medium": medium, "grade": grade},
                  bool(rows), len(rows))

        if not rows:
            return {"error": "未匹配到限值记录",
                    "hint": "尝试放宽 grade/condition，或先用 search_standard 定位标准号再查表"}
        return {
            "pollutant": pollutant, "medium": medium, "region": region,
            "count": len(rows),
            "records": [{
                "std_id": l["std_id"], "std_title": l["_std_title"], "table": l["table"],
                "grade": l["grade"], "condition": l["condition"],
                "value": l["value"], "value_text": l.get("value_text"),
                "value_display": l.get("value_display") or "",
                "unit": l["unit"], "is_current": l["_is_current"],
            } for l in rows],
            "notice": "以上为库内记录，引用前请用 check_standard_status 复核标准时效性",
        }

    # ---------------- 5. 达标判定
    def judge_compliance(self, measured: float, pollutant: str, medium: str = "",
                         grade: str = "", unit: str = "", condition: str = "",
                         region: str = "", margin: float = 0.0,
                         prefer_period: bool = False) -> Dict:
        lk = self.lookup_limit(pollutant, medium, grade, condition, region,
                               prefer_period=prefer_period)
        if lk.get("error"):
            return lk
        rec = lk["records"][0]
        if rec["value"] is None:
            return {"error": f"{rec['std_id']} 该项限值为非数值表述（{rec.get('value_text')}），无法自动判定",
                    "raw": rec}
        limit_val = float(rec["value"])
        try:
            m = convert(measured, unit, rec["unit"]) if unit else measured
        except ValueError as e:
            return {"error": str(e)}

        threshold = limit_val * (1 - margin) if margin else limit_val
        compliant = m <= threshold
        return {
            "verdict": "达标" if compliant else "超标",
            "compliant": compliant,
            "standard": f"{rec['std_id']} {rec['table']}（{rec['grade']}，{rec['condition']}）",
            "limit": limit_val, "limit_unit": rec["unit"],
            "limit_display": rec.get("value_display") or str(limit_val),
            "measured_normalized": round(m, 4), "measured_input": measured, "input_unit": unit or rec["unit"],
            "exceedance_ratio": round(m / limit_val - 1, 4),
            "margin_applied": margin,
            "caveat": "判定仅基于库内限值记录；实际执法判定还需考虑监测方法、采样频次与数据有效性规定",
        }

    # ---------------- 6. 替代链追溯
    def trace_standard_chain(self, std_id: str) -> Dict:
        seen, fwd, cur = set(), [], std_id
        while cur and cur not in seen:
            seen.add(cur)
            nxt = self._std_map.get(cur, {}).get("replaced_by") or []
            nxt = [n for n in nxt if n in self._std_map]
            if not nxt:
                break
            fwd.append({"from": cur, "to": nxt[0]})
            cur = nxt[0]
        seen_b, back, cur = set(), [], std_id
        while cur and cur not in seen_b:
            seen_b.add(cur)
            prev = self._std_map.get(cur, {}).get("replaces") or []
            prev = [p for p in prev if p in self._std_map]
            if not prev:
                break
            back.append({"from": prev[0], "to": cur})
            cur = prev[0]
        self._log("trace_standard_chain", {"std_id": std_id}, True, len(fwd) + len(back))
        return {
            "std_id": std_id,
            "current_in_line": cur or std_id,
            "ancestors": back[::-1],
            "descendants": fwd,
            "note": "链条末端即为当前应引用的版本" if (fwd or back) else "该标准无已知替代关系",
        }

    # ---------------- 7. 标准适用优先级裁决 ★核心
    def resolve_applicable_standard(self, pollutant: str = "", medium: str = "",
                                    region: str = "", industry: str = "",
                                    grade: str = "", prefer_period: bool = False) -> Dict:
        """裁决优先级（依据《标准化法》与综合排放标准自身的适用条款）：
           ① 行业标准 > 综合/通用标准
           ② 地方标准 > 国家标准（地标严于国标时，属地内执行地标）
           ③ 现行 > 已替代
        """
        # grade 必须一起传下去：否则"废水一级标准"会被只按地标层级裁决成 DB33 的
        # 冬季限值 2.5 mg/L（正确答案是 GB 8978 一级标准的 15 mg/L）。等级是用户
        # 已经明确给出的约束，裁决只能在其候选集内进行，不能跨等级重选。
        # prefer_period 同理：时期也是用户给出的约束，且它先于"同分从严"生效。
        cands = self.lookup_limit(pollutant, medium, grade, region=region,
                                  prefer_period=prefer_period).get("records", [])
        if not cands:
            return {"error": "无候选标准"}

        def rank(rec):
            s = self._std_map.get(rec["std_id"], {})
            lvl = s.get("level", "national")
            app = s.get("applicability", {})
            app_ind = [str(i) for i in (app.get("industry") or [])]
            is_generic = any(g in x for x in app_ind for g in ("通用", "全国"))
            is_local_hit = bool(region) and region in (app.get("region") or [])
            ind_hit = bool(industry) and any(industry in x for x in app_ind)
            # 行业不匹配且该标准并非通用标准 → 降级。
            # 例：问"工业废水"时，只管城镇污水处理厂的 DB33 2169 必须让位给综合排放标准。
            # 少了这一档，就会仅凭"地标优先"把错误的专项标准推给用户。
            ind_mismatch = 0 if (not industry or is_generic or ind_hit) else 1
            return (
                0 if s.get("status") == "active" else 1,        # ① 时效性
                ind_mismatch,                                    # ② 适用行业
                0 if is_local_hit else 1,                        # ③ 属地优先
                0 if lvl in ("industry", "local") else 1,        # ④ 标准层级
                # ⑤ 时期匹配：与查询的时期限定一致者优先（见 _cond_has_period）
                int(_cond_has_period(rec.get("condition", "")) != bool(prefer_period)),
                # ⑥ 同分从严：限值更小者优先。
                # 写成 -value 再升序 = 值最大者排第一，与"从严"正好相反；
                # 缺值不能当 0（当 0 会排到最前），统一沉底。
                float(rec["value"]) if rec.get("value") is not None else float("inf"),
            )

        cands = sorted(cands, key=rank)
        winner = cands[0]
        reasons = []
        ws = self._std_map.get(winner["std_id"], {})
        if ws.get("status") == "active":
            reasons.append("该标准为现行有效版本")
        if region and region in (ws.get("applicability", {}).get("region") or []):
            reasons.append(f"{region} 地方标准在本行政区域内优先适用")
        if industry:
            app_ind = ws.get("applicability", {}).get("industry") or []
            if any(industry in str(x) for x in app_ind):
                reasons.append(f"该标准适用对象「{industry}」，与问题场景一致")
            else:
                # 注意不能直接把 app_ind 插进 f-string —— 那会把 Python 列表的
                # repr（['通用']）原样写进给用户看的结论里。
                reasons.append(f"该标准为通用标准（适用对象："
                               f"{'、'.join(str(x) for x in app_ind)}），覆盖「{industry}」场景")
        if len(cands) > 1:
            reasons.append(f"另有 {len(cands) - 1} 个候选标准被其覆盖，详见 candidates")
        self._log("resolve_applicable_standard",
                  {"pollutant": pollutant, "medium": medium, "region": region}, True, len(cands))
        return {
            "applicable": {
                "std_id": winner["std_id"], "title": winner.get("std_title", ""),
                "grade": winner["grade"], "condition": winner["condition"],
                "value": winner["value"], "value_display": winner.get("value_display") or "",
                "unit": winner["unit"],
            },
            "reasons": reasons,
            "candidates": [{"std_id": c["std_id"], "grade": c["grade"], "condition": c["condition"],
                            "value": c["value"], "unit": c["unit"],
                            "is_current": c.get("is_current")} for c in cands],
        }

    # ---------------- 8. 环评类别判定
    def classify_eia_category(self, project_type: str, scale: str = "") -> Dict:
        """依据《建设项目环境影响评价分类管理名录（2021年版）》判定。

        ⚠️ 此处内置的是**结构演示表**，覆盖少量常见行业。
        生产环境必须从名录官方文本构建完整规则表（约 50 大类 / 200+ 小类）。
        """
        table = [
            {"cat": "污水处理及其再生利用", "kw": ["污水处理", "污水厂"],
             "rules": [("日处理能力>=10万吨", "报告书"), ("日处理能力<500吨", "登记表"),
                       ("其他", "报告表")]},
            {"cat": "危险废物贮存", "kw": ["危废贮存", "危险废物贮存", "暂存"],
             "rules": [("全部", "报告表（依据2021年版名录相关条目）")]},
            {"cat": "机制纸及纸板制造", "kw": ["造纸"],
             "rules": [("全部", "报告书")]},
            {"cat": "餐饮服务", "kw": ["餐饮", "饭店"],
             "rules": [("全部", "登记表")]},
        ]
        hit = next((t for t in table if any(k in project_type for k in t["kw"])), None)
        self._log("classify_eia_category", {"project_type": project_type}, hit is not None, 1 if hit else 0)
        if not hit:
            return {
                "error": f"演示表未覆盖「{project_type}」",
                "hint": "名录未列出的建设项目不需办理环评手续，但结论必须回到官方名录原文核对",
                "demo_table_only": True,
            }
        matched = "其他"
        for cond, res in hit["rules"]:
            if "全部" in cond:
                matched = res
                break
            if scale and cond.startswith("日处理能力>=") and scale:
                try:
                    num = float(re.sub(r"[^0-9.]", "", scale))
                    if num >= 10 and "万吨" in cond:
                        matched = res
                        break
                except ValueError:
                    pass
            if scale and cond.startswith("日处理能力<") and "吨" in scale:
                matched = res
        return {
            "project_type": project_type, "scale": scale,
            "matched_category": hit["cat"], "suggested_type": matched,
            "source_std_id": "名录-2021",
            "demo_table_only": True,
            "must_verify": "环评类别直接决定合规成本与审批流程，结论必须回溯《名录》官方文本",
        }

    # ---------------- 工具声明（喂给 LLM 的 schema）
    @staticmethod
    def declarations() -> List[Dict]:
        return [
            {"name": "search_standard",
             "description": "按自然语言检索环保标准条文。适用于不知道具体标准号时的探索式查询。",
             "parameters": {"query": "检索词，如'地表水III类COD限值'",
                            "medium": "water_surface|water_wastewater|air_ambient|air_emission|noise|solid_waste|eia|monitoring",
                            "grade": "Ⅰ/Ⅱ/Ⅲ/Ⅳ/Ⅴ 或 一级/二级/三级",
                            "region": "省份名", "top_n": "返回条数，默认6"}},
            {"name": "get_clause",
             "description": "按条文 ID 取原文全文。生成答案引用前必须调用，禁止凭记忆复述条文。",
             "parameters": {"clause_id": "形如 GB 3838-2002#4.1"}},
            {"name": "check_standard_status",
             "description": "校验标准是否现行有效。任何引用标准号之前都必须调用。",
             "parameters": {"std_id": "标准号", "as_of": "截止日期 YYYY-MM-DD，默认今天"}},
            {"name": "lookup_limit",
             "description": "结构化查询污染物浓度/排放限值，返回带适用条件的精确数值。",
             "parameters": {"pollutant": "COD|NH3-N|TP|PM10|PM2.5|SO2|NO2|石油类|噪声",
                            "medium": "介质", "grade": "等级", "condition": "适用条件", "region": "属地",
                            "prefer_period": "问题是否限定时期（冬季/汛期/具体月份），true 时优先取带时段条件的记录"}},
            {"name": "judge_compliance",
             "description": "达标判定：给实测值，返回是否超标、超标倍数，自动处理单位换算。",
             "parameters": {"measured": "实测数值", "pollutant": "污染物", "medium": "介质",
                            "grade": "等级", "unit": "实测值单位", "condition": "适用条件",
                            "region": "属地", "margin": "安全余量，0-1",
                            "prefer_period": "问题是否限定时期"}},
            {"name": "trace_standard_chain",
             "description": "追溯标准的替代链条，确认当前应引用的版本。",
             "parameters": {"std_id": "标准号"}},
            {"name": "resolve_applicable_standard",
             "description": "当国标/行标/地标同时存在时，裁决应适用哪一个。涉及多标准冲突时必须调用。",
             "parameters": {"pollutant": "污染物", "medium": "介质",
                            "region": "属地省份", "industry": "行业", "grade": "等级约束",
                            "prefer_period": "问题是否限定时期，true 时优先取带时段条件的记录"}},
            {"name": "classify_eia_category",
             "description": "判定项目应编制环境影响报告书/报告表/登记表。",
             "parameters": {"project_type": "项目类型", "scale": "规模描述"}},
        ]
