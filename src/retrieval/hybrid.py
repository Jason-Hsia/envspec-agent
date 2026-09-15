# -*- coding: utf-8 -*-
"""混合检索：查询理解 -> 元数据过滤 -> 双路召回 -> RRF 融合 -> 时效性重排。

这是整个项目最核心的模块。三处不放入"标准 RAG 教程"的设计：

1. RRF 而非加权求和 —— BM25 与向量的分数量纲不可比，加权需要反复调参，
   RRF 只用排名，跨查询稳定得多。
2. 时效性参与重排 —— active 标准加分、superseded 标准降权但仍保留，
   保留是为了让下游工具能主动提示"你查的这条已被替代"。
3. 结构化过滤前置 —— 介质/等级/区域先在元数据层剪枝，避免检索在做
   向量算力的无用功，也避免"废水标准答成地表水标准"这类致命错误。
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import (BM25_TOP_K, FUSION_TOP_K, MIN_SCORE_THRESHOLD, RRF_K,
                        VECTOR_TOP_K, RERANK_TOP_N)
from src.kb.build_kb import load_index
from .bm25 import BM25
from .tokenizer import normalize_query
from .vector import VectorIndex

# ----------------------------------------------------------- 查询理解（槽位）

POLLUTANT_ALIASES = {
    "COD": ["cod", "codcr", "化学需氧量"],
    "NH3-N": ["氨氮", "nh3-n", "氨氮（以n计）"],
    "TP": ["总磷", "tp"],
    "TN": ["总氮", "tn"],
    "PM10": ["pm10", "可吸入颗粒物"],
    "PM2.5": ["pm2.5", "细颗粒物"],
    "SO2": ["so2", "二氧化硫"],
    "NO2": ["no2", "二氧化氮"],
    "石油类": ["石油类", "石油"],
    "噪声": ["噪声", "厂界噪声", "laeq"],
}

MEDIUM_ALIASES = {
    "water_surface": ["地表水", "地表水体", "江河", "湖泊", "水库"],
    "water_wastewater": ["废水", "污水", "排放口", "污水处理厂", "排放标准"],
    "air_ambient": ["环境空气", "大气环境", "空气质量"],
    "air_emission": ["废气", "有组织排放", "无组织排放"],
    "noise": ["噪声", "声环境"],
    "solid_waste": ["固废", "固体废物", "危险废物", "危废", "贮存"],
    "eia": ["环评", "环境影响评价", "报告书", "报告表", "登记表"],
    "monitoring": ["自行监测", "监测方案", "监测点位"],
}

# ⚠️ 等级识别必须用正则 + 有序候选，不能用 `alias in query` 的子串判断。
# 原因："iii类" 同时包含 "ii类" 和 "i类"，朴素子串匹配会把 III 类误判成 I、II、III 三类，
# 进而让限值过滤失效、取到错误档位 —— 这是环保检索里代价最高的一类 bug。
_ROMAN_NORM = str.maketrans({
    "Ⅰ": "i", "Ⅱ": "ii", "Ⅲ": "iii", "Ⅳ": "iv", "Ⅴ": "v",
    "ⅰ": "i", "ⅱ": "ii", "ⅲ": "iii", "ⅳ": "iv", "ⅴ": "v",
})
_ROMAN_TO_WATER_GRADE = {"i": "Ⅰ", "ii": "Ⅱ", "iii": "Ⅲ", "iv": "Ⅳ", "v": "Ⅴ",
                         "1": "Ⅰ", "2": "Ⅱ", "3": "Ⅲ", "4": "Ⅳ", "5": "Ⅴ"}

# 长候选优先，保证 "iii" 不被 "i" 抢先吃掉
_RE_WATER_GRADE = re.compile(r"(iii|ii|iv|i|v)\s*类")
# 声环境功能区类别是 0-4 类，语义与地表水类别完全不同，必须按介质区分
_RE_NOISE_GRADE = re.compile(r"([0-4])\s*类")
_RE_DISCHARGE_GRADE = re.compile(r"([一二三]|[1-3])\s*级")
_DISCHARGE_MAP = {"一": "一级", "二": "二级", "三": "三级",
                  "1": "一级", "2": "二级", "3": "三级"}

REGIONS = ["浙江省", "江苏省", "广东省", "上海市", "北京市", "杭州市"]

# ---- 受纳水体 → 排放等级 的翻译表
# GB 8978-1996 §4.2 是按**受纳水体**定级的，不是按限值严格程度定级：
#   排入 GB 3838 Ⅲ类水域（划定的保护区与游泳区除外）→ 执行一级标准
#   排入 GB 3838 Ⅳ、Ⅴ类水域                        → 执行二级标准
#   排入设置二级污水处理厂的城镇排水系统              → 执行三级标准
# 用户问"排入地表水Ⅲ类水域的废水执行几级标准"时，Ⅲ类描述的是**受纳水体**，
# 不是排放等级。少了这层翻译，裁决只能在三个等级之间按启发式挑一个 ——
# 挑错了答案就是 500 而非 100，而两个数字都会出现在答案正文里，
# 端到端的数值指标根本发现不了（见 tests/ 里的结论校验）。
_SURFACE_TO_DISCHARGE = {"Ⅰ": "一级", "Ⅱ": "一级", "Ⅲ": "一级",
                         "Ⅳ": "二级", "Ⅴ": "二级"}
_RE_SEWER_DISCHARGE = re.compile(r"(纳管|下水道|市政管网|城镇排水系统|排入城镇)")

# ---- 时期条件
# 同一标准对同一污染物可能给出两条限值，差别只在适用时期：
#   DB33/ 2169-2018 氨氮：常规时段 1.5 mg/L ｜ 11月1日至次年3月31日 2.5 mg/L
# 取哪一条跟"哪个限值更严"无关，只跟问题有没有限定时期有关。
# 之前的做法是拿"同分从严"当替代品——那等于用限值的严格程度去猜时期，
# 蒙对是巧合，蒙错就是给出错误的合规底线。
_RE_PERIOD = re.compile(r"(\d{1,2}\s*月|冬季|冬期|春季|夏季|秋季|汛期|非汛期|枯水期|丰水期)")


def infer_discharge_grade(q: str, slots: "QuerySlots") -> str:
    """把受纳水体类别翻译成排放标准等级。只对废水/废气类问题生效。"""
    explicit = next((g for g in slots.grades if g.endswith("级")), "")
    if explicit:
        return explicit
    if slots.is_comparison:
        return ""       # 对比题不锁等级
    if not ({"water_wastewater", "air_emission"} & set(slots.media)):
        return ""
    if _RE_SEWER_DISCHARGE.search(q):
        return "三级"
    for g in slots.grades:
        if g in _SURFACE_TO_DISCHARGE:
            return _SURFACE_TO_DISCHARGE[g]
    return ""

# 行业槽位：决定"城镇污水处理厂专项标准"是否该被排在"工业废水"问题前面。
# 缺了它，resolve_applicable_standard 会仅凭"地方标准优先"把 DB33 2169（只管城镇污水厂）
# 推给问工业废水的用户 —— 这是最典型的一类"看起来有理、实际答错"。
INDUSTRY_ALIASES = {
    "城镇污水处理厂": ["城镇污水处理厂", "城市污水处理厂", "污水处理厂", "污水厂"],
    "工业": ["工业废水", "工业企业", "工业"],
    "危险废物": ["危险废物", "危废"],
    "建设项目": ["建设项目", "项目"],
}


def extract_grades(low: str, media: List[str]) -> List[str]:
    """按介质选择对应的等级体系，避免'2类噪声'被当成'Ⅲ类水'。"""
    grades: List[str] = []

    def push(g):
        if g and g not in grades:
            grades.append(g)

    is_noise = "noise" in media
    is_surface = "water_surface" in media
    is_waste = "water_wastewater" in media
    no_media = not media

    if is_noise:
        for m in _RE_NOISE_GRADE.finditer(low):
            push(m.group(1) + "类")
    if is_surface or no_media:
        for m in _RE_WATER_GRADE.finditer(low):
            push(_ROMAN_TO_WATER_GRADE.get(m.group(1).lower()))
    if is_waste or no_media:
        for m in _RE_DISCHARGE_GRADE.finditer(low):
            push(_DISCHARGE_MAP.get(m.group(1)))
    return grades


@dataclass
class QuerySlots:
    raw: str
    std_refs: List[str] = field(default_factory=list)
    pollutants: List[str] = field(default_factory=list)
    media: List[str] = field(default_factory=list)
    grades: List[str] = field(default_factory=list)
    discharge_grade: str = ""      # 由受纳水体推导出的排放等级（见 infer_discharge_grade）
    period_stated: bool = False    # 查询是否限定了时期（"11月到次年3月" / 冬季 / 汛期）
    regions: List[str] = field(default_factory=list)
    industry: List[str] = field(default_factory=list)
    wants_limit: bool = False
    wants_eia_class: bool = False
    wants_status: bool = False
    is_comparison: bool = False
    as_of_date: str = "2026-09-14"

    def to_dict(self):
        return self.__dict__.copy()


_RE_STD_QUERY = re.compile(r"\b(?:GB|HJ|DB\d{2})\s*/?\s*T?\s*\d+(?:\.\d+)?\s*-?\s*\d{0,4}",
                           re.IGNORECASE)


def understand(query: str) -> QuerySlots:
    q = normalize_query(query)
    low = q.lower().translate(_ROMAN_NORM)
    s = QuerySlots(raw=q)

    # 标准号（允许只写"3838"这类残缺形态，交由模糊匹配补全）
    for m in _RE_STD_QUERY.finditer(q):
        s.std_refs.append(re.sub(r"\s+", " ", m.group(0)).upper())

    for canon, alias in POLLUTANT_ALIASES.items():
        if any(a.lower() in low for a in alias):
            s.pollutants.append(canon)

    for canon, alias in MEDIUM_ALIASES.items():
        if any(a in q for a in alias):
            s.media.append(canon)

    s.grades = extract_grades(low, s.media)

    for r in REGIONS:
        if r in q:
            s.regions.append(r)

    # 行业按别名长度降序匹配，"城镇污水处理厂"优先于"工业"这类宽泛词
    for canon, alias in sorted(INDUSTRY_ALIASES.items(),
                               key=lambda kv: -max(len(a) for a in kv[1])):
        if any(a in q for a in alias):
            s.industry.append(canon)

    s.wants_limit = bool(s.pollutants) or bool(
        re.search(r"(限值|标准值|多少|不得超过|最高允许)", q))
    s.wants_eia_class = bool(re.search(
        r"(报告书|报告表|登记表|环评类别|要不要做环评|环评吗|需不需要环评|要不要环评|做环评)", q))
    s.wants_status = bool(re.search(r"(现行|最新|废止|失效|替代|还有效)", q))
    # 对比意图必须显式识别。不能只看"出现多个介质"就判定为对比题 ——
    # "排入地表水III类水域的工业废水执行几级标准"里也同时出现地表水和废水，
    # 但它是单一问题（地表水只是用来定级的受纳水体），误判成对比会答偏。
    s.is_comparison = bool(re.search(r"(区别|对比|差异|不同|一样吗|哪个更|vs)", q))
    s.discharge_grade = infer_discharge_grade(q, s)
    s.period_stated = bool(_RE_PERIOD.search(q))
    return s


# ----------------------------------------------------------- 检索器

class HybridRetriever:
    def __init__(self, index: Dict = None):
        self.index = index or load_index()
        self.clauses = self.index["clauses"]
        self.clause_map = {c["clause_id"]: c for c in self.clauses}
        self.std_map = {s["std_id"]: s for s in self.index["standards"]}
        corpus = [(c["clause_id"], c["searchable_text"]) for c in self.clauses]
        self.bm25 = BM25().fit(corpus)
        self.vec = VectorIndex().fit(corpus)

    # ---- 元数据过滤：硬条件不满足直接剪枝
    def _passes_filter(self, c: Dict, s: QuerySlots) -> bool:
        if s.media:
            if not set(c["medium"]) & set(s.media):
                return False
        if s.regions:
            regions = c.get("region") or []
            if regions and not set(regions) & set(s.regions) and "全国" not in regions:
                return False
        return True

    def _rrf(self, lists: List[List[Tuple[str, float]]]) -> Dict[str, float]:
        fused: Dict[str, float] = {}
        for rl in lists:
            for rank, (doc_id, _score) in enumerate(rl, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    # ---- 重排：在 RRF 分之上叠加领域先验
    def _rerank(self, fused: Dict[str, float], s: QuerySlots) -> List[Tuple[str, float]]:
        if not fused:
            return []
        mx = max(fused.values())
        out = []
        for cid, sc in fused.items():
            c = self.clause_map[cid]
            base = sc / mx
            boost = 0.0

            # 1) 明确点名的标准号，命中即强提升
            if s.std_refs:
                norm_id = c["std_id"].upper().replace(" ", "")
                if any(r.replace(" ", "") in norm_id or norm_id in r.replace(" ", "")
                       for r in s.std_refs):
                    boost += 0.45

            # 2) 问限值时，限值类条文优先
            if s.wants_limit and c["clause_type"] == "limit":
                boost += 0.18

            # 3) 问环评类别时，管理类条文优先
            if s.wants_eia_class and c["clause_type"] in ("management", "method"):
                boost += 0.10

            # 4) 等级命中
            if s.grades and any(g in c["text"] for g in s.grades):
                boost += 0.10

            # 4b) 污染物命中：一张限值表里每行只是一个污染物，表标题、单位、
            #     等级说明逐字相同，BM25 与向量对"是哪一行"的判别力很弱
            #     （实测问 COD 时氨氮行会与之并列）。查询点名了污染物，
            #     就把对应那行显式提上来 —— 与上面"等级命中"同类的领域先验。
            if s.pollutants:
                names = [a for p in s.pollutants
                         for a in POLLUTANT_ALIASES.get(p) or [p.lower()]]
                text_low = c["text"].lower()
                if any(n and n in text_low for n in names):
                    boost += 0.15

            # 4c) 行业不匹配：把 tools 层已经验证过的裁决规则前置到召回排序里。
            #     问"工业废水"时，只管城镇污水处理厂的 DB33/2169 不能凭"地标优先"
            #     排到第一 —— 它的限值再低，也不是这道题该用的标准。
            #     （同一判断在 resolve_applicable_standard 里已经修过一轮，
            #       这里只是让召回排序与裁决口径一致。）
            if s.industry:
                std = self.std_map.get(c["std_id"], {})
                app_ind = [str(x) for x in
                           ((std.get("applicability") or {}).get("industry") or [])]
                generic = any("通用" in x or "全国" in x for x in app_ind)
                hit = any(k in x for x in app_ind for k in s.industry)
                if app_ind and not generic and not hit:
                    boost -= 0.15

            # 5) 时效性：现行标准加分；失效标准保留下沉，供下游提示
            if c["std_status"] == "active":
                boost += 0.12
            elif c["std_status"] == "superseded":
                boost -= 0.18

            # 6) 地方标准在属地查询下优先于国标（地标严于国标）
            if s.regions and c["std_level"] == "local" and \
                    set(c.get("region") or []) & set(s.regions):
                boost += 0.25

            out.append((cid, base + boost))
        out.sort(key=lambda x: -x[1])
        return out

    def retrieve(self, query: str, top_n: int = RERANK_TOP_N,
                 slots: QuerySlots = None) -> Tuple[List[Dict], QuerySlots]:
        s = slots or understand(query)

        bm = self.bm25.search(query, BM25_TOP_K)
        vc = self.vec.search(query, VECTOR_TOP_K)
        bm = [(d, sc) for d, sc in bm if self._passes_filter(self.clause_map[d], s)]
        vc = [(d, sc) for d, sc in vc if self._passes_filter(self.clause_map[d], s)]

        fused = self._rrf([bm, vc])
        ranked = self._rerank(fused, s)[:max(top_n, FUSION_TOP_K)][:top_n]
        ranked = [(cid, sc) for cid, sc in ranked if sc >= MIN_SCORE_THRESHOLD]

        results = []
        for cid, sc in ranked:
            c = dict(self.clause_map[cid])
            c["score"] = round(sc, 4)
            c["is_stale"] = c["std_status"] == "superseded"
            c["replacement"] = (c.get("std_replaced_by") or [None])[0] if c["is_stale"] else None
            results.append(c)
        return results, s

    def get_clause(self, clause_id: str) -> Optional[Dict]:
        return self.clause_map.get(clause_id)

    def max_score(self, query: str) -> float:
        res, _ = self.retrieve(query, top_n=1)
        return res[0]["score"] if res else 0.0
