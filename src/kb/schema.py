# -*- coding: utf-8 -*-
"""数据模型与校验。

设计要点：
1. Standard / Clause / LimitRecord 三层分离 —— 限值必须结构化，不能留在文本里。
2. 每个对象带 verified 标记，未核验数据可入库但会在评测中被单独统计。
3. 替代链是独立结构，支撑 trace_standard_chain 工具与时效性判定。
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

VALID_MEDIA = {
    "water_surface", "water_wastewater", "water_ground",
    "air_ambient", "air_emission",
    "soil", "noise", "solid_waste", "eia", "monitoring",
}
VALID_STATUS = {"active", "superseded", "abolished", "draft"}


@dataclass
class Standard:
    std_id: str
    title: str
    level: str                      # national | industry | local | group
    medium: List[str] = field(default_factory=list)
    issuing_body: str = ""
    publish_date: str = ""
    effective_date: str = ""
    status: str = "active"
    replaces: List[str] = field(default_factory=list)
    replaced_by: List[str] = field(default_factory=list)
    applicability: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False

    def validate(self) -> List[str]:
        errs = []
        if self.status not in VALID_STATUS:
            errs.append(f"{self.std_id}: 非法 status={self.status}")
        for m in self.medium:
            if m not in VALID_MEDIA:
                errs.append(f"{self.std_id}: 未知介质 {m}")
        if self.status == "superseded" and not self.replaced_by:
            errs.append(f"{self.std_id}: 状态为 superseded 但未标注 replaced_by")
        if self.status == "active" and self.replaced_by:
            errs.append(f"{self.std_id}: 状态为 active 却存在 replaced_by，数据矛盾")
        return errs

    def is_valid_on(self, date_str: str) -> bool:
        """在给定日期该标准是否有效。注意：superseded 标准在其被替代前仍是有效的。"""
        if self.status in ("abolished", "draft"):
            return False
        if self.effective_date and date_str < self.effective_date:
            return False
        return True


@dataclass
class Clause:
    clause_id: str
    std_id: str
    clause_no: str
    path: List[str] = field(default_factory=list)
    clause_type: str = "management"   # definition|limit|method|management|monitoring
    text: str = ""
    refs: List[str] = field(default_factory=list)
    limit_ref: Optional[str] = None
    verified: bool = False

    @property
    def searchable_text(self) -> str:
        """用于稀疏检索的拼接文本。标准号 + 章节路径 + 正文，三者权重天然由词频体现。"""
        return f"{self.std_id} {self.clause_no} {' '.join(self.path)} {self.text}"

    @property
    def is_stale_trap(self) -> bool:
        """该条文所属标准是否已被替代 —— 由 build_kb 回填。"""
        return bool(getattr(self, "_std_superseded_by", []))


@dataclass
class LimitRecord:
    limit_id: str
    std_id: str
    table: str
    pollutant: str
    pollutant_alias: List[str]
    medium: str
    unit: str
    grade: str
    condition: str
    limit_type: str
    value: Optional[float] = None
    value_text: Optional[str] = None   # 非数值表述，如"不限"
    # 限值的展示形态，由数据（而非编程语言的浮点格式化）决定。
    # 同一份 JSON 里 5.0 在 Python 是 float、在 JS 是 number，若直接插值，
    # 同一份知识库会渲染出 "5.0" 和 "5" 两种文本；而标准原文写的就是 5.0。
    # 展示形态必须跟数据走，否则前端与后端的答案无法逐字节一致。
    value_display: str = ""

    def matches(self, pollutant: str = "", medium: str = "", grade: str = "") -> bool:
        def ok(field_val, query):
            if not query:
                return True
            return query in field_val or field_val in query
        p_ok = ok(self.pollutant, pollutant) or any(
            ok(a, pollutant) for a in self.pollutant_alias
        )
        return p_ok and ok(self.medium, medium) and ok(self.grade, grade)


def to_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
