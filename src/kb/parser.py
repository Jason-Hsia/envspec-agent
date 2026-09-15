# -*- coding: utf-8 -*-
"""知识库解析层。

两个入口：
  load_structured()  —— 读取已结构化的 seed/生产数据
  parse_raw_standard() —— 从标准正文纯文本切分出条文（真实数据接入用）
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .schema import Clause, LimitRecord, Standard

# ---------------------------------------------------------------- 结构化加载

def load_structured(path: Path) -> Tuple[List[Standard], List[Clause], List[LimitRecord], dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    standards: List[Standard] = []
    for s in raw.get("standards", []):
        standards.append(Standard(**{k: v for k, v in s.items()
                                     if k in Standard.__dataclass_fields__}))

    std_map = {s.std_id: s for s in standards}

    clauses: List[Clause] = []
    for c in raw.get("clauses", []):
        c = dict(c)
        std_id = c["std_id"]
        clause_id = f"{std_id}#{c['clause_no']}"
        c["clause_id"] = clause_id
        # 未显式给出的字段补默认值
        for k in ("path", "refs"):
            c.setdefault(k, [])
        obj = Clause(**{k: v for k, v in c.items() if k in Clause.__dataclass_fields__})
        # 回填时效性标记：所属标准是否已被替代
        std = std_map.get(std_id)
        obj._std_superseded_by = list(std.replaced_by) if std else []
        obj._std_status = std.status if std else "unknown"
        clauses.append(obj)

    limits = [LimitRecord(**{k: v for k, v in l.items()
                             if k in LimitRecord.__dataclass_fields__})
              for l in raw.get("limits", [])]

    # 回填限值的展示形态。必须在 Python 侧定下来并写进索引，因为只有这里
    # 还保留着原始字面量：5.0 是 float（-> "5.0"）、100 是 int（-> "100"）。
    # 一旦序列化成 JSON 再被 JS 读走，这个区别就永久丢失了。
    for l in limits:
        if not l.value_display:
            l.value_display = str(l.value) if l.value is not None else (l.value_text or "")

    chain = raw.get("replacement_chain", [])
    meta = raw.get("_meta", {})
    return standards, clauses, limits, {"chain": chain, "meta": meta}


# ---------------------------------------------------------------- 条文级切分

# 条文编号形态： 4.1 / 4.1.1 / 6.2.3.4 ；表号： 表1 / 附录A
_RE_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\s+(\S.*)$")
_RE_TABLE = re.compile(r"^\s*(表\s*\d+[\.\d]*)\s*(.*)$")
_RE_CHAPTER = re.compile(r"^\s*(\d+)\s+([\u4e00-\u9fa5].*)$")


def parse_raw_standard(text: str, std_id: str) -> List[Clause]:
    """把标准正文切分为条文级 chunk。

    切分策略：以条文编号为切分锚点，保留章节路径（面包屑）写入 metadata。
    相比固定长度滑窗，这种做法能让"某一条"成为不可分割的检索单元，
    从而支撑精确引用（引用必须落到条文号）。
    """
    clauses: List[Clause] = []
    chapter = ""
    section_path: List[str] = []
    buf: List[str] = []
    cur_no = ""
    cur_type = "management"

    def flush():
        nonlocal buf, cur_no, cur_type
        if cur_no and buf:
            clauses.append(Clause(
                clause_id=f"{std_id}#{cur_no}",
                std_id=std_id,
                clause_no=cur_no,
                path=list(section_path),
                clause_type=cur_type,
                text=" ".join(x.strip() for x in buf if x.strip()),
            ))
        buf, cur_no, cur_type = [], "", "management"

    for line in text.splitlines():
        if not line.strip():
            continue

        m_chap = _RE_CHAPTER.match(line)
        m_tab = _RE_TABLE.match(line)
        m_cl = _RE_CLAUSE.match(line)

        if m_chap and not m_cl:
            flush()
            chapter = f"第{m_chap.group(1)}章 {m_chap.group(2).strip()}"
            section_path = [chapter]
            continue

        if m_tab:
            flush()
            section_path = [chapter, m_tab.group(1).replace(" ", "")]
            cur_no = f"{m_tab.group(1).replace(' ', '')}({len(clauses) + 1})"
            cur_type = "limit"
            buf = [line]
            continue

        if m_cl:
            flush()
            cur_no = m_cl.group(1)
            body = m_cl.group(2)
            depth = cur_no.count(".")
            section_path = [chapter] + section_path[1:depth] if depth else [chapter]
            cur_type = _infer_clause_type(body)
            buf = [line]
            continue

        buf.append(line)

    flush()
    return clauses


def _infer_clause_type(text: str) -> str:
    if re.search(r"(限值|标准值|不得超过|最高允许|浓度)", text):
        return "limit"
    if re.search(r"(监测|采样|分析方法|测定)", text):
        return "monitoring"
    if re.search(r"(定义|术语|是指|称为)", text):
        return "definition"
    if re.search(r"(应|不得|禁止|要求|程序)", text):
        return "management"
    return "management"


# ---------------------------------------------------------------- 元数据抽取

_RE_STD_NO = re.compile(
    r"\b((?:GB|HJ|DB\d{2}|GBZ|GB/T|HJ/T)\s*\d+(?:\.\d+)?\s*[-—]\s*\d{4})\b",
    re.IGNORECASE,
)


def extract_standard_refs(text: str) -> List[str]:
    """从条文正文中抽取被引用的标准号，用于构建 refs 图。

    注意归一化：全角破折号、多余空格统一掉，否则 'GB 3838—2002'
    和 'GB 3838-2002' 会被当成两个不同标准，这是环保语料的经典坑。
    """
    found = set()
    for m in _RE_STD_NO.finditer(text):
        s = m.group(1).replace("—", "-").replace("–", "-")
        s = re.sub(r"\s+", " ", s).upper().strip()
        s = re.sub(r"^([A-Z/]+)(\d)", r"\1 \2", s)
        found.add(s)
    return sorted(found)


def link_refs(clauses: List[Clause]) -> None:
    """回填条文间引用关系（标准内 + 跨标准）。"""
    by_no = {}
    for c in clauses:
        by_no.setdefault((c.std_id, c.clause_no), c.clause_id)

    std_ids = {c.std_id for c in clauses}
    for c in clauses:
        refs = set(c.refs)
        for s in extract_standard_refs(c.text):
            if s in std_ids:
                refs.add(f"{s}#*")          # 跨标准引用，具体条文由工具运行时解析
        c.refs = sorted(refs)


def validate(standards: List[Standard], clauses: List[Clause]) -> List[str]:
    errs: List[str] = []
    for s in standards:
        errs.extend(s.validate())
    seen = set()
    for c in clauses:
        if c.clause_id in seen:
            errs.append(f"条文 ID 重复: {c.clause_id}")
        seen.add(c.clause_id)
        if not c.text.strip():
            errs.append(f"条文正文为空: {c.clause_id}")
    return errs
