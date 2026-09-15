# -*- coding: utf-8 -*-
"""索引构建入口。

产出 out/kb_index/index.json，包含：
  clauses   : 可检索条文（含元数据，用于过滤与展示）
  limits    : 结构化限值表（供 lookup_limit 工具直查）
  standards : 标准元数据 + 时效性状态
  chain     : 替代链
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATA_DIR, INDEX_DIR
from src.kb import parser as P
from src.kb.schema import to_dict


def build(seed_path: Path = None) -> dict:
    seed_path = seed_path or (DATA_DIR / "standards_seed.json")

    standards, clauses, limits, extra = P.load_structured(seed_path)
    P.link_refs(clauses)

    errs = P.validate(standards, clauses)
    if errs:
        print("[warn] 数据校验问题：")
        for e in errs:
            print("   -", e)

    std_index = {s.std_id: s for s in standards}

    clause_records = []
    stale_count = 0
    for c in clauses:
        std = std_index.get(c.std_id)
        if getattr(c, "_std_superseded_by", []):
            stale_count += 1
        clause_records.append({
            "clause_id": c.clause_id,
            "std_id": c.std_id,
            "clause_no": c.clause_no,
            "path": c.path,
            "clause_type": c.clause_type,
            "text": c.text,
            "searchable_text": c.searchable_text,
            "refs": c.refs,
            "limit_ref": c.limit_ref,
            # ↓ 检索后可用的过滤维度（元数据过滤的抓手）
            "std_status": std.status if std else "unknown",
            "std_level": std.level if std else "unknown",
            "std_title": std.title if std else "",
            "std_effective_date": std.effective_date if std else "",
            "std_replaced_by": std.replaced_by if std else [],
            "medium": std.medium if std else [],
            "region": (std.applicability or {}).get("region", []) if std else [],
            "verified": std.verified if std else False,
        })

    index = {
        "clauses": clause_records,
        "limits": [to_dict(l) for l in limits],
        "standards": [to_dict(s) for s in standards],
        "chain": extra["chain"],
        "meta": extra["meta"],
        "stats": {
            "n_standards": len(standards),
            "n_clauses": len(clauses),
            "n_limits": len(limits),
            "n_stale_clauses": stale_count,
            "n_verified_standards": sum(1 for s in standards if s.verified),
        },
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / "index.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    st = index["stats"]
    print(f"[ok] 索引已写入 {out}")
    print(f"     标准 {st['n_standards']} 个 / 条文 {st['n_clauses']} 条 / 限值 {st['n_limits']} 条")
    print(f"     其中失效标准条文(时效性陷阱) {st['n_stale_clauses']} 条")
    return index


def load_index() -> dict:
    p = INDEX_DIR / "index.json"
    if not p.exists():
        return build()
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build()
