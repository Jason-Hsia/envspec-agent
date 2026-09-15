# -*- coding: utf-8 -*-
"""前端构建脚本：把知识库索引与评测结果注入 template.html，产出单文件 index.html。

为什么要构建而不是让前端 fetch：
  单文件产物可以直接双击打开、可以发给别人、可以放进任何静态托管，
  不需要起服务、不需要配 CORS。数据在构建时从 out/kb_index/index.json 注入，
  因此前端用的索引与后端跑评测用的索引是同一份，不会漂移。

用法：
    python web/build_web.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUT_DIR          # noqa: E402
from src.kb.build_kb import load_index            # noqa: E402

WEB = ROOT / "web"
TEMPLATE = WEB / "template.html"
OUTPUT = WEB / "index.html"
PLACEHOLDER = "/*__ENVSPEC_DATA__*/null"


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    # 索引走 load_index()，保证与后端使用的是同一份（不存在则自动构建）
    index = load_index()
    golden = _read_json(DATA_DIR / "golden_set.json") or {"items": []}
    eval_report = _read_json(OUT_DIR / "eval_report.json")
    diagnosis = _read_json(OUT_DIR / "diagnosis.json")

    payload = {
        "generated_at": index.get("meta", {}).get("loaded_at", ""),
        "index": index,
        "golden": golden,
        "eval": eval_report,
        "diagnosis": diagnosis,
    }

    # 关键：转义 </ 防止 JSON 字符串里出现 </script> 提前闭合脚本标签
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    tpl = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in tpl:
        print("[error] 模板中未找到数据占位符", PLACEHOLDER)
        return 1
    html = tpl.replace(PLACEHOLDER, data_js)
    OUTPUT.write_text(html, encoding="utf-8")

    st = index["stats"]
    print(f"[ok] 已生成 {OUTPUT}")
    print(f"     标准 {st['n_standards']} / 条文 {st['n_clauses']} / 限值 {st['n_limits']} / 失效条文 {st['n_stale_clauses']}")
    print(f"     黄金集 {len(golden.get('items', []))} 题 | 评测报告 {'有' if eval_report else '缺失'} | 归因清单 {'有' if diagnosis else '缺失'}")
    print(f"     文件大小 {OUTPUT.stat().st_size / 1024:.1f} KB")
    if not eval_report:
        print("     [warn] 缺 out/eval_report.json，评测面板将提示先跑 run_eval")
    if not index.get("meta", {}).get("verified"):
        print("     [warn] 知识库为演示数据（verified=false），生产环境需重新灌入官方数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
