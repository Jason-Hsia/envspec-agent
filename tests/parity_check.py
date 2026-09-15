# -*- coding: utf-8 -*-
"""一致性校验 · Python 侧

前端内置了一份 JS 版引擎，用于零后端运行。但「两份实现」是长期风险：
一旦漂移，用户在前端看到的结果就与评测报告对不上，整个 Demo 失去可信度。

这个脚本把这件事变成可执行的检查：
    1. 用 Python Agent 跑一遍黄金集，记录规划 / 引用 / 拒答 / 答案哈希
    2. 调 Node 跑同一份数据的前端内置引擎
    3. 逐题逐字段比对，任何差异都以非零码退出

用法：
    python tests/parity_check.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.loop import EnvSpecAgent          # noqa: E402
from src.config import DATA_DIR                  # noqa: E402

JS_RESULT = ROOT / "out" / "_js_parity.json"


def find_node() -> str:
    """定位 node，不写死任何机器上的绝对路径。

    顺序：环境变量 NODE → PATH → 常见安装位置。
    找不到就报清楚的错，而不是让 subprocess 抛一个看不懂的异常。
    """
    env = os.environ.get("NODE")
    if env and Path(env).exists():
        return env
    which = shutil.which("node")
    if which:
        return which
    for pat in (r"C:\Program Files\nodejs\node.exe",
                r"C:\Program Files (x86)\nodejs\node.exe",
                "/usr/local/bin/node", "/usr/bin/node"):
        if Path(pat).exists():
            return pat
    raise SystemExit("[error] 找不到 node。请安装 Node.js，或用 NODE 环境变量指定其路径。")


NODE = find_node()


def py_results():
    golden = json.loads((DATA_DIR / "golden_set.json").read_text(encoding="utf-8"))["items"]
    agent = EnvSpecAgent()
    out = {}
    for it in golden:
        res = agent.run(it["query"])
        out[it["id"]] = {
            "query": it["query"],
            "plan": [p["tool"] for p in res["plan"]],
            "citations": [{"idx": c["idx"], "std_id": c["std_id"], "table": c.get("table", ""),
                           "clause_no": c.get("clause_no", ""), "value": c.get("value"),
                           "unit": c.get("unit", "")} for c in res["citations"]],
            "refused": res["refused"],
            "stale_warnings": res["stale_warnings"],
            "n_limits": len(res["limits"]),
            "answer_sha": hashlib.sha256(res["answer"].encode("utf-8")).hexdigest()[:16],
            "answer_len": len(res["answer"]),
        }
    return out


def run_node():
    r = subprocess.run([NODE, str(ROOT / "tests" / "engine_parity.js"), str(JS_RESULT)],
                       capture_output=True, text=True, encoding="utf-8")
    sys.stdout.write(r.stdout or "")
    if r.returncode != 0:
        sys.stderr.write(r.stderr or "")
        raise SystemExit("[error] Node 侧执行失败")


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("=" * 66)
    print("前端内置引擎 与 Python 后端 一致性校验")
    print("=" * 66)

    print("\n[1/2] 运行 Python Agent ...")
    py = py_results()
    print(f"      完成 {len(py)} 题")

    print("[2/2] 运行前端 JS 引擎 ...")
    run_node()
    js = json.loads(JS_RESULT.read_text(encoding="utf-8"))

    fields = ["plan", "refused", "stale_warnings", "n_limits", "answer_sha", "answer_len", "citations"]
    diffs = []
    for cid in sorted(py):
        if cid not in js:
            diffs.append((cid, "缺失", "JS 侧无该题结果"))
            continue
        for f in fields:
            a, b = py[cid].get(f), js[cid].get(f)
            if a != b:
                diffs.append((cid, f, f"py={a!r}  js={b!r}"))

    if not diffs:
        print("\n" + "=" * 66)
        print(f"✅ 完全一致：{len(py)} 题 × {len(fields)} 个字段，零差异")
        print("=" * 66)
        print("   含义：前端无需后端即可给出与评测报告完全一致的结果，")
        print("        两份实现未发生漂移。改动任一侧后请重跑本脚本。")
        return 0

    print("\n" + "=" * 66)
    print(f"❌ 发现 {len(diffs)} 处不一致")
    print("=" * 66)
    for cid, f, d in diffs[:40]:
        print(f"  [{cid}] {f:<16} {d[:240]}")
    if len(diffs) > 40:
        print(f"  ... 另有 {len(diffs) - 40} 处")
    return 1


if __name__ == "__main__":
    sys.exit(main())
