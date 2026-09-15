# -*- coding: utf-8 -*-
"""生成一份可读的命令行演示记录（写入 out/demo.txt）。

为什么单独写一个脚本、而不是 `python -m src.agent.loop "..." > out/demo.txt`：
Windows 控制台默认按 GBK 解码子进程 stdout，重定向后中文会变成乱码并**落盘固化**，
再打开就是一串 `鍦拌〃姘碷`。所以这里显式以 UTF-8 打开文件写入。

题目全部取自 data/golden_set.json —— 不另造问题。理由：
演示记录里的每一条都必须在评测报告里查得到，否则演示就成了"挑好看的展示"。
标注类别也照抄黄金集的 tags，不手写印象式标签。

用法：
    python -m src.demo_cli
"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.agent.loop import EnvSpecAgent      # noqa: E402
from src.config import DATA_DIR              # noqa: E402

OUT = ROOT / "out" / "demo.txt"
W = 72

# 覆盖四类能力边界：检索 / 裁决 / 时效 / 拒答
CASES = ["Q001", "Q007", "Q006", "Q003", "Q012"]


def main() -> int:
    golden = {it["id"]: it for it in
              json.loads((DATA_DIR / "golden_set.json").read_text(encoding="utf-8"))["items"]}
    agent = EnvSpecAgent()

    lines = [
        "=" * W,
        "环保规范检索 Agent · 命令行演示记录",
        "（离线模式：规则规划器 + 本地 BM25/哈希向量；数据为演示级，见 README §8）",
        "（题目均取自 data/golden_set.json，可在前端 Demo 里逐题复现）",
        "=" * W,
    ]

    for cid in CASES:
        it = golden[cid]
        res = agent.run(it["query"], verbose=False)
        calls = [c["tool"] for c in res["tool_calls"]]
        lines += [
            "",
            "─" * W,
            f"[{cid}] 类别：{it.get('category', '-')}｜预期拒答：{it.get('should_refuse')}",
            f"提问：{it['query']}",
            "─" * W,
            f"工具调用链：{' → '.join(calls) if calls else '（未调用）'}",
        ]
        if res["stale_warnings"]:
            lines.append(f"时效性拦截：{len(res['stale_warnings'])} 条 —— 已废止条文不得进入引用")
            for w in res["stale_warnings"]:
                # stale_warnings 的元素形态在不同工具分支下不一致（str / dict 都有）
                lines.append(f"  · {w if isinstance(w, str) else ' '.join(str(x) for x in w.values())}")
        lines += [
            "",
            res["answer"],
            "",
            f"[拒答={res['refused']} | 引用={len(res['citations'])} | 限值命中={len(res['limits'])}]",
        ]

    lines += [
        "",
        "=" * W,
        "说明：以上输出与前端 Demo（web/index.html）共用同一份索引与同一套逻辑，",
        "      两侧结果经 tests/parity_check.py 逐字段校验一致。",
        "=" * W,
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] 已写入 {OUT}（{len(lines)} 行，UTF-8）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
