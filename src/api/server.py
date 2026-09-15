# -*- coding: utf-8 -*-
"""前端 Demo 的后端接口。零第三方依赖，只用标准库。

    python -m src.api.server                # 默认 http://127.0.0.1:8770
    python -m src.api.server --port 9000

路由：
    GET  /                -> web/index.html（前端会自动探测后端并切换执行路径）
    GET  /api/health      -> 存活探测 + 索引规模
    POST /api/ask         -> {"query": "..."} 真实跑一遍 Agent，返回与离线引擎同构的 JSON
    GET  /api/eval        -> 最近一次评测报告
    GET  /api/tools       -> 工具声明（可直接喂给 LLM 的 schema）

设计要点：
  · Agent 与索引延迟初始化，避免启动时就把索引读进内存。
  · 单线程锁保护 Tool.call_log —— EnvSpecTools 是有状态的，并发复用会串日志。
  · 不做鉴权、不绑定 0.0.0.0，这是本地演示工具，不该暴露到网络上。
"""
import argparse
import io
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.loop import EnvSpecAgent                     # noqa: E402
from src.config import LLM_PROVIDER, OUT_DIR                # noqa: E402
from src.tools.registry import EnvSpecTools                 # noqa: E402

WEB_DIR = ROOT / "web"

_agent = None
_tools = None
_lock = threading.Lock()


def get_agent():
    """延迟初始化：第一次请求时才建工具与索引。"""
    global _agent, _tools
    if _agent is None:
        _tools = EnvSpecTools()
        _agent = EnvSpecAgent(_tools)
    return _agent, _tools


class Handler(SimpleHTTPRequestHandler):
    server_version = "EnvSpecDemo/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # ---------------- 工具方法 ----------------
    def _send_json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, code: int = 200, ctype: str = "text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- GET ----------------
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/health":
            try:
                _, tools = get_agent()
                st = tools.index.get("stats", {})
                return self._send_json({
                    "ok": True, "provider": LLM_PROVIDER,
                    "index_stats": {
                        "n_standards": st.get("n_standards", 0),
                        "n_clauses": st.get("n_clauses", 0),
                        "n_limits": st.get("n_limits", 0),
                        "n_stale_clauses": st.get("n_stale_clauses", 0),
                    },
                    "verified": tools.index.get("meta", {}).get("verified", False),
                })
            except Exception as e:                                  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(e)}, 500)

        if path == "/api/eval":
            p = OUT_DIR / "eval_report.json"
            if not p.exists():
                return self._send_json({"error": "尚未生成评测报告，请先运行 python -m src.eval.run_eval"}, 404)
            return self._send_json(json.loads(p.read_text(encoding="utf-8")))

        if path == "/api/tools":
            return self._send_json({"tools": EnvSpecTools.declarations()})

        # 其余交给静态文件服务；/ 映射到 index.html
        if path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    # ---------------- POST ----------------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as e:
            return self._send_json({"error": f"请求体解析失败: {e}"}, 400)

        if path != "/api/ask":
            return self._send_json({"error": f"未知接口 {path}"}, 404)

        query = (payload.get("query") or "").strip()
        if not query:
            return self._send_json({"error": "query 不能为空"}, 400)
        if len(query) > 500:
            return self._send_json({"error": "query 过长（上限 500 字）"}, 400)

        try:
            agent, _ = get_agent()
            # 有状态对象必须串行执行，否则 call_log 会被并发请求交叉污染
            with _lock:
                res = agent.run(query)
            return self._send_json(res)
        except Exception as e:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._send_json({"error": f"执行失败: {e}"}, 500)

    def log_message(self, fmt, *args):
        # 静音静态资源访问日志，只保留接口调用
        if "/api/" in (self.path or ""):
            sys.stderr.write("  %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser(description="环保规范检索 Agent · 前端 Demo 后端")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    index_html = WEB_DIR / "index.html"
    if not index_html.exists():
        print("[warn] 未找到 web/index.html，请先运行 python web/build_web.py 生成")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 58)
    print("  环保规范检索 Agent · Demo 后端已启动")
    print(f"  http://{args.host}:{args.port}")
    print(f"  规划器 provider = {LLM_PROVIDER}")
    print("  Ctrl+C 停止")
    print("=" * 58)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        httpd.server_close()


if __name__ == "__main__":
    main()
