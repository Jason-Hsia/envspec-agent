# -*- coding: utf-8 -*-
"""全局配置。默认离线模式，可零 API 预算跑通全链路评测。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
INDEX_DIR = OUT_DIR / "kb_index"

# ---------------- 模型接入 ----------------
# offline           : 无外部依赖，规则化规划器 + 哈希向量检索
# openai_compatible : 兼容 OpenAI 协议的任意服务（含本地 vLLM / Ollama）
LLM_PROVIDER = os.environ.get("ENVSPEC_LLM_PROVIDER", "offline")
LLM_BASE_URL = os.environ.get("ENVSPEC_LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("ENVSPEC_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("ENVSPEC_LLM_MODEL", "gpt-4o-mini")

EMBED_PROVIDER = os.environ.get("ENVSPEC_EMBED_PROVIDER", "hash")  # hash | api
EMBED_MODEL = os.environ.get("ENVSPEC_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 512  # hash 模式下的向量维度

RERANK_ENABLED = os.environ.get("ENVSPEC_RERANK", "0") == "1"

# ---------------- 检索参数 ----------------
BM25_TOP_K = 20
VECTOR_TOP_K = 20
RRF_K = 60
FUSION_TOP_K = 15
RERANK_TOP_N = 6
MIN_SCORE_THRESHOLD = 0.05   # 低于此分判为"无依据"，触发拒答

# ---------------- Agent 参数 ----------------
MAX_TOOL_ROUNDS = 4          # 单次问答最多工具调用轮数，防死循环
MAX_TOOL_CALLS = 8

# ---------------- 评测门槛（发布门禁） ----------------
GATES = {
    "recall_at_5": 0.85,
    "mrr": 0.70,
    "citation_precision": 0.90,
    "numerical_accuracy": 0.90,
    "conclusion_accuracy": 0.95,     # 裁决题给出唯一结论且等于标注答案
    "stale_citation_rate": 0.0,      # 硬门槛：一票否决
    "tool_selection_accuracy": 0.90,
    "refusal_accuracy": 0.80,
}
