# -*- coding: utf-8 -*-
"""稠密检索。

默认使用 hashing + TF-IDF 权重的本地向量（零依赖、可离线、可复现）。
接入真实 Embedding 时把 _encode 换成 API 调用即可，其余逻辑不变。
"""
import hashlib
import math
from collections import Counter
from typing import Dict, List, Tuple

from src.config import EMBED_DIM, EMBED_PROVIDER
from .tokenizer import tokenize


def _build_idf(doc_tokens: List[List[str]]) -> Dict[str, float]:
    """语料级 IDF（与 bm25.py 用同一个公式，保持两条通道口径一致）。

    这条通道原先完全没有 IDF：条文之间共享大量模板表述（GB 3095 表1 的四行，
    章节标题、适用范围、单位说明逐字相同，只有污染物名不同），经余弦归一化后
    这些高频词会吃掉绝大部分权重，真正有判别力的实体 token 被淹没 —— 表现为
    "环境空气 PM2.5 二级标准年平均限值" 的正确条文掉到向量第 7 名。
    """
    df = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1
    N = len(doc_tokens)
    return {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}


def _hash_vec(tokens: List[str], dim: int = EMBED_DIM,
              idf: Dict[str, float] = None) -> List[float]:
    vec = [0.0] * dim
    if not tokens:
        return vec
    tf = Counter(tokens)
    n = len(tokens)
    for t, c in tf.items():
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        # idf=None 时退化为无 IDF 的旧行为（仅用于单测）。
        # 语料里不存在的查询词权重记 0：它本来就匹配不到任何文档，
        # 不该占着归一化的份额去稀释真正能匹配上的词。
        w_idf = idf.get(t, 0.0) if idf is not None else 1.0
        vec[idx] += sign * (1 + math.log(c)) * \
            (math.log(1 + n / (1 + c)) if c < n else 1.0) * w_idf
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))   # 已归一化


class VectorIndex:
    def __init__(self, provider: str = None):
        self.provider = provider or EMBED_PROVIDER
        self.doc_ids: List[str] = []
        self.vecs: List[List[float]] = []
        self.idf: Dict[str, float] = {}

    def _encode(self, text: str, for_query: bool = False) -> List[float]:
        if self.provider == "api":
            return self._encode_api(text)
        return _hash_vec(tokenize(text, for_query=for_query), EMBED_DIM, self.idf)

    def _encode_api(self, text: str) -> List[float]:
        """接入真实 Embedding 服务的挂载点。"""
        raise NotImplementedError(
            "EMBED_PROVIDER=api 时请在此接入：openai.embeddings.create(model=EMBED_MODEL, input=text)"
        )

    def fit(self, corpus: List[Tuple[str, str]]) -> "VectorIndex":
        self.doc_ids, self.vecs = [], []
        # 先把全语料分词一次：IDF 必须由整个语料统计出来，且文档侧与查询侧共用同一张表，
        # 否则两侧权重不同源，余弦相似度就没有意义。
        doc_tokens = [tokenize(text) for _doc_id, text in corpus]
        self.idf = _build_idf(doc_tokens)
        for (doc_id, _text), toks in zip(corpus, doc_tokens):
            self.doc_ids.append(doc_id)
            self.vecs.append(_hash_vec(toks, EMBED_DIM, self.idf))
        return self

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        qv = self._encode(query, for_query=True)
        sims = [(self.doc_ids[i], _cosine(qv, v)) for i, v in enumerate(self.vecs)]
        sims.sort(key=lambda x: -x[1])
        out = [(d, max(0.0, s)) for d, s in sims[:top_k] if s > 0]
        if not out:
            return []
        hi = out[0][1] or 1.0
        return [(d, s / hi) for d, s in out]
