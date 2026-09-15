# -*- coding: utf-8 -*-
"""稀疏检索（BM25）。

为什么环保场景离不开 BM25：用户大量输入"GB 3838-2002 表1 COD Ⅲ类"这类
强符号查询，这类查询的判别信息全在罕见 token 上，稠密向量表达不了。
"""
import math
from collections import Counter
from typing import Dict, List, Tuple

from .tokenizer import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: List[str] = []
        self.docs: List[Counter] = []
        self.df: Counter = Counter()
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0

    def fit(self, corpus: List[Tuple[str, str]]) -> "BM25":
        """corpus: [(doc_id, text), ...]"""
        self.doc_ids, self.docs, self.df, self.doc_len = [], [], Counter(), []
        for doc_id, text in corpus:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_ids.append(doc_id)
            self.docs.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] += 1
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # BM25+ 形式，避免高频词出现负 IDF
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        q_toks = tokenize(query, for_query=True)
        if not q_toks or not self.N:
            return []
        scores: Dict[int, float] = {}
        for t in set(q_toks):
            if t not in self.df:
                continue
            idf = self._idf(t)
            q_weight = q_toks.count(t) / len(q_toks)
            for i, tf in enumerate(self.docs):
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / denom * (0.5 + q_weight)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        if not ranked:
            return []
        hi = ranked[0][1] or 1.0
        # 归一化到 0-1，便于与向量分做加权融合与阈值判断
        return [(self.doc_ids[i], s / hi) for i, s in ranked]
