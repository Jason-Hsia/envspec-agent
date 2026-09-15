# -*- coding: utf-8 -*-
"""中文分词 + 领域词典。

环保领域分词的坑：通用分词器会把 "CODcr"、"HJ 2.1"、"GB 3838-2002"、
"PM2.5"、"氨氮" 切碎。这里用领域词典 + 正则保护先做一遍归一化。
"""
import re
from typing import List

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

DOMAIN_TERMS = [
    "环境影响评价", "环评", "排污许可", "自行监测", "竣工验收", "总量控制",
    "化学需氧量", "生化需氧量", "氨氮", "总磷", "总氮", "石油类", "悬浮物",
    "挥发性有机物", "颗粒物", "二氧化硫", "氮氧化物", "二氧化氮", "臭氧",
    "危险废物", "固体废物", "一般工业固体废物", "医疗废物",
    "地表水", "地下水", "环境空气", "厂界噪声", "声环境功能区",
    "一级标准", "二级标准", "三级标准", "浓度限值", "排放限值",
    "标准状态", "无组织排放", "有组织排放", "排污口", "监测点位",
]

# 保护模式：这些实体一旦被通用分词器切开就再也拼不回来，必须先整体摘出。
#
# 三个必须显式处理的坑（都是实测踩出来的，不是推演）：
#   ① 大小写 —— tokenize() 先做 lower()，模式里写死大写（GB/HJ/mg/L）就会全部失配。
#      不要假设输入的大小写，统一用 IGNORECASE。
#      注意 IGNORECASE 只管 ASCII/拉丁大小写，管不了 Unicode 大小写映射：
#      lower("Ⅲ类") == "ⅲ类"（U+2162 → U+2172），罗马数字那一条要把小写形式也写进字符类。
#   ② 词边界 —— \b 在 Python 里是 Unicode 语义（汉字算词字符），在 JS 里是 ASCII 语义。
#      "氨氮NH3-N" 在 Python 下 \b 不成立、在 JS 下成立，两套实现直接分叉。
#      改用 ASCII 字符类的否定环视，两边语义一致，也不会被相邻汉字卡住。
#   ③ 地标格式 —— "DB33/ 2169-2018" 斜杠后面没有 T，旧的 (?:\s*/\s*T)? 把它整个挡掉了，
#      而地标号恰恰是数据里最常见的一种。斜杠后的 T/Z 必须可选。
#   ④ 罗马数字 —— Ⅰ(U+2160) 与 ASCII 的 I 是两套字符，但指同一件事。
#      语料写"Ⅲ类"，用户敲"III类"，两边除了"类"字之外零重合。而且 ASCII 的
#      i/i/i 在中文文本里是罕见字符，一旦引入 IDF 就会霸占查询向量（实测把
#      正确条文挤出了向量前 20）。统一归一化为 ASCII 小写罗马数字。
_ROMAN_UPPER = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
_ROMAN_LOWER = "ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ"
_ROMAN_ASCII = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_ROMAN_TRANS = {ord(ch): _ROMAN_ASCII[i] for i, ch in enumerate(_ROMAN_UPPER)}
_ROMAN_TRANS.update({ord(ch): _ROMAN_ASCII[i] for i, ch in enumerate(_ROMAN_LOWER)})

PROTECT_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])(?:GB|HJ|DB\d{2}|GBZ)\s*(?:/\s*(?:T|Z)?)?"
               r"\s*\d+(?:\.\d+)?\s*[-–—]\s*\d{4}", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:PM2\.5|PM10|PM|NOx|SO2|NO2|CO2|O3|CODcr|COD|"
               r"BOD5|BOD|NH3-N|TOC|TSP|VOCs)(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:mg/L|mg/m3|ug/m3|dB\(A\)|dB|t/a|万t/a)",
               re.IGNORECASE),
    re.compile(r"[ivx]{1,4}类|\d+类", re.IGNORECASE),
]


def _spans(text: str) -> List[tuple]:
    """定位全部受保护实体，返回按起点排序、互不重叠的 [(start, end, entity)]。

    这里刻意不用「占位符替换 → 分词 → 还原」那套做法。降级分词器是逐字符切分的，
    占位符本身会被切成单字符，还原时找不到完整键，结果是实体丢失、\\x01 碎片
    却作为 token 留在了词表里（"35 ug/m3" 会整个消失）。直接按区间切片没有这个风险。
    """
    found = []
    for pat in PROTECT_PATTERNS:
        for m in pat.finditer(text):
            found.append((m.start(), m.end(), m.group(0)))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))    # 起点升序；同起点取更长的
    spans, end = [], -1
    for s, e, g in found:
        if s >= end:                                     # 与已选区间重叠则丢弃
            spans.append((s, e, g))
            end = e
    return spans


def _base_tokenize(seg: str) -> List[str]:
    """对不含受保护实体的片段做基础分词。"""
    if not seg.strip():
        return []
    if _HAS_JIEBA:
        return [w for w in jieba.lcut(seg) if w.strip()]
    # 降级：字符 unigram + bigram，对中文短查询仍可用
    clean = re.sub(r"\s+", "", seg)
    if not clean:
        return []
    return list(clean) + [clean[i:i + 2] for i in range(len(clean) - 1)]


def tokenize(text: str, for_query: bool = False) -> List[str]:
    text = text.lower().replace("μ", "u").translate(_ROMAN_TRANS)

    if _HAS_JIEBA and for_query:
        for t in DOMAIN_TERMS:
            jieba.add_word(t, freq=10_000)

    toks, pos = [], 0
    for s, e, g in _spans(text):
        toks += _base_tokenize(text[pos:s])
        # 实体整体成词，并压掉内部空白："GB 3838-2002" 与 "GB3838-2002" 归一为同一个词，
        # 否则用户少打一个空格就检索不到。
        toks.append(re.sub(r"\s+", "", g))
        pos = e
    toks += _base_tokenize(text[pos:])
    return [t for t in toks if t and t.strip()]


def normalize_query(q: str) -> str:
    """查询归一化：全角转半角、破折号统一、标准号补空格。"""
    table = str.maketrans("０１２３４５６７８９（）－—～", "0123456789()--~")
    q = q.translate(table)
    q = q.replace("—", "-").replace("–", "-")
    q = re.sub(r"\s+", " ", q)
    return q.strip()
