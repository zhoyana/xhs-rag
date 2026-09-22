"""SQLite FTS5 全文检索：中文单字切分与查询表达式构造。

这一层存在的唯一理由，是 FTS5 内置分词器对中文等于失效：

- 默认的 unicode61 把**连续汉字当成一个 token**，于是"小红书运营技巧"整串进
  倒排索引，查询"运营"永远匹配不上——中文两字词在 FTS 上直接落空。
- trigram 分词器要求查询串至少 3 个字符，"运营""排版"这类两字词同样失效。

这里的做法是在**写入与查询两侧**都先把 CJK 字符逐字用空格隔开，让每个汉字
成为独立 token，任意长度的子串就都能命中；非 CJK 部分（英文单词、型号、数字）
原样保留，仍交给 unicode61 处理。两侧必须使用同一套切分，否则索引与查询不在
同一个 token 空间里，表现为"索引建了却一条都搜不到"。

查询侧不是把整句单字化后直接扔给 MATCH：那样在长查询下几乎不可能命中。这里按
标点与空白切成片段，片段内用**短语查询**（引号）保证字符连续相邻，片段之间用
OR 连接——精确性由短语负责，召回由 OR 负责，最终排序交给 BM25。
"""
import re

# CJK 统一表意文字（基本区 + 扩展 A + 兼容区）、日文假名、韩文音节。
# 覆盖假名和韩文，是因为收藏内容里会出现日文/韩文专有名词，它们同样是
# "连续字符被当成一个 token"的受害者，一起处理不需要额外的分支。
_CJK = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)

# 片段边界：空白、全角空格、ASCII 标点、CJK 标点。
# 刻意不含下划线——unicode61 把 `_` 当 token 字符，若在这里把它当分隔符，
# 查询 `a_b` 会被切成两个 token，与索引里的单词 `a_b` 对不上。
_BOUNDARY = re.compile(
    r"[\s\u3000!\"#$%&'()*+,\-./:;<=>?@\[\\\]^`{|}~"
    r"，。、；：！？（）【】「」『』《》“”‘’…—·]+"
)

# 与 unicode61 的取词规则对齐：连续的字母/数字/下划线构成一个 token。
# 用它而不是 split()，是为了让 "a+b" 在查询侧也变成 a 与 b 两个 token，
# 与索引侧 unicode61 的处理一致。
_WORD = re.compile(r"\w+", re.UNICODE)

# 片段长度到多少以内，就整体当一个短语查。超过这个长度的整句短语基本不可能
# 命中（用户常把整句话粘进搜索框），改由相邻双字组合兜住。
MAX_PHRASE = 6


def segment(text: str) -> str:
    """把 CJK 字符逐字空格隔开，其余内容原样保留。

    CJK 字符**前后都**留空格：unicode61 视字母为同类，
    "小红书abc" 若只在字间插空格会切出 "书abc" 这个混合 token。
    """
    if not text:
        return ""
    parts = []
    for char in text:
        if _CJK.match(char):
            parts.append(" ")
            parts.append(char)
            parts.append(" ")
        else:
            parts.append(char)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def tokens(text: str) -> list[str]:
    """切分成 token 序列，供写入索引与构造查询表达式共用。"""
    return _WORD.findall(segment(text))


def build_match_query(query: str) -> str | None:
    """把自然语言查询转成 FTS5 MATCH 表达式；没有可用字词时返回 None。

    返回 None 表示"这一路无法工作"（空查询、纯标点），调用方据此跳过关键词
    召回，而不是拿一个空表达式去匹配全表。

    两级粒度并列，因为中文查询的长度分布是两个极端：

    - 短片段整串当短语（`"排 版"`）最精确，是首选。
    - 长片段（用户把整句话粘进来："怎么做预测基线"）整串当短语几乎不可能命中，
      于是拆成**相邻双字组合**——中文里双字词占绝大多数，bigram 是既不严到
      空手、又不宽到等于扫全表的粒度。单字 OR 试过，噪音太大，"的""了"这类
      高频字会命中一切。

    两者用 OR 并列：精确性由短语负责，召回由 bigram 负责，排序交给 BM25——
    整串命中的文档自然拿到更高分，bigram 只是保证它别掉出候选集。
    """
    phrases: list[str] = []
    for fragment in _BOUNDARY.split(query or ""):
        words = tokens(fragment)
        if not words:
            continue
        if len(words) == 1:
            phrases.append(words[0])
            continue
        if len(words) <= MAX_PHRASE:
            phrases.append(" ".join(words))
        phrases.extend(
            " ".join(words[index:index + 2]) for index in range(len(words) - 1))
    if not phrases:
        return None
    # 去重保序：短片段的整串短语往往与它的某个 bigram 完全相同。
    unique = list(dict.fromkeys(phrases))
    # 引号把每个短语变成字面量：既表达"这几个字相邻"，也让 AND/OR/NEAR 这类
    # FTS5 保留字在短语内部失去语法含义，无需另外转义。
    return " OR ".join(f'"{phrase}"' for phrase in unique)
