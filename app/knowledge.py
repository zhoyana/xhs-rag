"""知识库文档构建：把业务笔记转换为 LangChain Document 并切分。"""
import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 秋招知识点：RAG 的典型数据链路是 Load → Transform/Split → Embed → Store → Retrieve → Generate。
# 本文件只负责 Transform/Split；职责单一，后续替换 Embedding 或向量库时不用改这里。
# 第一版按字符计数，便于初学和人工检查。中文通常没有空格，因此显式加入中文标点。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
TOPIC_PATTERN = re.compile(r"#([^#\n]+?)\[话题\]#")
# 视觉模型转写的图片文字并入正文时加的标记，便于检索命中后分辨内容来源。
IMAGE_TEXT_HEADER = "【图片文字】"


def text_hash(text: str) -> str:
    """内容指纹：相同文本总会得到相同结果，用于避免重复索引。"""
    # 秋招知识点：SHA-256 在这里用于内容寻址和幂等判断，不用于密码存储。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_topics(content: str) -> tuple[str, list[str]]:
    """把小红书话题从正文移到 metadata，避免生成只有标签的低信息 Chunk。"""
    topics = list(dict.fromkeys(topic.strip() for topic in TOPIC_PATTERN.findall(content)))
    cleaned = TOPIC_PATTERN.sub("", content)
    # 连续话题常写成 `#话题[话题]##下一个[话题]#`，移除后可能残留一个 #。
    cleaned = re.sub(r"(?m)^\s*#+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip(" #\t\r\n")
    return cleaned, topics


def document_hash(document: Document) -> str:
    """标题、正文或话题任一变化，都应使派生索引失效并安全重建。"""
    metadata = document.metadata
    topics = "\n".join(metadata.get("topics", []))
    return text_hash(f"{metadata['title']}\n{document.page_content}\n{topics}")


def note_document(note: dict) -> Document:
    """将 notes 表中的业务数据转换为 LangChain 的标准文档。"""
    # 秋招知识点：Document 将正文与 metadata 分离，是 LangChain 检索链的标准输入。
    title = note.get("title", "").strip()
    content, topics = extract_topics(note.get("content", "").strip())
    # 小红书大量笔记把内容写在长图里，正文只剩话题标签；视觉模型转写的 image_text
    # 才是真正的内容。不并进正文，这些笔记就永远索引为空、收藏完搜不到。
    image_text = str(note.get("image_text") or "").strip()
    if image_text:
        content = (f"{content}\n\n{IMAGE_TEXT_HEADER}\n{image_text}" if content
                   else f"{IMAGE_TEXT_HEADER}\n{image_text}")
    if not title or not content:
        raise ValueError("标题和正文不能为空，无法建立知识库文档。")

    # 标题不参与切分，防止短标题独占一个 Chunk；Embedding 时再给每段补充标题上下文。
    page_content = content
    return Document(
        page_content=page_content,
        metadata={
            "note_id": note["note_id"],
            "title": title,
            "source_type": "xiaohongshu",
            "source_url": note.get("source_url", ""),
            "topics": topics,
        },
    )


def split_note(note: dict) -> tuple[Document, list[Document]]:
    """构建原始文档，并使用 LangChain 递归切分为可检索片段。"""
    # 秋招知识点：递归字符切分优先保留段落/句子边界，比固定长度硬切更有语义完整性。
    document = note_document(note)
    splitter = RecursiveCharacterTextSplitter(
        # 秋招知识点：overlap 用少量重复换取跨边界上下文，但过大会增加向量存储和召回冗余。
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        # 秋招知识点：start_index 提供可追溯性，可把检索结果定位回原文位置。
        add_start_index=True,
    )
    chunks = splitter.split_documents([document])
    for index, chunk in enumerate(chunks):
        # chunk_index 表示阅读顺序；start_index 由 LangChain 计算原文起点。
        chunk.metadata["chunk_index"] = index
        chunk.metadata["content_hash"] = text_hash(chunk.page_content)
    return document, chunks
