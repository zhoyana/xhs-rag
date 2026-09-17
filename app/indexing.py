"""把「笔记 → 可检索」串成一步：建文档、中文分块、生成向量。

索引与向量化在 HTTP 层是两个端点（`POST /notes/{id}/index`、
`POST /documents/{id}/embed`），拆开是因为它们可以独立重试；但用户视角只有一件事
——收藏完能搜到。本模块把两步组合成可复用的服务函数，让 HTTP 端点与 Agent 收藏工具
共用同一份实现，避免"手动点一次能搜、收藏后搜不到"这类两处逻辑漂移。
"""
from datetime import datetime, timezone

from .embeddings import EXPECTED_DIMENSION, MODEL_NAME, embed_texts, vector_to_blob
from .knowledge import document_hash, split_note


class EmptyContent(ValueError):
    """笔记没有可索引的文字（标题或正文都空）。

    继承 ValueError，是为了让只关心"内容不合规"的旧调用方不用改代码；
    语义上它属于业务跳过而不是程序故障——纯图片笔记没转写出文字时就会走到这里。
    """


def embed_document(store, document_id, *, embedder=None, model=MODEL_NAME, now=None) -> dict:
    """为尚无向量、或内容指纹已过期的 Chunk 生成向量。

    秋招知识点：只计算缺失/过期的部分，这是 Embedding 层的幂等与增量更新——
    重复调用不会重复消耗模型调用，只有内容改过才会重新计算。
    """
    embedder = embedder or embed_texts
    now = now or datetime.now(timezone.utc).isoformat()
    chunks = store.document_chunks(document_id)
    if not chunks:
        # "文档没有分块"和"分块都已有向量"是两种不同的状态，不能都返回 0。
        return {"status": "missing", "model": model, "dimension": EXPECTED_DIMENSION,
                "total": 0, "embedded": 0, "skipped": 0}

    pending = store.chunks_needing_embeddings(document_id, model)
    if pending:
        # 秋招知识点：切分内容与向量化输入可以不同——存进库里的是干净正文，
        # 而喂给模型时补上标题，给每段孤立的片段一点主题上下文。
        inputs = [f"标题：{chunk['title']}\n正文片段：{chunk['content']}" for chunk in pending]
        vectors = embedder(inputs)
        store.save_embeddings([{
            "chunk_id": chunk["id"],
            "model": model,
            "dimension": len(vector),
            "vector": vector_to_blob(vector),
            "content_hash": chunk["content_hash"],
        } for chunk, vector in zip(pending, vectors, strict=True)], now)

    return {
        "status": "embedded" if pending else "existing",
        "model": model,
        "dimension": EXPECTED_DIMENSION,
        "total": store.embedding_count(document_id, model),
        "embedded": len(pending),
        "skipped": len(chunks) - len(pending),
    }


def index_note(store, note_id, *, now=None) -> dict:
    """建立文档与分块（不生成向量）；内容为空时抛 EmptyContent。

    与端点同名的这个函数刻意**不**做向量化：手动重建索引时可以只重建分块、
    不花模型调用；需要一步到位就用 index_and_embed。
    """
    note = store.get(note_id)
    if note is None:
        raise KeyError(note_id)
    now = now or datetime.now(timezone.utc).isoformat()
    try:
        document, chunks = split_note(note)
    except ValueError as exc:
        # knowledge 层只会因为"标题或正文为空"抛 ValueError，这里统一转成业务异常。
        raise EmptyContent(str(exc)) from None
    saved, changed = store.index_document(note, document, chunks, document_hash(document), now)
    return {"status": "indexed" if changed else "existing", "document": saved,
            "chunk_count": saved["chunk_count"], "changed": changed}


def index_and_embed(store, note_id, *, embedder=None, model=MODEL_NAME, now=None) -> dict:
    """建索引 + 生成向量，一步到位。

    返回结构化摘要而**不抛异常**：收藏流程里"这篇笔记暂时没有可检索正文"
    不该让整次收藏失败，调用方据此如实向用户汇报即可。
    status 取值：indexed（新建/更新了索引）/ existing（内容没变，索引与向量都在位）
    / skipped（没有可索引文字）/ failed（由调用方包装，见 xhs_mcp._index_safely）。
    """
    now = now or datetime.now(timezone.utc).isoformat()
    try:
        indexed = index_note(store, note_id, now=now)
    except EmptyContent as exc:
        # searchable 在任何分支都返回，调用方才能稳定地判断"现在能不能搜到"。
        return {"status": "skipped", "note_id": note_id, "searchable": False,
                "reason": str(exc)}

    embedded = embed_document(
        store, indexed["document"]["id"], embedder=embedder, model=model, now=now)
    return {
        "status": indexed["status"],
        "note_id": note_id,
        "document_id": indexed["document"]["id"],
        "chunk_count": indexed["chunk_count"],
        "model": embedded["model"],
        "embedded": embedded["embedded"],
        "skipped_vectors": embedded["skipped"],
        "vector_total": embedded["total"],
        # 收藏后能否立刻语义检索，取决于向量是否真的落库了，而不是索引建没建。
        "searchable": embedded["total"] > 0,
    }
