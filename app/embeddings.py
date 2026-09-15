"""Embedding 模型与向量二进制转换。

这个模块只负责“文本 ↔ 向量”和“向量 ↔ bytes”，不直接操作 SQLite。
"""
from functools import lru_cache

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EXPECTED_DIMENSION = 512
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def embedding_model() -> HuggingFaceEmbeddings:
    """延迟加载并缓存模型，避免每个 HTTP 请求都重新读取权重。"""
    # 秋招知识点：lazy loading 缩短服务启动时间；进程内缓存降低重复加载成本。
    return HuggingFaceEmbeddings(
        model=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量生成文档向量；批处理通常比逐条调用模型更高效。"""
    if not texts:
        return []
    vectors = embedding_model().embed_documents(texts)
    if any(len(vector) != EXPECTED_DIMENSION for vector in vectors):
        raise ValueError("Embedding 模型返回了非预期维度。")
    return vectors


def embed_query(text: str) -> list[float]:
    """生成查询向量；BGE v1.5 的查询需要添加检索指令。"""
    return embedding_model().embed_query(QUERY_INSTRUCTION + text)


def vector_to_blob(vector: list[float]) -> bytes:
    """把 Python 浮点列表压缩为 SQLite BLOB 使用的 float32 字节。"""
    # 秋招知识点：float32 每维 4 字节；512 维只需 2048 字节，比 JSON 更紧凑。
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or len(array) != EXPECTED_DIMENSION:
        raise ValueError(f"向量必须是一维且长度为 {EXPECTED_DIMENSION}。")
    return array.tobytes()


def blob_to_vector(blob: bytes, dimension: int) -> np.ndarray:
    """按 float32 规则从 BLOB 还原向量，并校验维度。"""
    array = np.frombuffer(blob, dtype=np.float32)
    if len(array) != dimension:
        raise ValueError("BLOB 字节长度与记录的向量维度不一致。")
    return array

# 计算相似度
def cosine_similarity(left, right) -> float:
    """计算两个向量夹角的余弦值，越接近 1 通常表示语义越相近。"""
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ValueError("参与比较的向量维度不一致。")
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0:
        return 0.0
    # 秋招知识点：余弦相似度只比较方向，不受向量长度影响。
    return float(np.dot(left_array, right_array) / denominator)
