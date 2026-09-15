r"""本地中文 Embedding 冒烟测试。

运行方式：
    .\.venv-local\Scripts\python.exe scripts\embedding_smoke.py

首次运行会下载 BAAI/bge-small-zh-v1.5；不需要任何 API Key。
"""
import math

from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EXPECTED_DIMENSION = 512

# 秋招知识点：旧版 BGE 检索模型采用“非对称检索”。查询加 instruction，文档正文不加。
# 查询和文档虽然使用同一个编码器，但输入角色不同；不要给文档也重复加查询指令。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def dot(left: list[float], right: list[float]) -> float:
    """计算点积；向量归一化后，点积就等于余弦相似度。"""
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: list[float]) -> float:
    """计算 L2 范数；归一化向量的范数应接近 1。"""
    return math.sqrt(dot(vector, vector))


def main() -> None:
    print(f"正在加载本地模型：{MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        # 1.2.2 版本的参数名是 model；锁定 requirements 后接口才可复现。
        model=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        # 秋招知识点：归一化后可以直接用点积计算余弦相似度，也更便于比较分数。
        encode_kwargs={"normalize_embeddings": True},
    )

    query = "时间序列预测应该怎样建立 baseline？"
    related = "做时间序列预测时，应先建立简单可靠的基线模型，再逐步增加复杂特征。"
    unrelated = "西红柿切块后加入鸡蛋翻炒，是一道常见的家常菜。"

    query_vector = embeddings.embed_query(QUERY_INSTRUCTION + query)
    document_vectors = embeddings.embed_documents([related, unrelated])
    related_score = dot(query_vector, document_vectors[0])
    unrelated_score = dot(query_vector, document_vectors[1])

    # 冒烟测试检查“管线是否基本可信”，不是正式检索效果评测。
    assert len(query_vector) == EXPECTED_DIMENSION, "向量维度与模型说明不一致。"
    assert all(len(vector) == EXPECTED_DIMENSION for vector in document_vectors)
    assert abs(norm(query_vector) - 1.0) < 1e-5, "查询向量没有正确归一化。"
    assert related_score > unrelated_score, "相关文本没有排在无关文本之前。"

    print(f"向量维度：{len(query_vector)}")
    print(f"查询向量 L2 范数：{norm(query_vector):.6f}")
    print(f"相关文本相似度：{related_score:.6f}")
    print(f"无关文本相似度：{unrelated_score:.6f}")
    print("冒烟测试通过：模型加载、向量生成、归一化和基本语义排序均正常。")


if __name__ == "__main__":
    # 秋招知识点：入口保护让文件既能作为脚本运行，也能被测试代码安全导入。
    main()
