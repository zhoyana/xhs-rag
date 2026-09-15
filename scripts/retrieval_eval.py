r"""运行真实语义检索评测。

PowerShell：
    .\.venv-local\Scripts\python.exe scripts\retrieval_eval.py

脚本默认只读：不会创建 Document、Chunk 或 Embedding，也不会修改人工标签。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx


# 直接运行 scripts 下的文件时，Python 默认只认识 scripts 目录。
# 把项目根目录加入模块搜索路径后，才能导入 app；这不是修改系统环境变量。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import DEFAULT_DB, NoteStore  # noqa: E402
from app.evaluation import hit_at_k, reciprocal_rank, summarize_metrics  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "tests" / "data" / "retrieval_eval.jsonl"
REPORT_DIR = PROJECT_ROOT / "data" / "eval_reports"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_cases(path: Path) -> list[dict]:
    """读取 JSONL；每一行都是一条可独立审查的人工标注。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank_candidates(
    client: httpx.Client, query: str, deduplicate_notes: bool
) -> tuple[list[dict], dict]:
    """调用真实 /search 接口，复用后端进程里已经加载的模型。"""
    # 秋招知识点：通过 API 做端到端评测，会覆盖序列化、路由、模型和数据库完整链路。
    response = client.post("/search", json={
        "query": query,
        "top_k": 20,
        "deduplicate_notes": deduplicate_notes,
    })
    response.raise_for_status()
    payload = response.json()
    return payload["items"], payload


def evaluate(cases: list[dict], base_url: str, deduplicate_notes: bool = True) -> dict:
    """逐题调用后端检索并计算指标；模型预热不计入平均查询耗时。"""
    results = []
    model = None
    candidate_count = 0

    # 秋招知识点：HTTPX 默认读取 HTTP_PROXY/HTTPS_PROXY。评测只访问本机服务，
    # 因此关闭 trust_env，避免 127.0.0.1 被错误发送给公司代理并得到 502。
    with httpx.Client(base_url=base_url, timeout=120.0, trust_env=False) as client:
        # 秋招知识点：首次请求包含模型冷启动；预热后再统计更稳定的 warm latency。
        rank_candidates(client, "检索模型预热", deduplicate_notes)
        for case in cases:
            started = time.perf_counter()
            ranked, payload = rank_candidates(client, case["query"], deduplicate_notes)
            latency_ms = (time.perf_counter() - started) * 1000
            model = payload["model"]
            candidate_count = payload["total_candidates"]
            ranked_note_ids = [item["note_id"] for item in ranked]
            relevant = set(case["relevant_note_ids"])

            result = {
                "id": case["id"],
                "query": case["query"],
                "difficulty": case["difficulty"],
                "relevant_note_ids": case["relevant_note_ids"],
                "latency_ms": latency_ms,
                "top_results": ranked[:5],
                # Top5 中重复笔记占掉的名额，可直接观察 Chunk 冗余是否改善。
                "duplicate_slots_top5": len(ranked[:5])
                - len({item["note_id"] for item in ranked[:5]}),
            }
            if relevant:
                result.update({
                    "hit_at_1": hit_at_k(ranked_note_ids, relevant, 1),
                    "hit_at_3": hit_at_k(ranked_note_ids, relevant, 3),
                    "hit_at_5": hit_at_k(ranked_note_ids, relevant, 5),
                    "reciprocal_rank": reciprocal_rank(ranked_note_ids, relevant),
                })
            else:
                # 无答案题没有“正确排名”，先记录最高分，后续用验证集选择拒答阈值。
                result["no_answer_max_score"] = ranked[0]["score"] if ranked else None
            results.append(result)

    answerable = [item for item in results if item["relevant_note_ids"]]
    no_answer = [item for item in results if not item["relevant_note_ids"]]
    summary = {
        "model": model,
        "case_count": len(results),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "chunk_candidate_count": candidate_count,
        "deduplicate_notes": deduplicate_notes,
        "average_duplicate_slots_top5": statistics.mean(
            item["duplicate_slots_top5"] for item in results
        ),
        **summarize_metrics(answerable),
        "average_latency_ms": statistics.mean(item["latency_ms"] for item in results),
        "no_answer_max_scores": [item["no_answer_max_score"] for item in no_answer],
    }
    return {"summary": summary, "cases": results}


def print_report(report: dict) -> None:
    """打印适合初学者阅读的简洁报告，不输出 512 维原始向量。"""
    for case in report["cases"]:
        top = case["top_results"][0] if case["top_results"] else None
        top_text = f"{top['title']} ({top['score']:.4f})" if top else "无候选"
        if case["relevant_note_ids"]:
            status = "命中" if case["hit_at_5"] else "未命中"
        else:
            score = case["no_answer_max_score"]
            status = f"无答案最高分={score:.4f}" if score is not None else "无候选"
        print(f"{case['id']} | {status} | Top1={top_text} | {case['latency_ms']:.1f} ms")

    summary = report["summary"]
    print("\n=== 汇总 ===")
    print(f"模型：{summary['model']}")
    print(f"问题：{summary['case_count']}，候选 Chunk：{summary['chunk_candidate_count']}")
    print(f"按笔记去重：{'是' if summary['deduplicate_notes'] else '否'}")
    print(f"Hit@1：{summary['hit_at_1']:.4f}")
    print(f"Hit@3：{summary['hit_at_3']:.4f}")
    print(f"Hit@5：{summary['hit_at_5']:.4f}")
    print(f"MRR：{summary['mrr']:.4f}")
    print(f"平均热查询耗时：{summary['average_latency_ms']:.1f} ms")
    print(f"Top5 平均重复槽位：{summary['average_duplicate_slots_top5']:.2f}")
    print(f"无答案最高分：{summary['no_answer_max_scores']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="评测本地小红书知识库语义检索")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据库路径")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="JSONL 评测集路径")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="正在运行的后端地址")
    parser.add_argument("--save-json", action="store_true", help="把完整结果保存到 data/eval_reports")
    parser.add_argument(
        "--raw-chunks",
        action="store_true",
        help="关闭按笔记去重，用于观察原始 Chunk 排名和做消融实验",
    )
    args = parser.parse_args()

    cases = load_cases(args.eval_file)
    # 这里只读取数据库检查覆盖率；真正检索统一走 HTTP API，避免加载第二份模型。
    candidates = NoteStore(args.db).searchable_embeddings(MODEL_NAME)
    if not candidates:
        print("没有可检索向量。请先对相关文档执行 /documents/{document_id}/embed。")
        return 2

    embedded_note_ids = {item["note_id"] for item in candidates}
    labeled_note_ids = {note_id for case in cases for note_id in case["relevant_note_ids"]}
    missing = sorted(labeled_note_ids - embedded_note_ids)
    if missing:
        print("评测已停止：以下人工相关笔记尚未生成 Embedding：")
        for note_id in missing:
            print(f"- {note_id}")
        print("先完成这些笔记的 index 和 embed，避免把数据缺失误判为检索失败。")
        return 2

    try:
        report = evaluate(cases, args.base_url, deduplicate_notes=not args.raw_chunks)
    except httpx.ConnectError:
        print(f"无法连接后端 {args.base_url}，请先启动 Uvicorn。")
        return 2
    except httpx.HTTPStatusError as exc:
        print(f"检索接口返回 HTTP {exc.response.status_code}：{exc.response.text}")
        return 2
    print_report(report)
    if args.save_json:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        mode = "raw" if args.raw_chunks else "dedup"
        output = REPORT_DIR / f"retrieval-{mode}-{timestamp}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整报告已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
