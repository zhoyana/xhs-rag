"""检索评测集结构测试：保护人工标注，避免格式错误进入评测流程。"""
import json
from pathlib import Path


EVAL_FILE = Path(__file__).parent / "data" / "retrieval_eval.jsonl"


def load_cases():
    """JSONL 每行是一个独立 JSON 对象，方便追加和逐条审查。"""
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_retrieval_eval_schema():
    cases = load_cases()
    assert 5 <= len(cases) <= 10
    assert len({case["id"] for case in cases}) == len(cases)
    assert any(not case["relevant_note_ids"] for case in cases), "至少保留一条无答案问题。"

    for case in cases:
        assert set(case) == {"id", "query", "relevant_note_ids", "difficulty", "reason"}
        assert case["id"].startswith("q")
        assert case["query"].strip()
        assert isinstance(case["relevant_note_ids"], list)
        assert case["difficulty"] in {"easy", "medium", "hard", "no_answer"}
        assert case["reason"].strip()


def test_eval_has_multiple_query_styles():
    cases = load_cases()
    difficulties = {case["difficulty"] for case in cases}
    assert {"easy", "medium", "hard", "no_answer"} <= difficulties
    assert any(len(case["relevant_note_ids"]) > 1 for case in cases)
