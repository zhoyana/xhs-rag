"""评估看板的数据层与三个只读端点。

看板本身是前端的事，但这三条保证必须由后端兜住：

1. 报告目录里混进坏文件（写了一半、被手工编辑坏的 JSON）不能让整页打不开 ——
   报告本来就是手工产出的文件，一份坏的连累另外四份是最容易踩的坑。
2. `GET /eval-reports/{name}` 的形状（客户端给名字、服务端按名字读文件）就是目录
   穿越的经典入口，名字必须被限死在报告目录内。
3. 评估（报告文件）与运行痕迹（tool_calls）不能混着算 —— 混起来之后"平均 2.5 秒"
   到底指检索还是模型，没人说得清。两者是不同接口、不同数据源。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import eval_reports
from app.database import NoteStore
from app.main import create_app

NOW = "2026-09-15T00:00:00+00:00"

SINGLE_SUMMARY = {
    "model": "BAAI/bge-small-zh-v1.5", "case_count": 1, "answerable_count": 1,
    "no_answer_count": 0, "chunk_candidate_count": 5, "deduplicate_notes": True,
    "average_latency_ms": 12.0, "average_duplicate_slots_top5": 0.0,
    "hit_at_1": 1.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "mrr": 1.0,
    "no_answer_max_scores": [0.39],
}

MULTIHOP_SUMMARY = {
    "model": "BAAI/bge-small-zh-v1.5", "eval_kind": "multihop",
    "retrieval_mode": "vector", "case_count": 1, "type_counts": {"bridge": 1},
    "coverage_at_5": 0.95, "all_hit_at_5": 0.9, "hop_recall_at_5": 1.0,
    "all_hops_recall_at_5": 1.0, "coverage_at_1": 0.4, "missing_hop_vector_scores": [0.44],
}

SINGLE_CASE = {
    "id": "q001", "query": "时间序列预测要怎么做", "difficulty": "easy",
    "relevant_note_ids": ["n1"], "latency_ms": 10.0,
    "top_results": [{"note_id": "n1", "title": "做时序预测", "chunk_index": 0,
                     "dimension": 512, "score": 0.75, "content": "正文"}],
    "hit_at_1": 1.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 1.0,
}


def write_report(directory, name, summary, cases=None, raw_text=None):
    """造一份报告文件；raw_text 用于故意写入坏内容。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if raw_text is not None:
        path.write_text(raw_text, encoding="utf-8")
        return path
    path.write_text(json.dumps({"summary": summary, "cases": cases or []},
                               ensure_ascii=False), encoding="utf-8")
    return path


def make_db(tmp_path):
    db_path = tmp_path / "eval.sqlite3"
    NoteStore(db_path)
    return db_path


# ---------------------------- 报告读取与分派 ----------------------------

def test_classify_dispatches_two_report_kinds():
    """两类报告的指标完全不同，按特征字段分派而不是按文件名猜。

    以后新增一类评测时，这个函数必须显式认识它——否则会被算成 unreadable，
    在界面上以"读不出来"的样子出现，比报错还难查。
    """
    assert eval_reports.classify(MULTIHOP_SUMMARY) == eval_reports.KIND_MULTIHOP
    assert eval_reports.classify(SINGLE_SUMMARY) == eval_reports.KIND_SINGLE
    assert eval_reports.classify({"model": "m"}) == eval_reports.KIND_UNREADABLE


def test_list_reports_survives_broken_json(tmp_path):
    """一份坏报告不影响另外两份：列表里多一行说明，而不是整页 500。"""
    directory = tmp_path / "eval_reports"
    write_report(directory, "good-single.json", SINGLE_SUMMARY, [SINGLE_CASE])
    write_report(directory, "good-multihop.json", MULTIHOP_SUMMARY)
    write_report(directory, "half-written.json", {}, raw_text='{"summary": {"model":')

    items = eval_reports.list_reports(directory)

    assert len(items) == 3
    broken = next(item for item in items if item["file"] == "half-written.json")
    assert broken["kind"] == eval_reports.KIND_UNREADABLE
    assert "error" in broken
    kinds = {item["file"]: item["kind"] for item in items}
    assert kinds["good-single.json"] == eval_reports.KIND_SINGLE
    assert kinds["good-multihop.json"] == eval_reports.KIND_MULTIHOP


def test_list_reports_treats_non_object_top_level_as_unreadable(tmp_path):
    """顶层是数组不是对象时同样只标记这一份，不抛异常。"""
    directory = tmp_path / "eval_reports"
    write_report(directory, "array.json", {}, raw_text="[1, 2, 3]")

    items = eval_reports.list_reports(directory)

    assert items[0]["kind"] == eval_reports.KIND_UNREADABLE
    assert items[0]["error"] == "报告顶层不是 JSON 对象"


def test_list_reports_missing_directory_is_empty(tmp_path):
    """报告目录还不存在（新库、还没跑过评测）是正常状态，不是错误。"""
    assert eval_reports.list_reports(tmp_path / "never-created") == []


def test_summarize_report_keeps_only_finite_metrics():
    """NaN / Infinity 会让前端 JSON.parse 直接炸，必须挡在服务端。"""
    summary = {**SINGLE_SUMMARY, "hit_at_1": float("nan"), "hit_at_3": float("inf")}

    out = eval_reports.summarize_report("x.json", {"summary": summary}, {})

    assert out["metrics"] == {"hit_at_5": 1.0, "mrr": 1.0}
    assert out["case_count"] == 1


def test_summarize_report_omits_absent_metrics_instead_of_faking_zero():
    """多跳报告没有 hit_at_k，就不能补 0.0 —— 0 会被读成"检索全错"。"""
    out = eval_reports.summarize_report("mh.json", {"summary": MULTIHOP_SUMMARY}, {})

    assert out["kind"] == eval_reports.KIND_MULTIHOP
    assert "hit_at_1" not in out["metrics"]
    assert out["metrics"]["coverage_at_5"] == 0.95
    assert out["type_counts"] == {"bridge": 1}


# ---------------------------- 路径越界 ----------------------------

def test_resolve_report_rejects_anything_outside_the_directory(tmp_path):
    """名字必须被限死在报告目录内：带路径、换后缀、空名字一律拒绝。"""
    directory = tmp_path / "eval_reports"
    write_report(directory, "ok.json", SINGLE_SUMMARY)
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")

    for bad in ("../outside.json", "sub/ok.json", "C:/Windows/win.ini", "ok.txt", ""):
        with pytest.raises(ValueError):
            eval_reports.resolve_report(directory, bad)

    with pytest.raises(FileNotFoundError):
        eval_reports.resolve_report(directory, "missing.json")

    assert eval_reports.resolve_report(directory, "ok.json").name == "ok.json"


def test_detail_endpoint_maps_errors_to_status_codes(tmp_path):
    """越界/缺失的返回码要能区分：不存在的报告是 404，名字不合法是 400。"""
    db_path = make_db(tmp_path)
    write_report(eval_reports.report_dir(db_path), "ok.json", SINGLE_SUMMARY, [SINGLE_CASE])

    with TestClient(create_app(db_path=db_path)) as client:
        assert client.get("/eval-reports/missing.json").status_code == 404
        assert client.get("/eval-reports/ok.txt").status_code == 400
        # 带斜杠的探测在路由层就匹配不到，压根到不了校验函数——所以校验函数
        # 的越界分支必须靠上面的单元用例覆盖，不能只指望 HTTP 用例。
        assert client.get("/eval-reports/../notes.sqlite3").status_code == 404
        assert client.get("/eval-reports/ok.json").status_code == 200


# ---------------------------- 端点 ----------------------------

def test_list_endpoint_reports_directory_and_summaries(tmp_path):
    """列表要带上目录路径：看板显示"报告从哪读的"，才对得上用户心里的那个目录。"""
    db_path = make_db(tmp_path)
    write_report(eval_reports.report_dir(db_path), "single.json", SINGLE_SUMMARY, [SINGLE_CASE])

    with TestClient(create_app(db_path=db_path)) as client:
        body = client.get("/eval-reports").json()

    assert body["total"] == 1
    assert body["directory"].endswith("eval_reports")
    assert body["items"][0]["file"] == "single.json"
    assert body["items"][0]["deduplicate_notes"] is True


def test_detail_endpoint_attaches_reason_from_case_file(tmp_path):
    """逐题明细要带上人工标注：没有"这题在考察什么"，未命中就只剩一堆 id。"""
    db_path = make_db(tmp_path)
    write_report(eval_reports.report_dir(db_path), "single.json", SINGLE_SUMMARY, [SINGLE_CASE])
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    (case_dir / "retrieval_eval.jsonl").write_text(json.dumps({
        "id": "q001", "query": "时间序列预测要怎么做", "relevant_note_ids": ["n1"],
        "difficulty": "easy", "reason": "考察时序方法论的召回",
    }, ensure_ascii=False), encoding="utf-8")

    with TestClient(create_app(db_path=db_path, case_dir=case_dir)) as client:
        detail = client.get("/eval-reports/single.json").json()

    assert detail["summary"]["file"] == "single.json"
    assert detail["cases"][0]["reason"] == "考察时序方法论的召回"


def test_detail_endpoint_works_without_case_files(tmp_path):
    """评测集缺失（题目版本对不上、文件没带）只是少一行标注，不是错误。"""
    db_path = make_db(tmp_path)
    write_report(eval_reports.report_dir(db_path), "single.json", SINGLE_SUMMARY, [SINGLE_CASE])

    with TestClient(create_app(db_path=db_path, case_dir=tmp_path / "no-cases")) as client:
        response = client.get("/eval-reports/single.json")

    assert response.status_code == 200
    assert "reason" not in response.json()["cases"][0]


def test_trace_stats_counts_only_turns_that_called_tools(tmp_path):
    """没有工具调用的回答不算一轮"执行路径"——否则平均步数会被闲聊稀释。"""
    db_path = make_db(tmp_path)
    store = NoteStore(db_path)
    store.create_thread("t1", "秋招准备", NOW)
    store.add_chat_message("t1", "user", "帮我找时序笔记", {}, NOW)
    store.add_chat_message("t1", "assistant", "直接回答，没调工具", {}, NOW)
    store.add_chat_message("t1", "assistant", "有工具调用的回答", {
        "tool_calls": [{"name": "search_notes", "args": {"query": "时序"}, "ok": True,
                        "duration_ms": 12.0, "result_count": 3, "top_score": 0.7,
                        "error_code": None}],
        "retrieved_note_ids": ["n1"], "cited_note_ids": ["n1"],
    }, NOW)

    with TestClient(create_app(db_path=db_path)) as client:
        stats = client.get("/traces/stats").json()

    assert stats["turn_count"] == 1
    assert stats["call_count"] == 1
    assert stats["turn_limit"] > 0
    assert stats["tools"] == [{"name": "search_notes", "calls": 1, "failed": 0,
                               "avg_ms": 12.0, "p95_ms": 12.0, "avg_results": 3.0}]
    assert stats["citation"]["rate"] == 1.0
    # 提问一起带出来：看板上"这轮在问什么"比一串 thread_id 有用。
    assert stats["slowest"][0]["label"] == "帮我找时序笔记"


def test_trace_stats_on_empty_database(tmp_path):
    """空库要给出全 None 而不是 0，0 会被读成"快得没耗时"。"""
    db_path = make_db(tmp_path)

    with TestClient(create_app(db_path=db_path)) as client:
        stats = client.get("/traces/stats").json()

    assert stats["turn_count"] == 0
    assert stats["duration"]["avg_ms"] is None
    assert stats["steps_per_turn"]["max"] is None
    assert stats["citation"]["rate"] is None
    assert stats["tools"] == []


# ---------------------------- 纯函数：逐题裁剪与聚合 ----------------------------

def test_simplify_case_truncates_tops_and_drops_vectors():
    """逐题明细不能把 512 维向量和全文带出去：一页 10 题就是几 MB。"""
    case = {**SINGLE_CASE, "top_results": [
        {"note_id": f"n{i}", "title": "t", "chunk_index": i, "dimension": 512,
         "score": 0.5, "content": "字" * 500} for i in range(8)
    ]}

    out = eval_reports.simplify_case(case)

    assert len(out["top_results"]) == 5
    assert "dimension" not in out["top_results"][0]
    assert len(out["top_results"][0]["content"]) == 240


def test_simplify_case_expands_multihop_hops():
    """多跳题要看得出每一跳查了什么、Top1 是谁，否则分不清"哪一跳断了"。"""
    case = {
        "id": "mh001", "type": "comparison", "query": "两份食谱各放多少洋葱",
        "relevant_note_ids": ["n1", "n2"],
        "hop_queries": ["消水肿沙拉 洋葱 放多少克", "西红柿炒裤带面 洋葱 用量"],
        "hop_note_ids": [["n1"], ["n2"]],
        "hop_latency_ms": [12.4, 13.6],
        "hop_top_results": [[{"note_id": "n1", "title": "消水肿沙拉", "score": 0.57}],
                            [{"note_id": "n2", "title": "裤带面", "score": 0.62}]],
        "top_results": [],
    }

    out = eval_reports.simplify_case(case)

    assert [hop["query"] for hop in out["hops"]] == case["hop_queries"]
    assert out["hops"][0]["top_title"] == "消水肿沙拉"
    assert out["hops"][0]["top_score"] == 0.57
    assert out["hops"][1]["latency_ms"] == 13.6


def test_percentile_handles_empty_and_single_samples():
    assert eval_reports.percentile([], 0.95) is None
    assert eval_reports.percentile([7.0], 0.95) == 7.0
    assert eval_reports.percentile([10.0, 30.0], 0.95) == pytest.approx(29.0)


def test_summarize_traces_aggregates_calls_failures_and_citation():
    """聚合口径钉在这里：调用次数、失败数、分位耗时、引用率（刻意不按笔记去重）。

    去重会让"同一篇笔记在两轮里各召回一次"压成一次，引用率虚高——比率本身是用来
    发现"检索到了却用不上"的，虚高会让它失去意义。
    """
    turns = [
        {"thread_id": "t1", "created_at": "2026-09-17T07:00:00+00:00", "label": "第一轮",
         "tool_calls": [{"name": "search_notes", "ok": True, "duration_ms": 10.0,
                         "result_count": 4, "top_score": 0.5, "error_code": None}],
         "retrieved_note_ids": ["n1", "n2"], "cited_note_ids": ["n1"]},
        {"thread_id": "t1", "created_at": "2026-09-17T08:00:00+00:00", "label": "第二轮",
         "tool_calls": [{"name": "search_notes", "ok": False, "duration_ms": 30.0,
                         "result_count": None, "top_score": None,
                         "error_code": "collector_unavailable"}],
         "retrieved_note_ids": ["n1"], "cited_note_ids": []},
    ]

    out = eval_reports.summarize_traces(turns)

    assert out["turn_count"] == 2
    assert out["call_count"] == 2
    assert out["failed_count"] == 1
    assert out["citation"] == {"retrieved": 3, "cited": 1, "rate": pytest.approx(1 / 3),
                               "retrieving_turns": 2, "citing_turns": 1}
    assert out["tools"] == [{"name": "search_notes", "calls": 2, "failed": 1,
                             "avg_ms": 20.0, "p95_ms": pytest.approx(29.0),
                             "avg_results": 4.0}]
    assert out["failures"][0]["error_code"] == "collector_unavailable"
    assert out["slowest"][0]["tool"] == "search_notes"
    assert out["slowest"][0]["label"] == "第二轮"


def test_summarize_traces_ignores_turns_without_calls():
    """防御性：即使上游漏传了空轮次，也不能把它算进平均步数。"""
    out = eval_reports.summarize_traces([{"tool_calls": []}, {"tool_calls": None}])

    assert out["turn_count"] == 0
    assert out["tools"] == []


def test_load_case_notes_merges_files_and_skips_broken_lines(tmp_path):
    """标注文件是手工维护的 jsonl：坏行跳过，好行照收，别为一行坏数据丢整份标注。"""
    (tmp_path / "a.jsonl").write_text(
        '{"id": "q001", "reason": "第一条"}\n{bad json\n{"id": "q002", "reason": "第二条"}\n',
        encoding="utf-8")
    (tmp_path / "multihop_eval.jsonl").write_text(
        '{"id": "mh001", "reason": "多跳第一条"}\n', encoding="utf-8")

    notes = eval_reports.load_case_notes([tmp_path / "a.jsonl",
                                          tmp_path / "multihop_eval.jsonl",
                                          tmp_path / "missing.jsonl"])

    assert notes["q001"]["reason"] == "第一条"
    assert notes["q002"]["reason"] == "第二条"
    assert notes["mh001"]["reason"] == "多跳第一条"


def test_attach_reasons_leaves_unmatched_cases_untouched():
    """报告与评测集不同版本时匹配不上，只是没有标注，不能贴错别人的解释。"""
    report = {"cases": [{"id": "q001"}, {"id": "q099"}]}

    eval_reports.attach_reasons(report, {"q001": {"reason": "只有这题有标注"}})

    assert report["cases"][0]["reason"] == "只有这题有标注"
    assert "reason" not in report["cases"][1]
