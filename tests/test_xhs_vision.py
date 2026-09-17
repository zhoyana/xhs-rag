"""读图测试：全部离线。stub 掉图片下载与视觉模型调用。

为什么这类测试必须离线：真实链路要下 3 张长图并打 3 次多模态请求，
既慢又要密钥；把下载与模型调用做成可注入的，就能把"图片如何变成文字"
这段工程逻辑单独钉死。
"""
import asyncio
import base64
import json

import pytest

from app import xhs_mcp
from app.xhs_vision import (
    MAX_IMAGES,
    _pick_images,
    allowed_image_url,
    transcribe_images,
)

IMG_A = "https://sns-webpic-qc.xhscdn.com/202609170002/aaa!nd_dft_wlteh_webp_3"
IMG_B = "https://sns-webpic-qc.xhscdn.com/202609170002/bbb!nd_dft_wlteh_webp_3"
IMG_C = "https://sns-webpic.xhscdn.com/202609170002/ccc!nd_dft_wlteh_webp_3"

PAGE_1 = "一面第 6 题：短期记忆/长期记忆怎么划分"
PAGE_2 = "二面第 1 题：多 Agent 协作模式"


def run(coro):
    return asyncio.run(coro)


def fetch_returning(mapping, calls=None):
    async def fetch(url):
        if calls is not None:
            calls.append(url)
        value = mapping.get(url)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("unexpected url")
        return value
    return fetch


def model_returning(*payloads, captured=None):
    """按顺序返回预设结果；用完后一直返回最后一个。"""
    queue = list(payloads)

    async def call(content):
        if captured is not None:
            captured.append(content)
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(value, Exception):
            raise value
        return value
    return call


# ---------------- 地址白名单 ----------------

def test_accepts_xhs_cdn_hosts():
    assert allowed_image_url(IMG_A) is True
    assert allowed_image_url(IMG_C) is True


@pytest.mark.parametrize("url", [
    "https://evil.example.com/a.jpg",
    "http://sns-webpic-qc.xhscdn.com.evil.com/a.jpg",
    "file:///c:/windows/win.ini",
    "https://sns-webpic-qc.xhscdn.com:8080/a.jpg",
    "https://user:pass@sns-webpic-qc.xhscdn.com/a.jpg",
    "",
    None,
])
def test_rejects_non_cdn_targets(url):
    """白名单必须写死：否则这个工具会变成任意 URL 下载器（SSRF）。"""
    assert allowed_image_url(url) is False


def test_pick_images_keeps_note_order_and_drops_others():
    urls, dropped = _pick_images([IMG_A, "https://evil.example.com/x.jpg", IMG_B], MAX_IMAGES)
    assert urls == [IMG_A, IMG_B]
    assert dropped == 1


def test_pick_images_caps_count():
    many = [f"https://sns-webpic-qc.xhscdn.com/p/{i}.webp" for i in range(MAX_IMAGES + 3)]
    urls, dropped = _pick_images(many, MAX_IMAGES)
    assert len(urls) == MAX_IMAGES
    assert dropped == 3


# ---------------- 转写主流程 ----------------

def test_transcribes_each_image_in_one_request():
    """每张图单独一次请求：单张输出才不会被总长度上限截断。"""
    captured = []
    result = run(transcribe_images(
        [IMG_A, IMG_B],
        title="淘宝AI Agent开发面经",
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp"),
                                     IMG_B: (b"BBB", "image/webp")}),
        call_model=model_returning(PAGE_1, PAGE_2, captured=captured),
    ))

    assert result["status"] == "ok"
    assert result["read"] == 2
    # 每段前面由本地补上序号，模型不需要自己编号。
    assert "【第 1 张图】" in result["text"] and "【第 2 张图】" in result["text"]
    assert result["text"].index(PAGE_1) < result["text"].index(PAGE_2)

    assert len(captured) == 2
    for content in captured:
        blocks = [block for block in content if block["type"] == "image_url"]
        assert len(blocks) == 1
        assert blocks[0]["image_url"]["url"].startswith("data:image/webp;base64,")
        # 长图小字需要 original 档位才读得清。
        assert blocks[0]["image_url"]["detail"] == "original"
    assert "第 1 张图片" in captured[0][0]["text"]
    assert "第 2 张图片" in captured[1][0]["text"]
    assert base64.b64decode(
        captured[0][1]["image_url"]["url"].split(",", 1)[1]) == b"AAA"


def test_download_failure_does_not_block_other_images():
    result = run(transcribe_images(
        [IMG_A, IMG_B],
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp"),
                                     IMG_B: RuntimeError("timeout")}),
        call_model=model_returning(PAGE_1),
    ))
    assert result["status"] == "partial"
    assert result["read"] == 1
    assert PAGE_1 in result["text"]
    assert result["failed"] == [2]


def test_one_bad_image_does_not_ruin_the_rest():
    """第二张图模型报错时，第一张的内容必须保住。"""
    result = run(transcribe_images(
        [IMG_A, IMG_B],
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp"),
                                     IMG_B: (b"BBB", "image/webp")}),
        call_model=model_returning(PAGE_1, RuntimeError("502")),
    ))
    assert result["status"] == "partial"
    assert PAGE_1 in result["text"]
    assert result["failed"] == [2]


def test_all_downloads_failed_reports_failure():
    captured = []
    result = run(transcribe_images(
        [IMG_A, IMG_B],
        fetch_image=fetch_returning({IMG_A: RuntimeError("boom"), IMG_B: RuntimeError("boom")}),
        call_model=model_returning(PAGE_1, captured=captured),
    ))
    assert result["status"] == "failed"
    assert result["text"] == ""
    # 图都下不来时不该浪费模型调用。
    assert captured == []


def test_model_failure_is_swallowed_not_raised():
    """读图失败只是少一段文字，绝不能把采集本身炸掉。"""
    result = run(transcribe_images(
        [IMG_A],
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp")}),
        call_model=model_returning(RuntimeError("502")),
    ))
    assert result["status"] == "failed"
    assert result["text"] == ""
    assert result["error"]


def test_empty_response_is_retried_once():
    """真机实测会遇到空 content；重试一次能恢复，不能直接把内容丢掉。"""
    captured = []
    result = run(transcribe_images(
        [IMG_A],
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp")}),
        call_model=model_returning("", PAGE_1, captured=captured),
    ))
    assert result["status"] == "ok"
    assert PAGE_1 in result["text"]
    assert len(captured) == 2


def test_two_empty_responses_end_as_failure():
    captured = []
    result = run(transcribe_images(
        [IMG_A],
        fetch_image=fetch_returning({IMG_A: (b"AAA", "image/webp")}),
        call_model=model_returning("", "", captured=captured),
    ))
    assert result["status"] == "failed"
    assert len(captured) == 2


def test_no_usable_images_skips_model_call():
    captured = []
    result = run(transcribe_images(
        ["https://evil.example.com/x.jpg"],
        fetch_image=fetch_returning({}),
        call_model=model_returning(PAGE_1, captured=captured),
    ))
    assert result["status"] == "empty"
    assert captured == []


# ---------------- 与采集工具的接线 ----------------

def collect_payload(image_list):
    return {"feed_id": "6aa3c6ff0000000025036dd4", "data": {"note": {
        "noteId": "6aa3c6ff0000000025036dd4",
        "title": "淘宝AI Agent开发面经",
        "desc": "#面经[话题]# #ai[话题]#",
        "type": "normal",
        "imageList": [{"urlDefault": url} for url in image_list],
    }}}


class FakeDetailTool:
    def __init__(self, payload):
        self.payload = json.dumps(payload)

    async def ainvoke(self, args):
        return self.payload


def build_tool(transcribe):
    return xhs_mcp.build_link_collect_tool(FakeDetailTool(collect_payload([IMG_A, IMG_B])),
                                           transcribe=transcribe)


LINK = ("https://www.xiaohongshu.com/explore/6aa3c6ff0000000025036dd4"
        "?xsec_token=ABXOT8pz&xsec_source=pc_user")


def test_collect_attaches_image_text_and_fixes_warnings():
    """读图成功后，"内容在图片里"这条警告必须撤掉，否则会误导模型拒绝回答。"""
    async def fake_transcribe(images, title="", **_kwargs):
        assert images == [IMG_A, IMG_B]
        return {"status": "ok", "text": f"【第 1 张图】\n{PAGE_1}", "model": "fake",
                "read": 2, "dropped": 0, "error": None}

    result = run(build_tool(fake_transcribe).ainvoke({"link": LINK}))
    data = result["data"]
    assert result["ok"] is True
    assert PAGE_1 in data["image_text"]
    assert data["image_count"] == 2
    assert not any("主要内容在图片里" in w for w in data["warnings"])
    assert not any("需及时下载" in w for w in data["warnings"])
    assert any("已用视觉模型识别" in w for w in data["warnings"])


def test_collect_can_skip_image_reading():
    calls = []

    async def fake_transcribe(images, title="", **_kwargs):
        calls.append(images)
        return {"status": "ok", "text": "x", "model": "fake",
                "read": 0, "dropped": 0, "error": None}

    tool = build_tool(fake_transcribe)
    result = run(tool.ainvoke({"link": LINK, "read_images": False}))
    assert result["ok"] is True
    assert calls == []
    assert "image_text" not in result["data"]


def test_collect_survives_transcribe_crash():
    async def broken_transcribe(images, title="", **_kwargs):
        raise RuntimeError("vision down")

    result = run(build_tool(broken_transcribe).ainvoke({"link": LINK}))
    assert result["ok"] is True
    assert "image_text" not in result["data"]
    assert any("图片识别" in w for w in result["data"]["warnings"])
