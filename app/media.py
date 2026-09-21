"""图片模块：HTTPX 下载原始字节，Pillow 校验，不修改图片内容。"""
import hashlib
import ipaddress
import re
import socket
import tempfile
import threading
import time
import warnings
from pathlib import Path
from urllib.parse import urlsplit, urljoin

import httpx
from PIL import Image, UnidentifiedImageError

from .extractor import ImportProblem

# 只接受已知图片 CDN，不能把本接口当作任意 URL 下载器。
IMAGE_HOSTS = {"sns-webpic-qc.xhscdn.com", "sns-webpic.xhscdn.com"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
FORMATS = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"),
           "WEBP": (".webp", "image/webp")}


# 采集同一轮里已经拿到手的图片字节，按地址暂存，供随即的落盘复用。
#
# 为什么需要这层：读图（喂视觉模型）和落盘如果各发一次请求，第二次撞上过期签名
# 就是必然——而第一次明明已经把字节拿到手了。界面「批量导入」当初就是死在这里：
# 读图成功、用户几小时后再点下载时地址已过期，只剩 403，一张也存不下来。
STASH_MAX_BYTES = 48 * 1024 * 1024
_stash_lock = threading.Lock()
_stash = {}
_stash_bytes = 0


def stash_image(url, data, mime):
    """暂存一张刚下载的图片字节，供同一次采集的落盘取用。

    只做有界缓存：一批长图笔记可能几十兆，落盘前没人取走的部分要能自己让位。
    """
    global _stash_bytes
    if not isinstance(url, str) or not url or not data or len(data) > MAX_BYTES:
        return
    with _stash_lock:
        if url in _stash:
            return
        _stash[url] = (data, mime)
        _stash_bytes += len(data)
        # 插入顺序即淘汰顺序：超上限时丢最早的，留下的仍是最近采集的那批。
        while _stash_bytes > STASH_MAX_BYTES and len(_stash) > 1:
            oldest = next(iter(_stash))
            dropped, _dropped_mime = _stash.pop(oldest)
            _stash_bytes -= len(dropped)


def take_stashed_image(url):
    """取走暂存的（字节, MIME）；取走即删除，没有暂存时返回 None。"""
    global _stash_bytes
    with _stash_lock:
        item = _stash.pop(url, None) if isinstance(url, str) else None
        if item is not None:
            _stash_bytes -= len(item[0])
    return item


def clear_stashed_images():
    """清空暂存；测试用，避免用例之间互相影响。"""
    global _stash_bytes
    with _stash_lock:
        _stash.clear()
        _stash_bytes = 0


def check_image_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in IMAGE_HOSTS
                or parsed.username or parsed.password or parsed.port not in {None, 80, 443}):
            raise ValueError()
    except ValueError:
        raise ImportProblem("图片地址不在允许的 CDN 范围内。") from None
    # 进一步拒绝 DNS 指向本机或内网。公网部署仍需出站网络隔离以处理 DNS 重绑定。
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        if not ipaddress.ip_address(result[4][0]).is_global:
            raise ImportProblem("图片域名解析到非公网地址，已拒绝。")


def storage_path(root, key):
    # 不信任文件名：解析后必须仍位于 media 根目录里，防止 ../ 越界访问。
    root = Path(root).resolve()
    path = (root / key).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ImportProblem("图片保存路径不合法。")
    return path


def local_file_valid(root, asset):
    if asset["status"] != "success" or not asset.get("storage_key"):
        return False
    try:
        path = storage_path(root, asset["storage_key"])
        if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
            return False
        # 哈希是文件指纹；发现文件丢失或被修改时，不直接跳过下载。
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest() == asset["sha256"]
    except (OSError, ImportProblem):
        return False


def _place_verified(asset, root, temporary, size, digest):
    """校验临时文件确为支持的图片，按内容指纹命名后原子替换到最终位置。

    下载路径与复用路径都要走这里：图片格式校验、命名规则、原子落盘只该有一份，
    否则"本地下载的图"和"复用字节落盘的图"会各有一套可信度。
    """
    # 不凭响应头/网址后缀相信文件类型。verify 检查结构，load 实际解码。
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(temporary) as picture:
            fmt, width, height = picture.format, picture.width, picture.height
            if fmt not in FORMATS or width * height > MAX_PIXELS or getattr(picture, "n_frames", 1) != 1:
                raise ImportProblem("仅支持像素数不超过2500万的静态 JPEG、PNG、WebP 图片。")
            picture.verify()
        with Image.open(temporary) as picture:
            picture.load()
    suffix, mime = FORMATS[fmt]
    # 加入文件指纹，避免不同内容在同一位置相互覆盖。
    key = f"{asset['note_id']}/{asset['position']:03d}-{digest[:16]}{suffix}"
    temporary.replace(storage_path(root, key))
    return {"storage_key": key, "mime_type": mime, "size_bytes": size,
            "width": width, "height": height, "sha256": digest}


def save_image_bytes(asset, root, data, mime=None):
    """把已经在内存里的图片字节落盘，不再发第二次请求。

    用在"采集同轮已经为读图下载过这张图"的场景：图床地址带签名且很快过期，
    第二次请求正是 403 的来源，而字节本来就在手上。mime 只作为兼容参数接收，
    真实类型一律以解码结果为准。
    """
    if not re.fullmatch(r"[a-zA-Z0-9]+", asset["note_id"]):
        raise ImportProblem("笔记编号不合法。")
    if len(data) > MAX_BYTES:
        raise ImportProblem("单张图片超过 20 MB。")
    folder = storage_path(root, asset["note_id"])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # 与下载路径一样先写同目录临时文件，校验通过后再原子替换。
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        return _place_verified(asset, root, temporary, len(data),
                               hashlib.sha256(data).hexdigest())
    except ImportProblem:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, SyntaxError, ValueError):
        raise ImportProblem("文件不是支持的有效图片，或图片像素数超限。") from None
    except OSError:
        raise ImportProblem("图片写入失败，请检查磁盘空间与权限。") from None
    finally:
        # 只删除本次创建的临时文件，不删除已保存的图片或原笔记。
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def download_image(asset, root):
    """下载单张图片。成功返回元数据；失败清理本次临时文件。"""
    if not re.fullmatch(r"[a-zA-Z0-9]+", asset["note_id"]):
        raise ImportProblem("笔记编号不合法。")
    folder = storage_path(root, asset["note_id"])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    started = time.monotonic()
    try:
        # 临时文件与最终文件在同一目录，验证后原子替换，避免暴露未完成文件。
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            url = asset["source_url"]
            with httpx.Client(timeout=15, follow_redirects=False,
                              headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.xiaohongshu.com/"}) as client:
                for _ in range(5):
                    check_image_url(url)
                    with client.stream("GET", url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            target = response.headers.get("location")
                            if not target:
                                raise ImportProblem("图片跳转缺少地址。")
                            url = urljoin(url, target)
                            continue  # 新目标也必须经过域名及地址校验。
                        if response.status_code != 200:
                            raise ImportProblem(f"图片下载返回 HTTP {response.status_code}，链接可能失效或受限。")
                        size = 0
                        digest = hashlib.sha256()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            size += len(chunk)
                            if size > MAX_BYTES:
                                raise ImportProblem("单张图片超过 20 MB。")
                            if time.monotonic() - started > 60:
                                raise ImportProblem("图片下载耗时过长。")
                            output.write(chunk)
                            digest.update(chunk)
                        break
                else:
                    raise ImportProblem("图片跳转次数过多。")
        return _place_verified(asset, root, temporary, size, digest.hexdigest())
    except ImportProblem:
        raise
    except httpx.TimeoutException:
        raise ImportProblem("图片下载超时，可重试。") from None
    except httpx.HTTPError:
        raise ImportProblem("无法连接图片站点，可重试；检查后端联网权限。") from None
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, SyntaxError, ValueError):
        raise ImportProblem("文件不是支持的有效图片，或图片像素数超限。") from None
    except OSError:
        raise ImportProblem("图片解码、网络解析或磁盘操作失败，请检查后重试。") from None
    finally:
        # 只删除本次创建的临时文件，不删除已保存的图片或原笔记。
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def download_note_assets(store, note, root):
    store.ensure_assets(note)
    downloaded = skipped = reused = 0
    for asset in store.assets(note["note_id"]):
        # 刷新后来源页面已不存在的旧位置只保留审计记录，不再向过期 URL 发请求。
        if asset["status"] == "obsolete":
            continue
        if local_file_valid(root, asset):
            skipped += 1
            continue
        store.asset_status(asset["id"], "downloading")
        try:
            # 采集同一轮为读图下载过的字节优先复用：省一次请求，也绕开
            # "用户手点下载时签名地址早已过期"这个必然发生的问题。
            staged = take_stashed_image(asset["source_url"])
            if staged is not None:
                info = save_image_bytes(asset, root, *staged)
                reused += 1
            else:
                info = download_image(asset, root)
            store.complete_asset(asset["id"], info)
            downloaded += 1
        except ImportProblem as exc:
            # 一张失败不阻断其他图片；再次调用会重试失败项。
            store.asset_status(asset["id"], "failed", str(exc))
    items = store.assets(note["note_id"])
    failed = sum(item["status"] == "failed" for item in items)
    # downloaded 含复用字节的那部分：对用户来说两者都是"这次存下来了"，
    # reused 只是把来源说清楚，前端提示与排查都靠它。
    return {"status": "complete" if not failed else "partial" if downloaded + skipped else "failed",
            "total": len(items), "downloaded": downloaded, "skipped": skipped,
            "reused": reused, "failed": failed}
