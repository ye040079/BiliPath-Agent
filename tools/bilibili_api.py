"""
B站API底层模块（无兜底版）
所有方法失败时抛出异常，不返回假数据
"""
import re
import time
import hashlib
import urllib.parse
import requests
from typing import List, Dict, Optional
from loguru import logger

import config


class BilibiliAPIError(Exception):
    """B站API调用失败"""
    pass


def _request(url: str, params: dict = None, timeout: int = 10) -> dict:
    """统一请求封装，失败抛异常"""
    try:
        resp = requests.get(url, params=params, headers=config.BILIBILI_HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise BilibiliAPIError(f"B站API返回错误 code={data.get('code')}: {data.get('message')}")
        return data
    except requests.exceptions.Timeout:
        raise BilibiliAPIError(f"请求超时: {url}")
    except requests.exceptions.ConnectionError:
        raise BilibiliAPIError(f"无法连接B站API: {url}")
    except BilibiliAPIError:
        raise
    except Exception as e:
        raise BilibiliAPIError(f"请求失败: {str(e)}")


# ============ wbi 签名（B站搜索接口反爬） ============
# mixin key 重排表：B站前端固定表，把 img_key+sub_key 混淆成 mixin_key
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]

_wbi_cache: Dict[str, object] = {"mixin_key": "", "ts": 0.0}


def _get_mixin_key(orig: str) -> str:
    """按固定重排表混淆得到 mixin_key（取前 32 位）"""
    return "".join(orig[i] for i in _MIXIN_KEY_ENC_TAB)[:32]


def _get_wbi_keys() -> str:
    """从 nav 接口获取 img_key/sub_key 并计算 mixin_key（缓存 1 小时）"""
    cached_key = _wbi_cache["mixin_key"]
    if cached_key and (time.time() - float(_wbi_cache["ts"])) < 3600:
        return str(cached_key)
    try:
        resp = requests.get(config.BILIBILI_NAV_URL, headers=config.BILIBILI_HEADERS, timeout=10)
        wbi_img = resp.json().get("data", {}).get("wbi_img", {})
        img_key = wbi_img.get("img_url", "").rsplit("/", 1)[-1].split(".")[0]
        sub_key = wbi_img.get("sub_url", "").rsplit("/", 1)[-1].split(".")[0]
        if img_key and sub_key:
            mixin_key = _get_mixin_key(img_key + sub_key)
            _wbi_cache["mixin_key"] = mixin_key
            _wbi_cache["ts"] = time.time()
            return mixin_key
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[B站API] 获取 wbi 密钥失败: {e}")
    return str(cached_key)


def _wbi_sign(params: dict) -> dict:
    """为请求参数追加 wts 与 w_rid（wbi 签名），失败时返回原参数"""
    mixin_key = _get_wbi_keys()
    if not mixin_key:
        return params
    signed = dict(params)
    signed["wts"] = int(time.time())
    signed = dict(sorted(signed.items()))
    # 过滤值中不允许的字符，再 URL 编码
    cleaned = {k: "".join(ch for ch in str(v) if ch not in "!'()*") for k, v in signed.items()}
    query = urllib.parse.urlencode(cleaned)
    signed["w_rid"] = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    return signed


def search_videos(keyword: str, page_size: int = 30, page: int = 1) -> List[Dict]:
    """
    搜索B站视频（不过滤，返回原始结果）

    Returns:
        视频列表，每个包含 title/author/bvid/play/duration/description/tag/typename 等
    """
    logger.info(f"[B站API] 搜索: {keyword}")
    params = {
        "search_type": "video",
        "keyword": keyword,
        "order": "totalrank",
        "page": page,
        "page_size": page_size,
    }
    params = _wbi_sign(params)  # wbi 签名，避免 412 风控
    data = _request(config.BILIBILI_SEARCH_URL, params=params)
    results = data.get("data", {}).get("result", [])
    if not results:
        raise BilibiliAPIError(f"搜索无结果: {keyword}")

    videos = []
    for item in results:
        bvid = item.get("bvid", "")
        if not bvid:
            continue  # 跳过无bvid的结果（通常是推广内容）
        videos.append({
            "title": re.sub(r"<[^>]+>", "", item.get("title", "")),
            "author": item.get("author", ""),
            "bvid": bvid,
            "aid": item.get("aid", 0),
            "play": item.get("play", 0),
            "duration": item.get("duration", "0:00"),
            "url": f"https://www.bilibili.com/video/{bvid}",
            "cover": item.get("pic", ""),
            "description": re.sub(r"<[^>]+>", "", item.get("description", "")),
            "tag": item.get("tag", ""),
            "typename": item.get("typename", ""),
            "favorites": item.get("favorites", 0),
            "video_review": item.get("video_review", 0),
        })
    if not videos:
        raise BilibiliAPIError(f"搜索无有效结果: {keyword}")
    logger.info(f"[B站API] 搜索返回 {len(videos)} 条有效结果")
    return videos


def get_video_info(bvid: str) -> Dict:
    """获取视频详细信息（含cid，用于获取字幕）"""
    logger.info(f"[B站API] 获取视频信息: {bvid}")
    data = _request(config.BILIBILI_VIDEO_INFO_URL, params={"bvid": bvid})
    info = data.get("data", {})
    return {
        "bvid": bvid,
        "aid": info.get("aid", 0),
        "cid": info.get("cid", 0),
        "title": info.get("title", ""),
        "desc": info.get("desc", ""),
        "owner": info.get("owner", {}).get("name", ""),
        "duration": info.get("duration", 0),
        "view": info.get("stat", {}).get("view", 0),
        "like": info.get("stat", {}).get("like", 0),
        "coin": info.get("stat", {}).get("coin", 0),
        "favorite": info.get("stat", {}).get("favorite", 0),
        "share": info.get("stat", {}).get("share", 0),
        "reply": info.get("stat", {}).get("reply", 0),
        "tname": info.get("tname", ""),
        "pubdate": info.get("pubdate", 0),
    }


def get_video_subtitles(bvid: str) -> Optional[str]:
    """
    获取视频字幕文本
    没有字幕时返回 None（不报错，由调用方决定怎么处理）
    """
    logger.info(f"[B站API] 获取字幕: {bvid}")
    try:
        info = get_video_info(bvid)
        cid = info["cid"]
        aid = info["aid"]
        if not cid:
            logger.info(f"[B站API] 视频 {bvid} 无cid，无法获取字幕")
            return None

        # 获取字幕列表
        player_url = "https://api.bilibili.com/x/player/v2"
        data = _request(player_url, params={"cid": cid, "aid": aid})
        subtitles = data.get("data", {}).get("subtitle", {}).get("subtitles", [])

        if not subtitles:
            logger.info(f"[B站API] 视频 {bvid} 无字幕")
            return None

        # 取第一个字幕（优先中文）
        sub_url = subtitles[0].get("subtitle_url", "")
        if sub_url.startswith("//"):
            sub_url = "https:" + sub_url

        resp = requests.get(sub_url, timeout=10)
        resp.raise_for_status()
        sub_data = resp.json()

        # 提取字幕文本
        lines = []
        for item in sub_data.get("body", []):
            lines.append(item.get("content", ""))
        full_text = "\n".join(lines)
        logger.info(f"[B站API] 视频 {bvid} 字幕获取成功，{len(full_text)} 字符")
        return full_text

    except BilibiliAPIError as e:
        logger.warning(f"[B站API] 获取字幕失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"[B站API] 获取字幕异常: {e}")
        return None


def get_comments(bvid: str, max_count: int = 100) -> List[Dict]:
    """
    获取视频热门评论
    失败时抛异常
    """
    logger.info(f"[B站API] 获取评论: {bvid}")
    info = get_video_info(bvid)
    aid = info["aid"]
    if not aid:
        raise BilibiliAPIError(f"视频 {bvid} 无法获取aid")

    comments = []
    page = 1
    while len(comments) < max_count and page <= 5:
        data = _request(config.BILIBILI_COMMENT_URL, params={
            "type": 1,
            "oid": aid,
            "mode": 3,  # 按热度排序
            "next": page,
            "ps": 20,
        })
        replies = data.get("data", {}).get("replies", [])
        if not replies:
            break

        for r in replies:
            content = r.get("content", {}).get("message", "")
            if len(content.strip()) >= 2:
                comments.append({
                    "content": content,
                    "like": r.get("like", 0),
                    "username": r.get("member", {}).get("uname", ""),
                })
                if len(comments) >= max_count:
                    break
        page += 1

    if not comments:
        raise BilibiliAPIError(f"视频 {bvid} 无评论数据")

    logger.info(f"[B站API] 获取到 {len(comments)} 条评论")
    return comments
