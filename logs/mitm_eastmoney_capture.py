from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mitmproxy import http


OUTPUT_PATH = Path(__file__).resolve().parent / "eastmoney_proxy_capture.jsonl"
MAX_BODY_BYTES = 32 * 1024
HOST_KEYWORDS = (
    "eastmoney.com",
    "dfcfw.com",
    "10jqka",
    "quote.eastmoney",
    "myfavor.eastmoney",
)
BODY_PATH_KEYWORDS = (
    "/v4/",
    "/zixuan/api/",
    "/newapi/portfolio",
    "/mycombin",
    "myfavor",
)


def _matches(host: str) -> bool:
    host = host.lower()
    return any(keyword in host for keyword in HOST_KEYWORDS)


def _append(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _should_capture_body(flow: http.HTTPFlow) -> bool:
    path = flow.request.path.lower()
    return flow.request.method.upper() != "GET" or any(
        keyword in path for keyword in BODY_PATH_KEYWORDS
    )


def _decode_content(content: bytes | None, headers) -> str | None:
    if not content:
        return None

    size = len(content)
    if size > MAX_BODY_BYTES:
        return f"<omitted:{size} bytes>"

    content_type = (headers.get("content-type", "") or "").lower()
    is_text = any(
        marker in content_type
        for marker in (
            "json",
            "javascript",
            "text/",
            "xml",
            "x-www-form-urlencoded",
        )
    )
    if not is_text:
        return f"<binary:{size} bytes>"

    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="replace")


def request(flow: http.HTTPFlow) -> None:
    host = flow.request.pretty_host or flow.request.host or ""
    if not _matches(host):
        return

    payload = {
        "event": "request",
        "time": datetime.now(timezone.utc).isoformat(),
        "method": flow.request.method,
        "scheme": flow.request.scheme,
        "host": host,
        "port": flow.request.port,
        "path": flow.request.path,
        "url": flow.request.pretty_url,
        "headers": dict(flow.request.headers),
    }
    if _should_capture_body(flow):
        body = _decode_content(flow.request.raw_content, flow.request.headers)
        if body is not None:
            payload["body"] = body

    _append(payload)


def response(flow: http.HTTPFlow) -> None:
    host = flow.request.pretty_host or flow.request.host or ""
    if not _matches(host):
        return

    payload = {
        "event": "response",
        "time": datetime.now(timezone.utc).isoformat(),
        "method": flow.request.method,
        "host": host,
        "path": flow.request.path,
        "status_code": flow.response.status_code,
        "headers": dict(flow.response.headers),
    }
    if _should_capture_body(flow):
        body = _decode_content(flow.response.raw_content, flow.response.headers)
        if body is not None:
            payload["body"] = body

    _append(payload)
