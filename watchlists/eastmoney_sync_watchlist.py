#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作用：把本地筛选出的股票列表同步到东方财富自选组合。
# 方案：优先直接调用 myfavor.eastmoney.com/v4 接口，不再逐条搜索和点击页面。

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import SCREENING_RESULTS_DIR, WATCHLISTS_DIR, ensure_project_dirs

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CANDIDATES = [
    SCREENING_RESULTS_DIR / "weekly_bullish_and_m30_dow_pullback_latest.csv",
    SCREENING_RESULTS_DIR / "weekly_bullish_latest.csv",
    SCREENING_RESULTS_DIR / "m30_dow_pullback_latest.csv",
]
DEFAULT_PROFILE_CANDIDATES = [
    WATCHLISTS_DIR / ".eastmoney_profile",
    PROJECT_DIR / ".eastmoney_profile",
    Path.home() / ".eastmoney_profile_sync",
]
DEFAULT_BID_FILE = WATCHLISTS_DIR / ".eastmoney_bid"
DEFAULT_COOKIE_FILE = WATCHLISTS_DIR / ".eastmoney_cookie"
EASTMONEY_URL = "https://quote.eastmoney.com/zixuan/?from=favor"
MYFAVOR_BASE_URL = "https://myfavor.eastmoney.com/v4/"
MYFAVOR_APPKEY = "e9166c7e9cdfad3aa3fd7d93b757e9b1"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_SECONDS = 30
BROWSER_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
JSONP_PATTERN = re.compile(r"^[^(]+\((.*)\)\s*;?\s*$", re.S)


class EastMoneySyncError(RuntimeError):
    """业务异常。"""


@dataclass(frozen=True)
class StockRef:
    symbol: str
    secid: str
    name: str | None = None


@dataclass
class BrowserSessionSnapshot:
    cookies: list[dict]
    user_agent: str
    logged_in: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把筛选出的股票自动同步到东方财富自选组合。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="要导入的 CSV 文件路径。默认自动按优先级查找项目内结果文件。",
    )
    parser.add_argument(
        "--column",
        default="symbol",
        help="股票代码所在列名，默认使用 symbol。",
    )
    parser.add_argument(
        "--file-symbol-column",
        default="file_symbol",
        help="市场前缀代码列名，默认使用 file_symbol，例如 sh600011、sz000001。",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="股票名称列名，默认使用 name。",
    )
    parser.add_argument(
        "--group-name",
        help="指定组合名。默认使用今天日期 MMDD，例如 0603。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只处理前 N 只股票，便于小范围验证。",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "anonymous", "account"],
        default="auto",
        help="接口模式。auto 会优先根据浏览器态判断，否则默认 anonymous。",
    )
    parser.add_argument(
        "--bid",
        help="直接指定匿名自选使用的 qgqp_b_id。提供后可完全跳过浏览器。",
    )
    parser.add_argument(
        "--bid-file",
        type=Path,
        default=DEFAULT_BID_FILE,
        help="本地保存 qgqp_b_id 的文件。默认 .eastmoney_bid。",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        default=DEFAULT_COOKIE_FILE,
        help="本地保存账号 Cookie 的文件。默认 .eastmoney_cookie。",
    )
    parser.add_argument(
        "--cookie",
        help="直接指定 Cookie 字符串。适合账号模式或你手动从浏览器复制。",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="浏览器持久化目录。仅在需要自动提取本地登录态或 qgqp_b_id 时使用。",
    )
    parser.add_argument(
        "--browser-path",
        type=Path,
        help="浏览器可执行文件路径。默认自动查找 Chrome / Edge。",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="提取本地浏览器态时使用无界面模式。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 CSV 和生成组合名，不真正调用接口。",
    )
    return parser


def resolve_input_path(user_input: Path | None) -> Path:
    if user_input is not None:
        input_path = user_input.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {input_path}")
        return input_path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"没有找到默认 CSV 文件，请用 --input 指定路径。已尝试:\n{tried}")


def resolve_profile_dir(user_input: Path | None) -> Path | None:
    if user_input is not None:
        profile_dir = user_input.expanduser().resolve()
        if not profile_dir.exists():
            raise FileNotFoundError(f"浏览器 profile 目录不存在: {profile_dir}")
        return profile_dir

    for candidate in DEFAULT_PROFILE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_browser_path(user_input: Path | None) -> Path:
    if user_input is not None:
        browser_path = user_input.expanduser().resolve()
        if not browser_path.exists():
            raise FileNotFoundError(f"浏览器路径不存在: {browser_path}")
        return browser_path

    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return candidate

    tried = "\n".join(str(path) for path in BROWSER_CANDIDATES)
    raise FileNotFoundError(
        "没有找到可用浏览器，请用 --browser-path 指定，"
        "或者直接通过 --bid / --cookie 跳过浏览器。已尝试:\n"
        f"{tried}"
    )


def normalize_symbol(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    match = re.search(r"(\d{6})", text)
    if not match:
        return None
    return match.group(1)


def normalize_symbol_from_file_symbol(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text or text == "nan":
        return None

    match = re.search(r"(?:sh|sz|bj)(\d{6})", text)
    if not match:
        return None
    return match.group(1)


def infer_market_from_code(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return "1"
    return "0"


def infer_secid(symbol: str, file_symbol: object | None) -> str:
    if file_symbol is not None:
        text = str(file_symbol).strip().lower()
        match = re.search(r"(sh|sz|bj)(\d{6})", text)
        if match:
            prefix, digits = match.groups()
            if prefix == "sh":
                return f"1.{digits}"
            return f"0.{digits}"

    return f"{infer_market_from_code(symbol)}.{symbol}"


def load_stocks(
    csv_path: Path,
    symbol_column: str,
    file_symbol_column: str,
    name_column: str,
) -> list[StockRef]:
    df = pd.read_csv(csv_path)
    if symbol_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"CSV 中找不到列 {symbol_column!r}，当前列有: {available}")

    seen: set[str] = set()
    stocks: list[StockRef] = []
    for _, row in df.iterrows():
        file_symbol_value = row[file_symbol_column] if file_symbol_column in df.columns else None
        symbol = normalize_symbol_from_file_symbol(file_symbol_value) or normalize_symbol(row[symbol_column])
        if symbol is None or symbol in seen:
            continue

        secid = infer_secid(
            symbol=symbol,
            file_symbol=file_symbol_value,
        )
        name = None
        if name_column in df.columns:
            raw_name = row[name_column]
            if raw_name is not None and str(raw_name).strip() and str(raw_name).lower() != "nan":
                name = str(raw_name).strip()

        seen.add(symbol)
        stocks.append(StockRef(symbol=symbol, secid=secid, name=name))

    if not stocks:
        raise ValueError(f"CSV 文件 {csv_path} 中没有可用股票代码。")

    return stocks


def default_group_name() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d")


def build_unique_group_name(base_name: str, existing_names: list[str]) -> str:
    existing = set(existing_names)
    if base_name not in existing:
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}{suffix:02d}"
        if candidate not in existing:
            return candidate
        suffix += 1


def parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for chunk in cookie_header.split(";"):
        item = chunk.strip()
        if item.lower().startswith("cookie:"):
            item = item.split(":", 1)[1].strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        cookies[name.strip()] = value.strip()
    return cookies


def load_cookie_from_file(cookie_file: Path | None) -> str | None:
    if cookie_file is None:
        return None
    if not cookie_file.exists():
        return None

    content = cookie_file.read_text(encoding="utf-8").strip()
    if not content:
        return None
    if content.lower().startswith("cookie:"):
        content = content.split(":", 1)[1].strip()
    return content or None


def save_cookie_to_file(cookie_header: str, cookie_file: Path | None) -> None:
    if cookie_file is None:
        return
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text(f"{cookie_header.strip()}\n", encoding="utf-8")


def load_bid_from_file(bid_file: Path | None) -> str | None:
    if bid_file is None:
        return None
    if not bid_file.exists():
        return None

    content = bid_file.read_text(encoding="utf-8").strip()
    return content or None


def save_bid_to_file(bid: str, bid_file: Path | None) -> None:
    if bid_file is None:
        return
    bid_file.parent.mkdir(parents=True, exist_ok=True)
    bid_file.write_text(f"{bid}\n", encoding="utf-8")


def make_anonymous_bid() -> str:
    return "".join([str(random.randint(1, 9))] + [str(random.randint(0, 9)) for _ in range(19)])


def extract_session_from_browser(
    profile_dir: Path,
    browser_path: Path,
    headless: bool,
) -> BrowserSessionSnapshot:
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            executable_path=str(browser_path),
            headless=headless,
            locale="zh-CN",
            args=["--start-maximized"],
            viewport={"width": 1440, "height": 960} if headless else None,
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(EASTMONEY_URL, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(3_000)
            body_head = page.locator("body").inner_text(timeout=10_000)[:400]
            cookies = context.cookies()
            user_agent = page.evaluate("navigator.userAgent")
            logged_in = "登录注册" not in body_head and "登录" not in body_head[:50]
            return BrowserSessionSnapshot(
                cookies=cookies,
                user_agent=user_agent,
                logged_in=logged_in,
            )
        finally:
            context.close()


def create_requests_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Referer": EASTMONEY_URL,
            "Accept": "*/*",
        }
    )
    return session


def seed_cookies_from_snapshot(session: requests.Session, snapshot: BrowserSessionSnapshot) -> None:
    for cookie in snapshot.cookies:
        domain = cookie.get("domain")
        path = cookie.get("path") or "/"
        session.cookies.set(cookie["name"], cookie["value"], domain=domain, path=path)


def seed_cookies_from_mapping(session: requests.Session, cookies: dict[str, str]) -> None:
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=".eastmoney.com", path="/")


def refresh_rskey_cookie(session: requests.Session) -> None:
    response = session.get(EASTMONEY_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    match = re.search(r'var rskey = "([^"]+)"', response.text)
    if match:
        session.cookies.set("rskey", match.group(1), domain=".eastmoney.com", path="/")


def parse_jsonp_payload(text: str) -> dict:
    match = JSONP_PATTERN.match(text.strip())
    if not match:
        raise EastMoneySyncError(f"接口返回不是合法 JSONP，原始内容: {text[:200]}")

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise EastMoneySyncError(f"接口返回 JSON 解析失败: {exc}") from exc


class MyFavorClient:
    def __init__(self, session: requests.Session, mode: str, bid: str | None):
        if mode not in {"anonymous", "account"}:
            raise ValueError(f"不支持的模式: {mode}")
        self.session = session
        self.mode = mode
        self.bid = bid
        self.base_url = MYFAVOR_BASE_URL + ("webouter/" if mode == "account" else "anonymwebouter/")

    def _base_params(self) -> dict[str, str]:
        params = {"appkey": MYFAVOR_APPKEY}
        if self.mode == "anonymous":
            if not self.bid:
                raise EastMoneySyncError("匿名模式下缺少 qgqp_b_id。")
            params["bid"] = self.bid
        return params

    def _call(self, endpoint: str, **data: str) -> dict:
        params = {
            **self._base_params(),
            **data,
            "cb": f"jQuery_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "_": str(int(time.time() * 1000)),
        }
        response = self.session.get(
            self.base_url + endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = parse_jsonp_payload(response.text)
        if payload.get("state") != 0:
            raise EastMoneySyncError(payload.get("message") or f"{endpoint} 调用失败")
        return payload

    def group_list(self) -> list[dict]:
        payload = self._call("ggdefstkindexinfos")
        groups = payload.get("data", {}).get("ginfolist") or []
        return [
            {
                "gid": str(item["gid"]),
                "gname": str(item["gname"]),
            }
            for item in groups
        ]

    def add_group(self, group_name: str) -> dict:
        return self._call("ag", gn=group_name).get("data", {})

    def edit_group_name(self, gid: str, group_name: str) -> dict:
        return self._call("mg", g=str(gid), gn=group_name).get("data", {})

    def delete_group(self, gid: str) -> None:
        self._call("dg", g=str(gid))

    def group_stocks(self, gid: str) -> list[dict]:
        payload = self._call("gstkinfos", g=str(gid))
        return payload.get("data", {}).get("stkinfolist") or []

    def add_stock(self, gid: str, secid: str) -> None:
        self._call("as", g=str(gid), sc=secid.replace(".", "$"))


def prepare_client(
    mode: str,
    bid: str | None,
    bid_file: Path | None,
    cookie_header: str | None,
    cookie_file: Path | None,
    profile_dir: Path | None,
    browser_path: Path | None,
    headless: bool,
) -> tuple[MyFavorClient, str, str | None]:
    source_desc = ""
    resolved_mode = mode
    resolved_bid = bid
    session = create_requests_session(DEFAULT_USER_AGENT)
    profile_logged_in = False

    effective_cookie_header = cookie_header
    if effective_cookie_header:
        save_cookie_to_file(effective_cookie_header, cookie_file)
        source_desc = "手动提供的 Cookie"
    else:
        effective_cookie_header = load_cookie_from_file(cookie_file)
        if effective_cookie_header:
            source_desc = f"本地 Cookie 文件 {cookie_file}"

    if effective_cookie_header:
        cookies = parse_cookie_header(effective_cookie_header)
        seed_cookies_from_mapping(session, cookies)
        if resolved_bid is None:
            resolved_bid = cookies.get("qgqp_b_id")
        if resolved_mode == "auto":
            resolved_mode = "account"

    if not effective_cookie_header and profile_dir is not None and resolved_mode != "anonymous":
        if browser_path is None:
            raise EastMoneySyncError("需要从本地浏览器提取会话，但没有可用浏览器路径。")

        snapshot = extract_session_from_browser(
            profile_dir=profile_dir,
            browser_path=browser_path,
            headless=headless,
        )
        session = create_requests_session(snapshot.user_agent or DEFAULT_USER_AGENT)
        seed_cookies_from_snapshot(session, snapshot)
        if resolved_bid is None:
            resolved_bid = session.cookies.get("qgqp_b_id", domain=".eastmoney.com")
        profile_logged_in = snapshot.logged_in
        if profile_logged_in:
            if resolved_mode == "auto":
                resolved_mode = "account"
            source_desc = f"本地浏览器 profile {profile_dir}"

    if not effective_cookie_header and not profile_logged_in and resolved_bid is None:
        resolved_bid = load_bid_from_file(bid_file)
        if resolved_bid:
            source_desc = f"本地 bid 文件 {bid_file}"

    if resolved_mode == "account" and not effective_cookie_header and not profile_logged_in:
        raise EastMoneySyncError("账号模式需要 --cookie、可用的本地 Cookie 文件，或已登录浏览器 profile。")

    if resolved_mode == "auto":
        resolved_mode = "anonymous"

    if resolved_mode == "anonymous" and resolved_bid is None:
        resolved_bid = make_anonymous_bid()
        source_desc = "新生成的匿名 qgqp_b_id"

    if resolved_mode == "anonymous" and resolved_bid:
        session.cookies.set("qgqp_b_id", resolved_bid, domain=".eastmoney.com", path="/")
        save_bid_to_file(resolved_bid, bid_file)

    refresh_rskey_cookie(session)
    client = MyFavorClient(session=session, mode=resolved_mode, bid=resolved_bid)
    default_source_desc = "默认账号模式" if client.mode == "account" else "默认匿名模式"
    return client, source_desc or default_source_desc, resolved_bid


def sync_symbols_http(
    client: MyFavorClient,
    csv_path: Path,
    stocks: list[StockRef],
    base_group_name: str,
) -> tuple[str, str, int, list[tuple[StockRef, str]]]:
    groups = client.group_list()
    group_names = [item["gname"] for item in groups]
    final_group_name = build_unique_group_name(base_group_name, group_names)

    print(f"准备创建组合: {final_group_name}")
    print(f"将导入 {len(stocks)} 只股票，来源文件: {csv_path}")

    group_data = client.add_group(final_group_name)
    gid = str(group_data["gid"])

    failures: list[tuple[StockRef, str]] = []
    for index, stock in enumerate(stocks, start=1):
        label = f"{stock.symbol} {stock.name}" if stock.name else stock.symbol
        try:
            print(f"[{index}/{len(stocks)}] 添加 {label} -> {stock.secid}")
            client.add_stock(gid, stock.secid)
        except Exception as exc:
            failures.append((stock, str(exc)))
            print(f"[{index}/{len(stocks)}] 失败 {label}: {exc}", file=sys.stderr)

    remote_stocks = client.group_stocks(gid)
    return final_group_name, gid, len(remote_stocks), failures


def main() -> int:
    ensure_project_dirs()
    parser = build_parser()
    args = parser.parse_args()

    csv_path = resolve_input_path(args.input)
    stocks = load_stocks(
        csv_path=csv_path,
        symbol_column=args.column,
        file_symbol_column=args.file_symbol_column,
        name_column=args.name_column,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0。")
        stocks = stocks[: args.limit]

    group_name = args.group_name or default_group_name()
    profile_dir = resolve_profile_dir(args.profile_dir)
    bid_file_path = args.bid_file.expanduser().resolve() if args.bid_file else None
    cookie_file_path = args.cookie_file.expanduser().resolve() if args.cookie_file else None
    cached_cookie = load_cookie_from_file(cookie_file_path)
    browser_path = None
    needs_profile = (
        args.cookie is None
        and cached_cookie is None
        and profile_dir is not None
        and args.mode != "anonymous"
    )
    if needs_profile:
        browser_path = resolve_browser_path(args.browser_path)

    print(f"CSV 文件: {csv_path}")
    print(f"股票数量: {len(stocks)}")
    print(f"基础组合名: {group_name}")
    if profile_dir is not None:
        print(f"可用浏览器 profile: {profile_dir}")
    if args.bid_file is not None:
        print(f"bid 文件: {bid_file_path}")
    if args.cookie_file is not None:
        print(f"cookie 文件: {cookie_file_path}")

    if args.dry_run:
        print("dry-run 模式，不真正调用接口。")
        preview = ", ".join(f"{stock.symbol}->{stock.secid}" for stock in stocks[:20])
        print("股票代码预览:", preview)
        return 0

    client, source_desc, resolved_bid = prepare_client(
        mode=args.mode,
        bid=args.bid,
        bid_file=bid_file_path,
        cookie_header=args.cookie,
        cookie_file=cookie_file_path,
        profile_dir=profile_dir,
        browser_path=browser_path,
        headless=args.headless,
    )

    print(f"接口模式: {client.mode}")
    print(f"会话来源: {source_desc}")
    if client.mode == "anonymous" and resolved_bid:
        print(f"qgqp_b_id: {resolved_bid}")

    final_group_name, gid, remote_count, failures = sync_symbols_http(
        client=client,
        csv_path=csv_path,
        stocks=stocks,
        base_group_name=group_name,
    )

    print(f"完成：组合 {final_group_name} (gid={gid}) 当前共有 {remote_count} 只股票。")
    if failures:
        print(f"其中失败 {len(failures)} 只：", file=sys.stderr)
        for stock, message in failures:
            label = f"{stock.symbol} {stock.name}" if stock.name else stock.symbol
            print(f"- {label}: {message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消。")
        raise SystemExit(1)
    except Exception as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
