from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib import error, parse, request


SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING_RESULTS_DIR = SCRIPT_DIR / "screening_results"
OUTPUT_CSV = SCREENING_RESULTS_DIR / "maotai_600519_daily_qfq_2y.csv"
OUTPUT_META = SCREENING_RESULTS_DIR / "maotai_600519_daily_qfq_2y.meta.json"

SYMBOL = "600519"
MARKET_PREFIX = "sh"
DISPLAY_NAME = "贵州茅台"
ADJUSTMENT = "qfq"
KLINE_TYPE = "day"
LOOKBACK_YEARS = 2
REFRESH_BUFFER_DAYS = 30
MAX_API_RECORDS = 1000
API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "\
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"


class DownloadError(RuntimeError):
    """Raised when the remote data source returns an unusable response."""


@dataclass(frozen=True)
class DailyBar:
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def today_local() -> date:
    return date.today()


def load_meta() -> dict | None:
    if not OUTPUT_META.exists():
        return None
    with OUTPUT_META.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def load_existing_rows() -> list[dict]:
    if not OUTPUT_CSV.exists():
        return []
    with OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_opener_without_proxy() -> request.OpenerDirector:
    return request.build_opener(request.ProxyHandler({}))


def fetch_remote_rows(start_date: date, end_date: date) -> list[DailyBar]:
    symbol = f"{MARKET_PREFIX}{SYMBOL}"
    param = f"{symbol},{KLINE_TYPE},{start_date.isoformat()},{end_date.isoformat()},{MAX_API_RECORDS},{ADJUSTMENT}"
    query = parse.urlencode({"param": param})
    req = request.Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://gu.qq.com/",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    opener = build_opener_without_proxy()
    try:
        with opener.open(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise DownloadError(f"网络请求失败: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DownloadError("远端返回了无法解析的 JSON 数据") from exc

    if payload.get("code") != 0:
        raise DownloadError(f"接口返回错误: {payload.get('msg') or payload}")

    data = payload.get("data") or {}
    symbol_data = data.get(symbol) or {}
    key = f"{ADJUSTMENT}{KLINE_TYPE}"
    raw_rows = symbol_data.get(key) or []
    if not raw_rows:
        return []

    rows: list[DailyBar] = []
    for item in raw_rows:
        try:
            trade_date, open_, close, high, low, volume = item[:6]
            rows.append(
                DailyBar(
                    trade_date=trade_date,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise DownloadError(f"发现无法解析的 K 线记录: {item}") from exc

    rows.sort(key=lambda x: x.trade_date)
    return rows


def determine_fetch_window(existing_rows: list[dict], meta: dict | None) -> tuple[date, date, str]:
    end_date = today_local()
    baseline_start = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    if not existing_rows:
        return baseline_start, end_date, "full_bootstrap"

    cached_last_date = max(row["trade_date"] for row in existing_rows)
    cached_last = datetime.strptime(cached_last_date, "%Y-%m-%d").date()

    if meta and meta.get("cached_end_date"):
        try:
            meta_last = datetime.strptime(meta["cached_end_date"], "%Y-%m-%d").date()
            cached_last = max(cached_last, meta_last)
        except ValueError:
            pass

    start_date = max(baseline_start, cached_last - timedelta(days=REFRESH_BUFFER_DAYS))
    if start_date > end_date:
        start_date = end_date

    return start_date, end_date, "incremental_refresh"


def merge_rows(existing_rows: list[dict], fetched_rows: Iterable[DailyBar]) -> list[dict]:
    merged: dict[str, dict] = {}

    for row in existing_rows:
        trade_date = row["trade_date"]
        merged[trade_date] = {
            "trade_date": trade_date,
            "open": f"{float(row['open']):.3f}",
            "high": f"{float(row['high']):.3f}",
            "low": f"{float(row['low']):.3f}",
            "close": f"{float(row['close']):.3f}",
            "volume": f"{float(row['volume']):.3f}",
        }

    for bar in fetched_rows:
        merged[bar.trade_date] = {
            "trade_date": bar.trade_date,
            "open": f"{bar.open:.3f}",
            "high": f"{bar.high:.3f}",
            "low": f"{bar.low:.3f}",
            "close": f"{bar.close:.3f}",
            "volume": f"{bar.volume:.3f}",
        }

    ordered = [merged[key] for key in sorted(merged)]
    return enrich_rows(ordered)


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    prev_close: float | None = None

    for row in rows:
        close = float(row["close"])
        change = close - prev_close if prev_close is not None else 0.0
        pct_change = (change / prev_close * 100.0) if prev_close not in (None, 0.0) else 0.0

        enriched.append(
            {
                "trade_date": row["trade_date"],
                "symbol": SYMBOL,
                "market": MARKET_PREFIX,
                "name": DISPLAY_NAME,
                "adjustment": ADJUSTMENT,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "prev_close": f"{prev_close:.3f}" if prev_close is not None else "",
                "change": f"{change:.3f}",
                "pct_change": f"{pct_change:.4f}",
            }
        )
        prev_close = close

    return enriched


def trim_to_recent_two_years(rows: list[dict], end_date: date) -> list[dict]:
    floor_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)
    trimmed = [
        row for row in rows
        if datetime.strptime(row["trade_date"], "%Y-%m-%d").date() >= floor_date
    ]
    return trimmed


def write_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trade_date",
        "symbol",
        "market",
        "name",
        "adjustment",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "change",
        "pct_change",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_meta(
    rows: list[dict],
    fetch_start: date,
    fetch_end: date,
    run_mode: str,
    fetched_count: int,
) -> None:
    OUTPUT_META.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": DISPLAY_NAME,
        "symbol": SYMBOL,
        "market": MARKET_PREFIX,
        "adjustment": ADJUSTMENT,
        "frequency": KLINE_TYPE,
        "lookback_years": LOOKBACK_YEARS,
        "refresh_buffer_days": REFRESH_BUFFER_DAYS,
        "data_source": "Tencent fqkline API",
        "output_csv": OUTPUT_CSV.name,
        "fetched_window_start": fetch_start.isoformat(),
        "fetched_window_end": fetch_end.isoformat(),
        "fetched_rows_this_run": fetched_count,
        "run_mode": run_mode,
        "cached_start_date": rows[0]["trade_date"] if rows else None,
        "cached_end_date": rows[-1]["trade_date"] if rows else None,
        "cached_total_rows": len(rows),
        "last_synced_at": datetime.now().isoformat(timespec="seconds"),
    }
    with OUTPUT_META.open("w", encoding="utf-8-sig") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def print_summary(rows: list[dict], fetch_start: date, fetch_end: date, run_mode: str, fetched_count: int) -> None:
    print(f"运行模式: {run_mode}")
    print(f"抓取区间: {fetch_start.isoformat()} -> {fetch_end.isoformat()}")
    print(f"本次抓取记录数: {fetched_count}")
    print(f"缓存总记录数: {len(rows)}")
    print(f"CSV 文件: {OUTPUT_CSV}")
    print(f"元数据文件: {OUTPUT_META}")
    if rows:
        print(f"缓存覆盖区间: {rows[0]['trade_date']} -> {rows[-1]['trade_date']}")


def main() -> int:
    existing_rows = load_existing_rows()
    meta = load_meta()
    fetch_start, fetch_end, run_mode = determine_fetch_window(existing_rows, meta)

    remote_rows = fetch_remote_rows(fetch_start, fetch_end)
    merged_rows = merge_rows(existing_rows, remote_rows)
    trimmed_rows = trim_to_recent_two_years(merged_rows, fetch_end)

    write_csv(trimmed_rows)
    write_meta(trimmed_rows, fetch_start, fetch_end, run_mode, len(remote_rows))
    print_summary(trimmed_rows, fetch_start, fetch_end, run_mode, len(remote_rows))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
