from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd
import requests
from bs4 import BeautifulSoup


SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLISTS_DIR = SCRIPT_DIR / "watchlists"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "market_store_config.json"
SUPPORTED_MARKETS = {"sh", "sz", "bj"}

FQ_API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
MINUTE_API_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
CONCEPT_URL_TEMPLATE = "https://basic.10jqka.com.cn/{symbol}/concept.html"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
DEFAULT_MINUTE_FETCH_COUNT = 1200
PERIOD_TO_SNAPSHOT_OPEN_INDEX = {
    "daily": 5,
    "weekly": 47,
    "monthly": 46,
}
PERIOD_TO_SNAPSHOT_HIGH_INDEX = {
    "daily": 33,
    "weekly": 33,
    "monthly": 33,
}
PERIOD_TO_SNAPSHOT_LOW_INDEX = {
    "daily": 34,
    "weekly": 34,
    "monthly": 34,
}

BAR_COLUMNS = [
    "symbol",
    "market",
    "name",
    "period",
    "adjustment",
    "open",
    "high",
    "low",
    "close",
    "volume_lots",
    "volume",
    "amount",
    "turnover",
    "source",
]

STOCK_META_COLUMNS = [
    "symbol",
    "market",
    "full_symbol",
    "name",
    "concept_count",
    "synced_periods",
    "last_bar_sync_at",
    "last_concept_sync_at",
    "store_root",
]

CONCEPT_COLUMNS = [
    "symbol",
    "market",
    "full_symbol",
    "name",
    "concept_rank",
    "concept_name",
    "concept_id",
    "leader_symbols",
    "leader_names",
    "summary",
    "description",
    "source_url",
    "scraped_at",
]

STOCK_POOL_COLUMNS = [
    "symbol",
    "market",
    "name",
    "price",
    "market_cap",
    "market_cap_yi",
]


class SyncError(RuntimeError):
    """Raised when a remote data source returns an unusable response."""


class BaoStockClient:
    """Thin login-managed wrapper around Baostock history queries."""

    def __init__(self) -> None:
        self._logged_in = False

    @staticmethod
    def _is_login_error(message: str) -> bool:
        text = str(message or "").strip().lower()
        return any(
            marker in text
            for marker in (
                "未登录",
                "login",
                "logged",
            )
        )

    def _reset_login(self) -> None:
        self._logged_in = False
        try:
            bs.logout()
        except Exception:
            pass

    def ensure_login(self) -> None:
        if self._logged_in:
            return

        login_result = bs.login()
        if login_result.error_code != "0":
            raise SyncError(f"Baostock 登录失败: {login_result.error_msg}")
        self._logged_in = True

    def query_history(self, **kwargs: Any) -> tuple[list[str], list[list[str]]]:
        last_error: SyncError | None = None
        max_attempts = 4
        for attempt in range(max_attempts):
            self.ensure_login()
            result = bs.query_history_k_data_plus(**kwargs)
            if result.error_code == "0":
                rows: list[list[str]] = []
                while result.next():
                    rows.append(result.get_row_data())
                return list(getattr(result, "fields", [])), rows

            message = str(getattr(result, "error_msg", "") or "")
            last_error = SyncError(f"Baostock 查询失败: {message}")
            if attempt < max_attempts - 1 and self._is_login_error(message):
                log(f"Baostock session lost, retrying query attempt {attempt + 2}/{max_attempts}")
                self._reset_login()
                time.sleep(0.5 * (attempt + 1))
                continue
            raise last_error

        raise last_error or SyncError("Baostock query failed")

    def close(self) -> None:
        if not self._logged_in:
            return

        try:
            bs.logout()
        except Exception:
            pass
        self._logged_in = False


@dataclass(frozen=True)
class RunOptions:
    config_path: Path
    selected_periods: set[str] | None
    selected_stocks: set[str] | None
    sync_start_date: date | None
    sync_end_date: date | None
    workers: int
    skip_concepts: bool
    skip_bars: bool
    stock_pool_only: bool
    skip_store_metadata: bool


@dataclass(frozen=True)
class StockPoolConfig:
    enabled: bool
    refresh_on_sync: bool
    include_config_stocks: bool
    output_path: Path
    max_price: float
    min_market_cap: float
    exclude_prefixes: tuple[str, ...]
    exclude_st: bool


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{timestamp}] {message}", flush=True)
    except OSError:
        # Some long-running sessions lose their stdout pipe after the parent process exits.
        # Logging should never crash the sync worker.
        pass


def parse_csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def normalize_stock_selector(value: str) -> str:
    return value.strip().lower().replace(".", "")


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def infer_market_from_symbol(symbol: str) -> str:
    if not symbol.isdigit() or len(symbol) != 6:
        raise SyncError(f"股票代码必须是 6 位数字，收到: {symbol}")

    if symbol.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    if symbol.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "sz"
    if symbol.startswith("920") or symbol.startswith(("4", "8")):
        return "bj"

    raise SyncError(f"无法从股票代码自动判断市场，请显式提供 market: {symbol}")


def normalize_stock_entry(raw_stock: dict[str, Any]) -> dict[str, Any]:
    symbol_text = str(raw_stock.get("symbol", "") or "").strip()
    market_text = str(raw_stock.get("market", "") or "").strip().lower()
    name_text = str(raw_stock.get("name", "") or "").strip()

    compact_symbol = symbol_text.lower().replace(".", "")
    if compact_symbol.startswith(("sh", "sz", "bj")) and len(compact_symbol) == 8:
        if not market_text:
            market_text = compact_symbol[:2]
        symbol_text = compact_symbol[2:]

    if not symbol_text:
        raise SyncError("股票清单里存在空的 symbol")
    if not symbol_text.isdigit() or len(symbol_text) != 6:
        raise SyncError(f"股票代码必须是 6 位数字，收到: {symbol_text}")

    market = market_text or infer_market_from_symbol(symbol_text)
    if market not in SUPPORTED_MARKETS:
        raise SyncError(f"不支持的 market: {market}，股票代码: {symbol_text}")

    return {
        "symbol": symbol_text,
        "market": market,
        "name": name_text,
    }


def load_stocks_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SyncError(f"股票清单文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, dtype=str).fillna("")
        records = frame.to_dict(orient="records")
    elif suffix == ".parquet":
        frame = pd.read_parquet(path).fillna("")
        records = frame.to_dict(orient="records")
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("stocks") or []
        else:
            raise SyncError(f"不支持的 JSON 股票清单结构: {path}")
    elif suffix == ".txt":
        records = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            records.append({"symbol": item})
    else:
        raise SyncError(f"股票清单文件格式不支持: {path}")

    if not records:
        return []

    return [normalize_stock_entry(dict(record)) for record in records]


def load_stock_universe(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    stock_pool = load_stock_pool_config(config=config, config_path=config_path)

    if stock_pool.include_config_stocks:
        for raw_stock in config.get("stocks") or []:
            stock = normalize_stock_entry(dict(raw_stock))
            merged[stock_key(stock)] = stock

    stocks_file_value = str(config.get("stocks_file", "") or "").strip()
    if stocks_file_value:
        stocks_file_path = resolve_path(config_path.parent, stocks_file_value)
        for stock in load_stocks_from_file(stocks_file_path):
            merged[stock_key(stock)] = stock

    return list(merged.values())


def parse_bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_stock_pool_config(config: dict[str, Any], config_path: Path) -> StockPoolConfig:
    pool_raw = dict(config.get("stock_pool") or {})
    output_path_raw = str(
        pool_raw.get("output_path")
        or config.get("stocks_file")
        or "watchlists/stock_universe.csv"
    ).strip()
    exclude_prefixes = tuple(
        str(item).strip()
        for item in (pool_raw.get("exclude_prefixes") or ["688"])
        if str(item).strip()
    )

    return StockPoolConfig(
        enabled=parse_bool_config(pool_raw.get("enabled"), False),
        refresh_on_sync=parse_bool_config(pool_raw.get("refresh_on_sync"), True),
        include_config_stocks=parse_bool_config(pool_raw.get("include_config_stocks"), False),
        output_path=resolve_path(config_path.parent, output_path_raw),
        max_price=float(pool_raw.get("max_price") or 1000),
        min_market_cap=float(pool_raw.get("min_market_cap") or 12_000_000_000),
        exclude_prefixes=exclude_prefixes,
        exclude_st=parse_bool_config(pool_raw.get("exclude_st"), True),
    )


def infer_market_from_eastmoney_code(symbol: str) -> str:
    if symbol.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    if symbol.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "sz"
    if symbol.startswith("920") or symbol.startswith(("4", "8")):
        return "bj"
    raise SyncError(f"无法根据代码识别市场: {symbol}")


def is_st_stock(name: str) -> bool:
    normalized = str(name or "").strip().upper().replace(" ", "")
    return "ST" in normalized or "*ST" in normalized


def should_keep_stock_pool_row(row: dict[str, Any], stock_pool: StockPoolConfig) -> bool:
    symbol = str(row["symbol"])
    name = str(row["name"])
    price = float(row["price"])
    market_cap = float(row["market_cap"])

    if any(symbol.startswith(prefix) for prefix in stock_pool.exclude_prefixes):
        return False
    if stock_pool.exclude_st and is_st_stock(name):
        return False
    if not math.isfinite(price) or price <= 0 or price > stock_pool.max_price:
        return False
    if not math.isfinite(market_cap) or market_cap < stock_pool.min_market_cap:
        return False
    return True


def is_a_share_stock(code: str) -> bool:
    return code.startswith(
        (
            "sh.600",
            "sh.601",
            "sh.603",
            "sh.605",
            "sh.688",
            "sh.689",
            "sz.000",
            "sz.001",
            "sz.002",
            "sz.003",
            "sz.300",
            "sz.301",
            "bj.4",
            "bj.8",
            "bj.920",
        )
    )


def fetch_active_a_share_list() -> list[dict[str, str]]:
    login_result = bs.login()
    if login_result.error_code != "0":
        raise SyncError(f"Baostock 登录失败: {login_result.error_msg}")

    try:
        result = bs.query_stock_basic()
        if result.error_code != "0":
            raise SyncError(f"Baostock 股票列表查询失败: {result.error_msg}")

        stocks: list[dict[str, str]] = []
        while result.next():
            code, code_name, _ipo, _out, stock_type, status = result.get_row_data()
            if stock_type != "1" or status != "1":
                continue
            if not is_a_share_stock(code):
                continue

            market, symbol = code.split(".", 1)
            stocks.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "name": code_name,
                    "full_symbol": code,
                }
            )
        return stocks
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def fetch_tencent_quote_batch(
    session: requests.Session,
    stocks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not stocks:
        return []

    query_codes = ",".join(stock_key(stock) for stock in stocks)
    last_error: Exception | None = None
    text = ""
    for attempt in range(3):
        try:
            response = session.get(f"{TENCENT_QUOTE_URL}{query_codes}", timeout=20)
            response.raise_for_status()
            text = response.text
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    else:  # pragma: no cover
        raise last_error or SyncError("腾讯行情接口请求失败")

    stock_map = {stock_key(stock): stock for stock in stocks}
    rows: list[dict[str, Any]] = []
    for raw_line in text.split(";"):
        line = raw_line.strip()
        if not line or "=\"" not in line:
            continue

        header, payload = line.split("=\"", 1)
        quote_key = header.replace("v_", "").strip().lower()
        values = payload.rstrip('"').split("~")
        if len(values) <= 45:
            continue

        stock = stock_map.get(quote_key)
        if not stock:
            continue

        price = pd.to_numeric(values[3], errors="coerce")
        market_cap_yi = pd.to_numeric(values[45], errors="coerce")
        if pd.isna(price) or pd.isna(market_cap_yi):
            continue

        market_cap_yi_value = float(market_cap_yi)
        rows.append(
            {
                "symbol": stock["symbol"],
                "market": stock["market"],
                "name": values[1].strip() or stock["name"],
                "price": float(price),
                "market_cap": market_cap_yi_value * 100000000.0,
                "market_cap_yi": round(market_cap_yi_value, 2),
            }
        )

    return rows


def fetch_filtered_stock_pool(
    session: requests.Session,
    stock_pool: StockPoolConfig,
) -> pd.DataFrame:
    selected_rows: list[dict[str, Any]] = []
    candidate_stocks = fetch_active_a_share_list()
    chunk_size = 60

    for start in range(0, len(candidate_stocks), chunk_size):
        chunk = candidate_stocks[start : start + chunk_size]
        chunk_number = start // chunk_size + 1
        total_chunks = math.ceil(len(candidate_stocks) / chunk_size)
        log(f"refreshing stock pool quotes {chunk_number}/{total_chunks} ...")

        for row in fetch_tencent_quote_batch(session=session, stocks=chunk):
            if should_keep_stock_pool_row(row, stock_pool):
                selected_rows.append(row)

    frame = pd.DataFrame(selected_rows, columns=STOCK_POOL_COLUMNS)
    if frame.empty:
        return frame

    frame = frame.sort_values(["market_cap", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return frame


def write_stock_pool_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.copy()
    if ordered.empty:
        ordered = pd.DataFrame(columns=STOCK_POOL_COLUMNS)
    ordered.to_csv(path, index=False, encoding="utf-8-sig")


def filter_stock_pool_csv_by_name_keywords(path: Path, keywords: list[str]) -> tuple[int, int]:
    frame = pd.read_csv(path, dtype={"symbol": str}).fillna("")
    if frame.empty or not keywords:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return len(frame), 0

    normalized_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if not normalized_keywords:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return len(frame), 0

    name_series = frame["name"].astype(str)
    matched_mask = pd.Series(False, index=frame.index)
    for keyword in normalized_keywords:
        matched_mask = matched_mask | name_series.str.contains(keyword, regex=False)

    removed_count = int(matched_mask.sum())
    filtered = frame.loc[~matched_mask].copy()
    filtered.to_csv(path, index=False, encoding="utf-8-sig")
    return len(filtered), removed_count


def refresh_stock_pool_file(config: dict[str, Any], config_path: Path, session: requests.Session) -> pd.DataFrame:
    stock_pool = load_stock_pool_config(config=config, config_path=config_path)
    if not stock_pool.enabled:
        raise SyncError("stock_pool.enabled 未开启，无法刷新自动股票池。")

    frame = fetch_filtered_stock_pool(session=session, stock_pool=stock_pool)
    write_stock_pool_csv(stock_pool.output_path, frame)
    log(f"stock pool refreshed: {stock_pool.output_path} ({len(frame)} stocks)")
    return frame


def stock_matches_selector(stock: dict[str, Any], selectors: set[str] | None) -> bool:
    if not selectors:
        return True

    symbol = str(stock["symbol"]).lower()
    full_symbol = stock_key(stock).lower()
    return symbol in selectors or full_symbol in selectors


def iter_date_ranges(start_date: date, end_date: date, chunk_days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return ranges


def build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Referer": "https://gu.qq.com/",
            "Accept": "application/json,text/plain,*/*,*/*;q=0.8",
        }
    )
    return session


def close_session(session: requests.Session) -> None:
    try:
        session.close()
    except Exception:
        pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def resolve_history_start(spec: dict[str, Any], end_date: date) -> date:
    raw_start = spec.get("history_start")
    if raw_start:
        return parse_date(str(raw_start))

    lookback_years = int(spec.get("lookback_years", 5))
    return end_date - timedelta(days=lookback_years * 365)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_local() -> date:
    return date.today()


def clamp_sync_end_date(requested_end_date: date | None) -> date:
    today = today_local()
    if requested_end_date is None:
        return today
    return min(requested_end_date, today)


def resolve_sync_fetch_window(
    existing: pd.DataFrame,
    spec: dict[str, Any],
    default_end_date: date,
    incremental_buffer_days: int,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> tuple[date, date, date, str]:
    fetch_end = clamp_sync_end_date(requested_end_date or default_end_date)
    history_start = resolve_history_start(spec=spec, end_date=fetch_end)

    if requested_start_date is not None or requested_end_date is not None:
        explicit_start = requested_start_date or history_start
        fetch_start = max(history_start, explicit_start)
        if fetch_start > fetch_end:
            raise SyncError(
                f"同步日期范围无效: start={fetch_start.isoformat()} end={fetch_end.isoformat()}"
            )
        return history_start, fetch_start, fetch_end, "manual_window_refresh"

    has_backfill_gap = existing.empty or existing.index.min().date() > history_start
    if has_backfill_gap:
        return history_start, history_start, fetch_end, "bootstrap_backfill"

    fetch_start = max(history_start, existing.index.max().date() - timedelta(days=incremental_buffer_days))
    return history_start, fetch_start, fetch_end, "incremental_refresh"


def coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def stock_key(stock: dict[str, Any]) -> str:
    return f"{stock['market']}{stock['symbol']}"


def baostock_symbol(stock: dict[str, Any]) -> str:
    return f"{stock['market']}.{stock['symbol']}"


def empty_bar_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=BAR_COLUMNS)
    frame.index = pd.DatetimeIndex([], name="datetime")
    return frame


def empty_flat_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def normalize_bar_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    numeric_columns = ["open", "high", "low", "close", "volume_lots", "volume", "amount", "turnover"]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "volume_lots" not in normalized.columns:
        normalized["volume_lots"] = math.nan
    if "volume" not in normalized.columns:
        normalized["volume"] = math.nan

    missing_volume_lots = normalized["volume_lots"].isna() & normalized["volume"].notna()
    if missing_volume_lots.any():
        normalized.loc[missing_volume_lots, "volume_lots"] = normalized.loc[missing_volume_lots, "volume"] / 100.0

    missing_volume = normalized["volume"].isna() & normalized["volume_lots"].notna()
    if missing_volume.any():
        normalized.loc[missing_volume, "volume"] = normalized.loc[missing_volume, "volume_lots"] * 100.0

    return normalized


def read_bar_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_bar_frame()

    frame = pd.read_parquet(path)
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index("datetime")
    else:
        frame.index = pd.to_datetime(frame.index)
    frame.index.name = "datetime"

    for column in BAR_COLUMNS:
        if column not in frame.columns:
            frame[column] = math.nan if column in {"amount", "turnover"} else pd.NA

    frame = normalize_bar_frame_columns(frame)
    return frame[BAR_COLUMNS].sort_index()


def write_bar_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = normalize_bar_frame_columns(frame).sort_index()
    ordered.index = pd.to_datetime(ordered.index)
    ordered.index.name = "datetime"
    ordered = ordered[BAR_COLUMNS]
    ordered.to_parquet(path, engine="pyarrow", compression="snappy")


def read_flat_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return empty_flat_frame(columns)

    frame = pd.read_parquet(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def write_flat_frame(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.copy()
    for column in columns:
        if column not in ordered.columns:
            ordered[column] = pd.NA
    ordered = ordered[columns]
    ordered.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def merge_bar_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, incoming], axis=0).sort_index()

    if merged.empty:
        return empty_bar_frame()

    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def filter_bar_frame_by_spec(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    history_start: date | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return empty_bar_frame()

    filtered = frame.sort_index()
    if history_start is not None:
        filtered = filtered.loc[filtered.index.date >= history_start]
        if filtered.empty:
            return empty_bar_frame()

    keep_start = str(spec.get("intraday_keep_start") or "").strip()
    keep_end = str(spec.get("intraday_keep_end") or "").strip()
    if keep_start or keep_end:
        if not keep_start or not keep_end:
            raise SyncError("intraday_keep_start and intraday_keep_end must be set together")
        try:
            keep_start_text = datetime.strptime(keep_start, "%H:%M").strftime("%H:%M")
            keep_end_text = datetime.strptime(keep_end, "%H:%M").strftime("%H:%M")
        except ValueError as exc:
            raise SyncError(f"invalid intraday keep window: {keep_start} -> {keep_end}") from exc

        filtered = filtered.between_time(keep_start_text, keep_end_text, inclusive="both")
        if filtered.empty:
            return empty_bar_frame()

    return filtered.sort_index()


def resolve_name(stock: dict[str, Any], stock_payload: dict[str, Any]) -> str:
    if stock.get("name"):
        return str(stock["name"])

    quote = (stock_payload.get("qt") or {}).get(stock_key(stock)) or []
    if isinstance(quote, list) and len(quote) > 1 and quote[1]:
        return str(quote[1])

    return stock_key(stock)


def parse_tencent_quote_payload(text: str) -> dict[str, list[str]]:
    quotes: dict[str, list[str]] = {}
    for raw_line in text.split(";"):
        line = raw_line.strip()
        if not line or "=\"" not in line:
            continue

        header, payload = line.split("=\"", 1)
        quote_key = header.replace("v_", "").strip().lower()
        values = payload.rstrip('"').split("~")
        if quote_key and values:
            quotes[quote_key] = values
    return quotes


def fetch_tencent_quote_map(
    session: requests.Session,
    stocks: list[dict[str, Any]],
) -> dict[str, list[str]]:
    if not stocks:
        return {}

    query_codes = ",".join(stock_key(stock) for stock in stocks)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(f"{TENCENT_QUOTE_URL}{query_codes}", timeout=20)
            response.raise_for_status()
            return parse_tencent_quote_payload(response.text)
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

    raise last_error or SyncError("鑵捐琛屾儏鎺ュ彛璇锋眰澶辫触")


def quote_float(values: list[str], index: int) -> float | None:
    if index >= len(values):
        return None
    value = pd.to_numeric(values[index], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def quote_int(values: list[str], index: int) -> int | None:
    number = quote_float(values=values, index=index)
    if number is None:
        return None
    return int(number)


def build_bar_frame_from_records(
    records: list[dict[str, Any]],
    stock: dict[str, Any],
    period_name: str,
    adjustment: str,
    source: str,
    volume_mode: str,
) -> pd.DataFrame:
    if not records:
        return empty_bar_frame()

    frame = pd.DataFrame.from_records(records)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    numeric_columns = ["open", "high", "low", "close", "volume_lots", "volume", "amount", "turnover"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if volume_mode == "lots":
        if "volume_lots" not in frame.columns and "volume" in frame.columns:
            frame["volume_lots"] = frame["volume"] / 100.0
        frame["volume"] = frame["volume_lots"] * 100.0
    elif volume_mode == "shares":
        if "volume" not in frame.columns and "volume_lots" in frame.columns:
            frame["volume"] = frame["volume_lots"] * 100.0
        frame["volume_lots"] = frame["volume"] / 100.0
    else:
        raise SyncError(f"不支持的成交量模式: {volume_mode}")

    if "amount" not in frame.columns:
        frame["amount"] = math.nan
    if "turnover" not in frame.columns:
        frame["turnover"] = math.nan
    frame["symbol"] = stock["symbol"]
    frame["market"] = stock["market"]
    frame["name"] = stock["name"]
    frame["period"] = period_name
    frame["adjustment"] = adjustment
    frame["source"] = source

    frame = frame.set_index("datetime").sort_index()
    frame.index.name = "datetime"
    return frame[BAR_COLUMNS]


def should_backfill_today_from_quote(period_name: str) -> bool:
    if period_name not in PERIOD_TO_SNAPSHOT_OPEN_INDEX:
        return False
    now = datetime.now()
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return now >= market_close


def build_partial_period_bar_from_quote(
    stock: dict[str, Any],
    period_name: str,
    adjustment: str,
    quote_values: list[str],
    existing: pd.DataFrame,
) -> pd.DataFrame:
    trade_date_text = str(quote_values[30]).strip() if len(quote_values) > 30 else ""
    if len(trade_date_text) < 8:
        return empty_bar_frame()

    trade_dt = datetime.strptime(trade_date_text[:8], "%Y%m%d")
    close_value = quote_float(quote_values, 3)
    open_value = quote_float(quote_values, PERIOD_TO_SNAPSHOT_OPEN_INDEX[period_name])
    high_value = quote_float(quote_values, PERIOD_TO_SNAPSHOT_HIGH_INDEX[period_name])
    low_value = quote_float(quote_values, PERIOD_TO_SNAPSHOT_LOW_INDEX[period_name])
    volume_lots = quote_float(quote_values, 6)
    amount = quote_float(quote_values, 37)

    if None in (close_value, open_value, high_value, low_value, volume_lots):
        return empty_bar_frame()
    if volume_lots <= 0 and (existing.empty or existing.index.max().date() < trade_dt.date()):
        return empty_bar_frame()

    records = [
        {
            "datetime": trade_dt,
            "open": open_value,
            "close": close_value,
            "high": high_value,
            "low": low_value,
            "volume_lots": volume_lots,
            "amount": amount,
        }
    ]
    return build_bar_frame_from_records(
        records=records,
        stock=stock,
        period_name=period_name,
        adjustment=adjustment,
        source="tencent_quote_snapshot",
        volume_mode="lots",
    )


def fetch_fq_window(
    session: requests.Session,
    stock: dict[str, Any],
    remote_period: str,
    start_date: date,
    end_date: date,
    max_records: int,
    adjustment: str,
    period_name: str,
) -> tuple[pd.DataFrame, str]:
    params = {
        "param": (
            f"{stock_key(stock)},{remote_period},"
            f"{start_date.isoformat()},{end_date.isoformat()},{max_records},{adjustment}"
        )
    }
    response = session.get(FQ_API_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise SyncError(f"{stock_key(stock)} {period_name} 接口返回异常: {payload.get('msg') or payload}")

    stock_payload = (payload.get("data") or {}).get(stock_key(stock)) or {}
    stock["name"] = resolve_name(stock, stock_payload)
    raw_rows = stock_payload.get(f"{adjustment}{remote_period}") or []

    records = []
    for row in raw_rows:
        if len(row) < 6:
            continue
        records.append(
            {
                "datetime": row[0],
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume_lots": row[5],
            }
        )

    frame = build_bar_frame_from_records(
        records=records,
        stock=stock,
        period_name=period_name,
        adjustment=adjustment,
        source="tencent_fqkline",
        volume_mode="lots",
    )
    return frame, stock["name"]


def fetch_fq_history(
    session: requests.Session,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    chunk_days = int(spec["chunk_days"])
    max_records = int(spec.get("max_records", 640))
    remote_period = str(spec["remote_period"])
    adjustment = str(spec["adjustment"])

    frames: list[pd.DataFrame] = []
    resolved_name = stock.get("name") or stock_key(stock)
    ranges = iter_date_ranges(start_date, end_date, chunk_days)
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        chunk_started_at = time.perf_counter()
        frame, resolved_name = fetch_fq_window(
            session=session,
            stock=stock,
            remote_period=remote_period,
            start_date=chunk_start,
            end_date=chunk_end,
            max_records=max_records,
            adjustment=adjustment,
            period_name=period_name,
        )
        if not frame.empty:
            frames.append(frame)
        elapsed = time.perf_counter() - chunk_started_at
        log(
            f"{stock_key(stock)} {period_name} chunk {index}/{len(ranges)} "
            f"{chunk_start.isoformat()} -> {chunk_end.isoformat()} "
            f"rows={len(frame)} elapsed={elapsed:.1f}s"
        )

    if not frames:
        return empty_bar_frame(), resolved_name

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, resolved_name


def fetch_single_tencent_quote(
    session: requests.Session,
    stock: dict[str, Any],
) -> list[str] | None:
    quote_map = fetch_tencent_quote_map(session=session, stocks=[stock])
    return quote_map.get(stock_key(stock))


def build_daily_bar_from_quote(
    stock: dict[str, Any],
    adjustment: str,
    quote_values: list[str],
) -> pd.DataFrame:
    trade_date_text = str(quote_values[30]).strip() if len(quote_values) > 30 else ""
    if len(trade_date_text) < 8:
        return empty_bar_frame()

    trade_dt = datetime.strptime(trade_date_text[:8], "%Y%m%d")
    close_value = quote_float(quote_values, 3)
    previous_close_value = quote_float(quote_values, 4)
    open_value = quote_float(quote_values, 5)
    high_value = quote_float(quote_values, 33)
    low_value = quote_float(quote_values, 34)
    volume_lots = quote_float(quote_values, 6)
    amount_yuan = quote_float(quote_values, 37)

    if (close_value is None or close_value <= 0) and previous_close_value is not None and previous_close_value > 0:
        close_value = previous_close_value

    halted_snapshot = (
        close_value is not None
        and close_value > 0
        and volume_lots is not None
        and volume_lots == 0
        and all(value is not None and value <= 0 for value in (open_value, high_value, low_value))
    )
    if halted_snapshot:
        open_value = close_value
        high_value = close_value
        low_value = close_value

    if None in (close_value, open_value, high_value, low_value, volume_lots):
        return empty_bar_frame()
    if volume_lots < 0:
        return empty_bar_frame()

    return build_bar_frame_from_records(
        records=[
            {
                "datetime": trade_dt,
                "open": open_value,
                "close": close_value,
                "high": high_value,
                "low": low_value,
                "volume_lots": volume_lots,
                "amount": amount_yuan,
            }
        ],
        stock=stock,
        period_name="daily",
        adjustment=adjustment,
        source="tencent_quote_snapshot",
        volume_mode="lots",
    )


def aggregate_bar_frame(
    daily_frame: pd.DataFrame,
    stock: dict[str, Any],
    period_name: str,
    adjustment: str,
) -> pd.DataFrame:
    if daily_frame.empty:
        return empty_bar_frame()

    frame = daily_frame.copy().sort_index()
    numeric_columns = ["open", "high", "low", "close", "volume_lots", "volume", "amount", "turnover"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if period_name == "weekly":
        # Use Friday-ending business weeks so Monday-Friday bars stay in the same A-share week.
        group_keys = frame.index.to_period("W-FRI")
    elif period_name == "monthly":
        group_keys = frame.index.to_period("M").start_time
    else:
        raise SyncError(f"不支持从日线聚合周期: {period_name}")

    records: list[dict[str, Any]] = []
    for group_start, group in frame.groupby(group_keys):
        group = group.sort_index()
        label_datetime = group.index.min() if period_name == "weekly" else group.index.max()
        records.append(
            {
                "datetime": label_datetime,
                "open": group["open"].iloc[0],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group["close"].iloc[-1],
                "volume_lots": group["volume_lots"].sum(min_count=1),
                "volume": group["volume"].sum(min_count=1) if "volume" in group.columns else math.nan,
                "amount": group["amount"].sum(min_count=1) if "amount" in group.columns else math.nan,
                "turnover": group["turnover"].sum(min_count=1) if "turnover" in group.columns else math.nan,
            }
        )

    return build_bar_frame_from_records(
        records=records,
        stock=stock,
        period_name=period_name,
        adjustment=adjustment,
        source="derived_from_daily",
        volume_mode="lots",
    )


def explicit_daily_fallback_window(
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> tuple[date, date] | None:
    if requested_start_date is None and requested_end_date is None:
        return None

    window_start = requested_start_date or requested_end_date
    window_end = requested_end_date or requested_start_date
    if window_start is None or window_end is None:
        return None
    if window_end >= today_local():
        return None
    if window_start > window_end:
        raise SyncError(f"invalid fallback window: {window_start.isoformat()} > {window_end.isoformat()}")
    return window_start, window_end


def aggregate_daily_frame_from_minute(
    minute_frame: pd.DataFrame,
    stock: dict[str, Any],
    adjustment: str,
    source_period: str,
) -> pd.DataFrame:
    if minute_frame.empty:
        return empty_bar_frame()

    frame = minute_frame.copy().sort_index()
    numeric_columns = ["open", "high", "low", "close", "volume_lots", "volume", "amount", "turnover"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    records: list[dict[str, Any]] = []
    for trade_day, group in frame.groupby(frame.index.normalize()):
        group = group.sort_index()
        records.append(
            {
                "datetime": trade_day.to_pydatetime(),
                "open": group["open"].iloc[0],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group["close"].iloc[-1],
                "volume_lots": group["volume_lots"].sum(min_count=1),
                "volume": group["volume"].sum(min_count=1) if "volume" in group.columns else math.nan,
                "amount": group["amount"].sum(min_count=1) if "amount" in group.columns else math.nan,
                "turnover": group["turnover"].sum(min_count=1) if "turnover" in group.columns else math.nan,
            }
        )

    return build_bar_frame_from_records(
        records=records,
        stock=stock,
        period_name="daily",
        adjustment=adjustment,
        source=f"derived_from_{source_period}",
        volume_mode="lots",
    )


def write_daily_period_from_local_minute(
    stock: dict[str, Any],
    spec: dict[str, Any],
    store_root: Path,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> dict[str, Any] | None:
    fallback_window = explicit_daily_fallback_window(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )
    if fallback_window is None:
        return None

    window_start, window_end = fallback_window
    history_start = resolve_history_start(spec=spec, end_date=window_end)
    daily_path = store_root / "daily" / f"{stock_key(stock)}.parquet"
    existing_daily = read_bar_frame(daily_path)

    for source_period in ("60m", "30m", "15m", "5m"):
        minute_path = store_root / source_period / f"{stock_key(stock)}.parquet"
        minute_frame = read_bar_frame(minute_path)
        if minute_frame.empty:
            continue

        scoped = minute_frame.loc[
            (minute_frame.index.date >= window_start) & (minute_frame.index.date <= window_end)
        ]
        if scoped.empty:
            continue

        derived_daily = aggregate_daily_frame_from_minute(
            minute_frame=scoped,
            stock=stock,
            adjustment=str(spec["adjustment"]),
            source_period=source_period,
        )
        if derived_daily.empty:
            continue

        merged_daily = merge_bar_frames(existing_daily, derived_daily)
        merged_daily = filter_bar_frame_by_spec(merged_daily, spec=spec, history_start=history_start)
        write_bar_frame(daily_path, merged_daily)
        return {
            "file": str(daily_path),
            "source_type": "fq",
            "remote_period": spec["remote_period"],
            "adjustment": spec["adjustment"],
            "run_mode": "derived_from_local_minute",
            "fetch_window_start": window_start.isoformat(),
            "fetch_window_end": window_end.isoformat(),
            "fetched_rows_this_run": int(len(derived_daily)),
            "cached_rows": int(len(merged_daily)),
            "cached_start": merged_daily.index.min().strftime("%Y-%m-%d %H:%M:%S") if not merged_daily.empty else None,
            "cached_end": merged_daily.index.max().strftime("%Y-%m-%d %H:%M:%S") if not merged_daily.empty else None,
            "update_note": f"rebuilt explicit daily window from local {source_period} bars",
            "today_bar_backfilled_from_quote": False,
        }

    return None


def write_derived_period_from_daily(
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    store_root: Path,
) -> dict[str, Any] | None:
    daily_path = store_root / "daily" / f"{stock_key(stock)}.parquet"
    daily_frame = read_bar_frame(daily_path)
    if daily_frame.empty:
        return None

    derived_frame = aggregate_bar_frame(
        daily_frame=daily_frame,
        stock=stock,
        period_name=period_name,
        adjustment=str(spec["adjustment"]),
    )
    if derived_frame.empty:
        return None

    period_path = store_root / period_name / f"{stock_key(stock)}.parquet"
    write_bar_frame(period_path, derived_frame)
    return {
        "file": str(period_path),
        "source_type": "fq",
        "remote_period": spec["remote_period"],
        "adjustment": spec["adjustment"],
        "run_mode": "derived_from_daily",
        "fetch_window_start": None,
        "fetch_window_end": today_local().isoformat(),
        "fetched_rows_this_run": 0,
        "cached_rows": int(len(derived_frame)),
        "cached_start": derived_frame.index.min().strftime("%Y-%m-%d %H:%M:%S") if not derived_frame.empty else None,
        "cached_end": derived_frame.index.max().strftime("%Y-%m-%d %H:%M:%S") if not derived_frame.empty else None,
        "update_note": "rebuilt from local daily bars",
        "today_bar_backfilled_from_quote": False,
    }


def fetch_minute_window(
    session: requests.Session,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    fetch_count = int(spec.get("fetch_count") or DEFAULT_MINUTE_FETCH_COUNT)
    remote_period = str(spec["remote_period"]).strip()
    minute_period = remote_period if remote_period.startswith("m") else f"m{remote_period}"
    params = {"param": f"{stock_key(stock)},{minute_period},,{fetch_count}"}
    response = session.get(MINUTE_API_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise SyncError(f"{stock_key(stock)} {period_name} 接口返回异常: {payload.get('msg') or payload}")

    stock_payload = (payload.get("data") or {}).get(stock_key(stock)) or {}
    stock["name"] = resolve_name(stock, stock_payload)
    raw_rows = stock_payload.get(minute_period) or stock_payload.get(remote_period) or []

    records = []
    for row in raw_rows:
        if len(row) < 6:
            continue
        records.append(
            {
                "datetime": datetime.strptime(row[0], "%Y%m%d%H%M"),
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume_lots": row[5],
            }
        )

    frame = build_bar_frame_from_records(
        records=records,
        stock=stock,
        period_name=period_name,
        adjustment=str(spec["adjustment"]),
        source="tencent_mkline",
        volume_mode="lots",
    )
    return frame, stock["name"]


def fetch_baostock_minute_history(
    baostock_client: BaoStockClient,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    chunk_days = int(spec.get("chunk_days", 90))
    ranges = iter_date_ranges(start_date, end_date, chunk_days)

    frames: list[pd.DataFrame] = []
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        chunk_started_at = time.perf_counter()
        _, rows = baostock_client.query_history(
            code=baostock_symbol(stock),
            fields="date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=chunk_start.isoformat(),
            end_date=chunk_end.isoformat(),
            frequency=str(spec["remote_period"]),
            adjustflag=str(spec.get("adjustflag", "2")),
        )

        records: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 9:
                continue
            records.append(
                {
                    "datetime": datetime.strptime(row[1][:12], "%Y%m%d%H%M"),
                    "open": row[3],
                    "high": row[4],
                    "low": row[5],
                    "close": row[6],
                    "volume": row[7],
                    "amount": row[8],
                }
            )

        frame = build_bar_frame_from_records(
            records=records,
            stock=stock,
            period_name=period_name,
            adjustment=str(spec["adjustment"]),
            source="baostock_minute",
            volume_mode="shares",
        )
        if not frame.empty:
            frames.append(frame)

        elapsed = time.perf_counter() - chunk_started_at
        log(
            f"{stock_key(stock)} {period_name} chunk {index}/{len(ranges)} "
            f"{chunk_start.isoformat()} -> {chunk_end.isoformat()} "
            f"rows={len(frame)} elapsed={elapsed:.1f}s"
        )

    if not frames:
        return empty_bar_frame(), stock.get("name") or stock_key(stock)

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, stock.get("name") or stock_key(stock)


def sync_fq_period(
    session: requests.Session,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    store_root: Path,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> dict[str, Any]:
    file_path = store_root / period_name / f"{stock_key(stock)}.parquet"
    existing = read_bar_frame(file_path)
    history_start = resolve_history_start(spec=spec, end_date=today_local())

    history_start, fetch_start, fetch_end, run_mode = resolve_sync_fetch_window(
        existing=existing,
        spec=spec,
        default_end_date=today_local(),
        incremental_buffer_days=int(spec.get("incremental_buffer_days", 30)),
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )

    incoming, resolved_name = fetch_fq_history(
        session=session,
        stock=stock,
        period_name=period_name,
        spec=spec,
        start_date=fetch_start,
        end_date=fetch_end,
    )
    stock["name"] = resolved_name
    today_bar_backfilled = False
    if period_name == "daily" and should_backfill_today_from_quote(period_name=period_name):
        latest_incoming_date = incoming.index.max().date() if not incoming.empty else None
        latest_existing_date = existing.index.max().date() if not existing.empty else None
        latest_cached_date = max(
            [item for item in (latest_existing_date, latest_incoming_date) if item is not None],
            default=None,
        )
        if latest_cached_date is None or latest_cached_date < fetch_end:
            quote_values = fetch_single_tencent_quote(session=session, stock=stock)
            if quote_values:
                snapshot_frame = build_daily_bar_from_quote(
                    stock=stock,
                    adjustment=str(spec["adjustment"]),
                    quote_values=quote_values,
                )
                if not snapshot_frame.empty:
                    incoming = merge_bar_frames(incoming, snapshot_frame)
                    today_bar_backfilled = True

    if existing.empty and incoming.empty and resolved_name == stock_key(stock):
        raise SyncError(f"{stock_key(stock)} 未返回有效日线数据，请检查股票代码是否正确")

    existing = filter_bar_frame_by_spec(existing, spec=spec, history_start=history_start)
    incoming = filter_bar_frame_by_spec(incoming, spec=spec, history_start=history_start)
    merged = merge_bar_frames(existing, incoming)
    merged = filter_bar_frame_by_spec(merged, spec=spec, history_start=history_start)
    write_bar_frame(file_path, merged)

    summary = {
        "file": str(file_path),
        "source_type": "fq",
        "remote_period": spec["remote_period"],
        "adjustment": spec["adjustment"],
        "run_mode": run_mode,
        "fetch_window_start": fetch_start.isoformat(),
        "fetch_window_end": fetch_end.isoformat(),
        "fetched_rows_this_run": int(len(incoming)),
        "cached_rows": int(len(merged)),
        "cached_start": merged.index.min().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "cached_end": merged.index.max().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "update_note": "chunked full-history fetch for long-term parquet storage",
        "today_bar_backfilled_from_quote": today_bar_backfilled,
    }
    return summary


def sync_minute_period(
    session: requests.Session,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    store_root: Path,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> dict[str, Any]:
    if requested_start_date is not None or requested_end_date is not None:
        raise SyncError(f"{stock_key(stock)} {period_name} 腾讯分钟线源不支持按日期窗口重刷")

    file_path = store_root / period_name / f"{stock_key(stock)}.parquet"
    existing = read_bar_frame(file_path)
    history_start = resolve_history_start(spec=spec, end_date=today_local())
    incoming, resolved_name = fetch_minute_window(
        session=session,
        stock=stock,
        period_name=period_name,
        spec=spec,
    )
    stock["name"] = resolved_name

    if existing.empty and incoming.empty and resolved_name == stock_key(stock):
        raise SyncError(f"{stock_key(stock)} 未返回有效分钟线数据，请检查股票代码是否正确")

    existing = filter_bar_frame_by_spec(existing, spec=spec, history_start=history_start)
    incoming = filter_bar_frame_by_spec(incoming, spec=spec, history_start=history_start)
    merged = merge_bar_frames(existing, incoming)
    merged = filter_bar_frame_by_spec(merged, spec=spec, history_start=history_start)
    write_bar_frame(file_path, merged)

    overlap_ok = existing.empty or incoming.empty or existing.index.max() >= incoming.index.min()
    summary = {
        "file": str(file_path),
        "source_type": "minute",
        "remote_period": spec["remote_period"],
        "adjustment": spec["adjustment"],
        "run_mode": "recent_bootstrap" if existing.empty else "rolling_incremental_archive",
        "source_window_rows": int(len(incoming)),
        "source_window_start": incoming.index.min().strftime("%Y-%m-%d %H:%M:%S") if not incoming.empty else None,
        "source_window_end": incoming.index.max().strftime("%Y-%m-%d %H:%M:%S") if not incoming.empty else None,
        "cached_rows": int(len(merged)),
        "cached_start": merged.index.min().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "cached_end": merged.index.max().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "overlap_ok": overlap_ok,
        "update_note": (
            "source only exposes a recent rolling window; keep syncing to accumulate a longer local archive"
        ),
    }
    return summary


def sync_minute_with_fallback(
    session: requests.Session,
    baostock_client: BaoStockClient | None,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    store_root: Path,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> dict[str, Any]:
    fallback_sources = [str(item).strip() for item in (spec.get("fallback_sources") or []) if str(item).strip()]
    attempts = [str(spec["source_type"]).strip(), *fallback_sources]
    errors: list[str] = []

    for attempt_index, source_type in enumerate(attempts, start=1):
        source_spec = dict(spec)
        source_spec["source_type"] = source_type
        try:
            if source_type == "minute_baostock":
                if baostock_client is None:
                    raise SyncError("minute_baostock sync requires an active Baostock client")
                summary = sync_baostock_minute_period(
                    baostock_client=baostock_client,
                    stock=stock,
                    period_name=period_name,
                    spec=source_spec,
                    store_root=store_root,
                    requested_start_date=requested_start_date,
                    requested_end_date=requested_end_date,
                )
            elif source_type == "minute":
                summary = sync_minute_period(
                    session=session,
                    stock=stock,
                    period_name=period_name,
                    spec=source_spec,
                    store_root=store_root,
                    requested_start_date=requested_start_date,
                    requested_end_date=requested_end_date,
                )
            else:
                raise SyncError(f"{stock_key(stock)} {period_name} 不支持的分钟线数据源: {source_type}")

            summary["requested_source_type"] = str(spec["source_type"])
            summary["actual_source_type"] = source_type
            if attempt_index > 1:
                summary["fallback_used"] = True
                summary["fallback_errors"] = errors
                log(
                    f"{stock_key(stock)} {period_name} fallback source succeeded: "
                    f"{source_type} after {attempt_index - 1} failure(s)"
                )
            else:
                summary["fallback_used"] = False
            return summary
        except Exception as exc:
            error_text = f"{source_type}: {exc}"
            errors.append(error_text)
            if attempt_index < len(attempts):
                log(
                    f"{stock_key(stock)} {period_name} source failed, trying fallback "
                    f"{attempt_index}/{len(attempts) - 1}: {error_text}"
                )
                continue
            raise SyncError(" | ".join(errors))

    raise SyncError(f"{stock_key(stock)} {period_name} minute sync failed without attempts")


def sync_baostock_minute_period(
    baostock_client: BaoStockClient,
    stock: dict[str, Any],
    period_name: str,
    spec: dict[str, Any],
    store_root: Path,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> dict[str, Any]:
    file_path = store_root / period_name / f"{stock_key(stock)}.parquet"
    existing = read_bar_frame(file_path)

    history_start, fetch_start, end_date, run_mode = resolve_sync_fetch_window(
        existing=existing,
        spec=spec,
        default_end_date=today_local(),
        incremental_buffer_days=int(spec.get("incremental_buffer_days", 7)),
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
    )

    incoming, resolved_name = fetch_baostock_minute_history(
        baostock_client=baostock_client,
        stock=stock,
        period_name=period_name,
        spec=spec,
        start_date=fetch_start,
        end_date=end_date,
    )
    stock["name"] = resolved_name

    if existing.empty and incoming.empty:
        raise SyncError(f"{stock_key(stock)} 未返回有效分钟线数据，请检查股票代码是否正确或该源是否支持该标的")

    merged = merge_bar_frames(existing, incoming)
    merged = merged[merged.index.date >= history_start]
    write_bar_frame(file_path, merged)

    summary = {
        "file": str(file_path),
        "source_type": "minute_baostock",
        "remote_period": spec["remote_period"],
        "adjustment": spec["adjustment"],
        "run_mode": run_mode,
        "fetch_window_start": fetch_start.isoformat(),
        "fetch_window_end": end_date.isoformat(),
        "fetched_rows_this_run": int(len(incoming)),
        "cached_rows": int(len(merged)),
        "cached_start": merged.index.min().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "cached_end": merged.index.max().strftime("%Y-%m-%d %H:%M:%S") if not merged.empty else None,
        "update_note": (
            "Baostock minute history was used for long-horizon archive sync; in local tests it returned "
            "complete minute history for sh600519 from 2020-01-02 onward."
        ),
    }
    return summary


def fetch_concepts(session: requests.Session, stock: dict[str, Any]) -> pd.DataFrame:
    url = CONCEPT_URL_TEMPLATE.format(symbol=stock["symbol"])
    response = session.get(url, timeout=20)
    response.raise_for_status()
    response.encoding = "gbk"

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", class_="gnContent")
    if table is None:
        return empty_flat_frame(CONCEPT_COLUMNS)

    rows = table.find_all("tr")
    scraped_at = now_iso()
    records: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        classes = row.get("class", [])
        if "extend_content" in classes:
            continue

        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        detail_text = ""
        if index + 1 < len(rows) and "extend_content" in rows[index + 1].get("class", []):
            detail_text = rows[index + 1].get_text(" ", strip=True)

        concept_cell = cells[1]
        leader_links = row.find_all("a", class_="gnltg")
        leader_symbols = "|".join(link.get("code", "").strip() for link in leader_links if link.get("code"))
        leader_names = "|".join(link.get_text(strip=True) for link in leader_links if link.get_text(strip=True))
        summary_text = cells[3].get_text(" ", strip=True).replace("展开", "").strip()

        records.append(
            {
                "symbol": stock["symbol"],
                "market": stock["market"],
                "full_symbol": stock_key(stock),
                "name": stock["name"],
                "concept_rank": int(cells[0].get_text(strip=True)),
                "concept_name": concept_cell.get_text(" ", strip=True),
                "concept_id": concept_cell.get("clid"),
                "leader_symbols": leader_symbols,
                "leader_names": leader_names,
                "summary": summary_text,
                "description": detail_text or summary_text,
                "source_url": url,
                "scraped_at": scraped_at,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return empty_flat_frame(CONCEPT_COLUMNS)
    return frame[CONCEPT_COLUMNS]


def replace_rows_by_keys(
    frame: pd.DataFrame,
    key_column: str,
    keys_to_remove: set[str],
    replacement: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    remaining = frame.copy()
    if not remaining.empty and key_column in remaining.columns and keys_to_remove:
        key_series = remaining[key_column].astype(str)
        remaining = remaining.loc[~key_series.isin(keys_to_remove)].copy()

    if replacement.empty:
        combined = remaining.reset_index(drop=True)
    else:
        combined = pd.concat([remaining, replacement], axis=0, ignore_index=True)

    for column in columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[columns].reset_index(drop=True)


def ordered_period_items(period_specs: dict[str, Any]) -> list[tuple[str, Any]]:
    fq_priority = {
        "daily": 0,
        "weekly": 1,
        "monthly": 2,
    }
    ordered = sorted(
        enumerate(period_specs.items()),
        key=lambda item: (
            0
            if str(item[1][1].get("source_type")) == "fq" and item[1][0] in fq_priority
            else 1,
            fq_priority.get(item[1][0], 99),
            item[0],
        ),
    )
    return [period_item for _, period_item in ordered]


def sync_stock(
    session: requests.Session,
    baostock_client: BaoStockClient | None,
    stock: dict[str, Any],
    period_specs: dict[str, Any],
    store_root: Path,
    skip_concepts: bool,
    skip_bars: bool,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    period_results: dict[str, Any] = {}
    if not skip_bars:
        for period_name, spec in ordered_period_items(period_specs):
            log(f"{stock_key(stock)} syncing {period_name} ...")
            if spec["source_type"] != "fq":
                period_results[period_name] = sync_minute_with_fallback(
                    session=session,
                    baostock_client=baostock_client,
                    stock=stock,
                    period_name=period_name,
                    spec=spec,
                    store_root=store_root,
                    requested_start_date=requested_start_date,
                    requested_end_date=requested_end_date,
                )
                continue

            if period_name == "weekly":
                if period_name in period_results:
                    continue

                derived_summary = write_derived_period_from_daily(
                    stock=stock,
                    period_name=period_name,
                    spec=spec,
                    store_root=store_root,
                )
                if not derived_summary:
                    raise SyncError(
                        f"{stock_key(stock)} weekly 依赖本地 daily 数据；"
                        "请先同步 daily 或在本次运行中包含 daily 周期。"
                    )
                period_results[period_name] = derived_summary
                continue

            try:
                period_results[period_name] = sync_fq_period(
                    session=session,
                    stock=stock,
                    period_name=period_name,
                    spec=spec,
                    store_root=store_root,
                    requested_start_date=requested_start_date,
                    requested_end_date=requested_end_date,
                )
                if period_name == "daily":
                    for derived_period in ("weekly", "monthly"):
                        derived_spec = period_specs.get(derived_period)
                        if not derived_spec or str(derived_spec.get("source_type")) != "fq":
                            continue
                        derived_summary = write_derived_period_from_daily(
                            stock=stock,
                            period_name=derived_period,
                            spec=derived_spec,
                            store_root=store_root,
                        )
                        if derived_summary:
                            period_results[derived_period] = derived_summary
            except Exception as exc:
                if period_name == "daily":
                    local_minute_summary = write_daily_period_from_local_minute(
                        stock=stock,
                        spec=spec,
                        store_root=store_root,
                        requested_start_date=requested_start_date,
                        requested_end_date=requested_end_date,
                    )
                    if local_minute_summary:
                        local_minute_summary["update_note"] = (
                            f"{local_minute_summary['update_note']}; fallback after fq error: {exc}"
                        )
                        period_results[period_name] = local_minute_summary
                        for derived_period in ("weekly", "monthly"):
                            derived_spec = period_specs.get(derived_period)
                            if not derived_spec or str(derived_spec.get("source_type")) != "fq":
                                continue
                            derived_summary = write_derived_period_from_daily(
                                stock=stock,
                                period_name=derived_period,
                                spec=derived_spec,
                                store_root=store_root,
                            )
                            if derived_summary:
                                period_results[derived_period] = derived_summary
                        continue

                    quote_values = fetch_single_tencent_quote(session=session, stock=stock)
                    if quote_values and should_backfill_today_from_quote(period_name="daily"):
                        daily_path = store_root / "daily" / f"{stock_key(stock)}.parquet"
                        existing_daily = read_bar_frame(daily_path)
                        snapshot_daily = build_daily_bar_from_quote(
                            stock=stock,
                            adjustment=str(spec["adjustment"]),
                            quote_values=quote_values,
                        )
                        if not snapshot_daily.empty:
                            merged_daily = merge_bar_frames(existing_daily, snapshot_daily)
                            write_bar_frame(daily_path, merged_daily)
                            period_results[period_name] = {
                                "file": str(daily_path),
                                "source_type": "fq",
                                "remote_period": spec["remote_period"],
                                "adjustment": spec["adjustment"],
                                "run_mode": "quote_snapshot_fallback",
                                "fetch_window_start": requested_start_date.isoformat() if requested_start_date else None,
                                "fetch_window_end": requested_end_date.isoformat() if requested_end_date else today_local().isoformat(),
                                "fetched_rows_this_run": int(len(snapshot_daily)),
                                "cached_rows": int(len(merged_daily)),
                                "cached_start": merged_daily.index.min().strftime("%Y-%m-%d %H:%M:%S") if not merged_daily.empty else None,
                                "cached_end": merged_daily.index.max().strftime("%Y-%m-%d %H:%M:%S") if not merged_daily.empty else None,
                                "update_note": f"fallback from tencent quote after fq error: {exc}",
                                "today_bar_backfilled_from_quote": True,
                            }
                            for derived_period in ("weekly", "monthly"):
                                derived_spec = period_specs.get(derived_period)
                                if not derived_spec or str(derived_spec.get("source_type")) != "fq":
                                    continue
                                derived_summary = write_derived_period_from_daily(
                                    stock=stock,
                                    period_name=derived_period,
                                    spec=derived_spec,
                                    store_root=store_root,
                                )
                                if derived_summary:
                                    period_results[derived_period] = derived_summary
                            continue

                if period_name in {"weekly", "monthly"}:
                    derived_summary = write_derived_period_from_daily(
                        stock=stock,
                        period_name=period_name,
                        spec=spec,
                        store_root=store_root,
                    )
                    if derived_summary:
                        derived_summary["run_mode"] = "derived_from_daily_fallback"
                        derived_summary["update_note"] = f"derived from local daily after fq error: {exc}"
                        period_results[period_name] = derived_summary
                        continue

                raise

    concept_error: str | None = None
    concepts = empty_flat_frame(CONCEPT_COLUMNS)
    if not skip_concepts:
        try:
            log(f"{stock_key(stock)} syncing concepts ...")
            concepts = fetch_concepts(session=session, stock=stock)
        except Exception as exc:  # pragma: no cover - best effort metadata sync
            concept_error = str(exc)

    stock_meta = {
        "symbol": stock["symbol"],
        "market": stock["market"],
        "full_symbol": stock_key(stock),
        "name": stock["name"],
        "concept_count": int(len(concepts)),
        "synced_periods": "|".join(period_specs.keys()),
        "last_bar_sync_at": now_iso(),
        "last_concept_sync_at": now_iso() if concept_error is None else "",
        "store_root": str(store_root),
    }

    return stock_meta, concepts, {"periods": period_results, "concept_error": concept_error}


def sync_stock_isolated(
    stock: dict[str, Any],
    period_specs: dict[str, Any],
    store_root: Path,
    skip_concepts: bool,
    skip_bars: bool,
    requested_start_date: date | None,
    requested_end_date: date | None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    task_stock = dict(stock)
    session = build_session()
    baostock_client = BaoStockClient()
    try:
        stock_meta, concepts, stock_registry = sync_stock(
            session=session,
            baostock_client=baostock_client,
            stock=task_stock,
            period_specs=period_specs,
            store_root=store_root,
            skip_concepts=skip_concepts,
            skip_bars=skip_bars,
            requested_start_date=requested_start_date,
            requested_end_date=requested_end_date,
        )
        return task_stock, stock_meta, concepts, stock_registry
    finally:
        baostock_client.close()
        close_session(session)


def ensure_store_structure(store_root: Path, period_specs: dict[str, Any]) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / "metadata").mkdir(parents=True, exist_ok=True)
    for period_name in period_specs:
        (store_root / period_name).mkdir(parents=True, exist_ok=True)


def sync_market_store(options: RunOptions) -> int:
    config = load_json(options.config_path)
    stock_pool = load_stock_pool_config(config=config, config_path=options.config_path)
    store_root = resolve_path(options.config_path.parent, str(config["store_root"]))
    period_specs = dict(config["periods"])
    session = build_session()

    if stock_pool.enabled and (options.stock_pool_only or stock_pool.refresh_on_sync):
        refresh_stock_pool_file(config=config, config_path=options.config_path, session=session)

    stocks = load_stock_universe(config=config, config_path=options.config_path)

    if options.selected_periods:
        period_specs = {k: v for k, v in period_specs.items() if k in options.selected_periods}
    if options.selected_stocks:
        stocks = [stock for stock in stocks if stock_matches_selector(stock, options.selected_stocks)]

    if (
        not options.stock_pool_only
        and not options.skip_bars
        and options.workers > 1
        and any(str(spec.get("source_type")) == "minute_baostock" for spec in period_specs.values())
        and len(stocks) > 1
    ):
        raise SyncError(
            "minute_baostock 周期不支持在 sync_market_data_store.py 中多线程并发；"
            "请改用 run_minute_sync_queue.py 的 --workers 并发模式。"
        )

    if options.stock_pool_only:
        print("")
        print(f"stock_pool_file: {stock_pool.output_path}")
        print(f"stock_count: {len(stocks)}")
        return 0

    if not stocks:
        raise SyncError("没有匹配到要同步的股票，请检查 --stocks 参数或股票池配置。")
    if not period_specs and not options.skip_bars:
        raise SyncError("没有匹配到要同步的周期，请检查 --periods 参数。")

    ensure_store_structure(store_root, period_specs)

    stocks_meta_path = store_root / "metadata" / "stocks.parquet"
    concepts_path = store_root / "metadata" / "stock_concepts.parquet"
    registry_path = store_root / "metadata" / "sync_registry.json"

    existing_stock_meta = (
        read_flat_frame(stocks_meta_path, STOCK_META_COLUMNS)
        if not options.skip_store_metadata
        else empty_flat_frame(STOCK_META_COLUMNS)
    )
    existing_concepts = (
        read_flat_frame(concepts_path, CONCEPT_COLUMNS)
        if not options.skip_store_metadata
        else empty_flat_frame(CONCEPT_COLUMNS)
    )

    baostock_client = BaoStockClient()
    registry: dict[str, Any] = {
        "store_root": str(store_root),
        "synced_at": now_iso(),
        "requested_start_date": options.sync_start_date.isoformat() if options.sync_start_date else None,
        "requested_end_date": options.sync_end_date.isoformat() if options.sync_end_date else None,
        "format": "parquet",
        "bars_layout": "one parquet file per symbol under each period directory",
        "bars_columns": BAR_COLUMNS,
        "metadata_tables": {
            "stocks": str(stocks_meta_path),
            "stock_concepts": str(concepts_path),
        },
        "symbols": {},
    }

    stock_meta_updates: list[pd.DataFrame] = []
    concept_updates: list[pd.DataFrame] = []
    stock_meta_keys_to_replace: set[str] = set()
    concept_keys_to_replace: set[str] = set()
    failures: list[dict[str, str]] = []
    existing_stock_meta_by_symbol: dict[str, dict[str, Any]] = {}
    if not existing_stock_meta.empty and "full_symbol" in existing_stock_meta.columns:
        for row in existing_stock_meta.to_dict(orient="records"):
            full_symbol = str(row.get("full_symbol", "")).strip()
            if full_symbol:
                existing_stock_meta_by_symbol[full_symbol] = row

    def record_success(
        stock: dict[str, Any],
        stock_meta: dict[str, Any],
        concepts: pd.DataFrame,
        stock_registry: dict[str, Any],
    ) -> None:
        full_symbol = str(stock_meta.get("full_symbol") or stock_key(stock))
        preserved_meta = existing_stock_meta_by_symbol.get(full_symbol, {})
        should_preserve_concepts = options.skip_concepts or stock_registry["concept_error"] is not None

        if should_preserve_concepts:
            preserved_concept_count = preserved_meta.get("concept_count")
            if coerce_int(preserved_concept_count, default=-1) >= 0:
                stock_meta["concept_count"] = coerce_int(preserved_concept_count)
            if preserved_meta.get("last_concept_sync_at"):
                stock_meta["last_concept_sync_at"] = preserved_meta.get("last_concept_sync_at")

        stock_meta_keys_to_replace.add(full_symbol)
        stock_meta_updates.append(pd.DataFrame([stock_meta], columns=STOCK_META_COLUMNS))

        if not should_preserve_concepts:
            concept_keys_to_replace.add(full_symbol)
        if not should_preserve_concepts and not concepts.empty:
            concept_updates.append(concepts)
        registry["symbols"][full_symbol] = {
            "name": stock["name"],
            "periods": stock_registry["periods"],
            "concept_count": coerce_int(stock_meta.get("concept_count"), default=int(len(concepts))),
            "concept_error": stock_registry["concept_error"],
            "status": "ok",
        }

    def record_failure(stock: dict[str, Any], error_message: str) -> None:
        failures.append({"symbol": stock_key(stock), "error": error_message})
        registry["symbols"][stock_key(stock)] = {
            "name": stock.get("name") or stock_key(stock),
            "periods": {},
            "concept_count": 0,
            "concept_error": error_message,
            "status": "failed",
        }
        log(f"failed stock {stock_key(stock)}: {error_message}")

    try:
        if options.workers <= 1 or len(stocks) <= 1:
            for stock in stocks:
                log(f"start stock {stock_key(stock)}")
                try:
                    stock_meta, concepts, stock_registry = sync_stock(
                        session=session,
                        baostock_client=baostock_client,
                        stock=stock,
                        period_specs=period_specs,
                        store_root=store_root,
                        skip_concepts=options.skip_concepts,
                        skip_bars=options.skip_bars,
                        requested_start_date=options.sync_start_date,
                        requested_end_date=options.sync_end_date,
                    )
                    record_success(stock=stock, stock_meta=stock_meta, concepts=concepts, stock_registry=stock_registry)
                    log(f"done stock {stock_key(stock)}")
                except Exception as exc:
                    record_failure(stock=stock, error_message=str(exc))
        else:
            worker_count = min(options.workers, len(stocks))
            log(f"parallel stock sync enabled workers={worker_count}")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_stock = {
                    executor.submit(
                        sync_stock_isolated,
                        stock,
                        period_specs,
                        store_root,
                        options.skip_concepts,
                        options.skip_bars,
                        options.sync_start_date,
                        options.sync_end_date,
                    ): stock
                    for stock in stocks
                }
                for future in as_completed(future_to_stock):
                    stock = future_to_stock[future]
                    try:
                        resolved_stock, stock_meta, concepts, stock_registry = future.result()
                        record_success(
                            stock=resolved_stock,
                            stock_meta=stock_meta,
                            concepts=concepts,
                            stock_registry=stock_registry,
                        )
                        log(f"done stock {stock_key(resolved_stock)}")
                    except Exception as exc:
                        record_failure(stock=stock, error_message=str(exc))
    finally:
        baostock_client.close()
        close_session(session)

    registry["success_count"] = len(stocks) - len(failures)
    registry["failure_count"] = len(failures)
    registry["failures"] = failures

    if not options.skip_store_metadata:
        replacement_stock_meta = (
            pd.concat(stock_meta_updates, axis=0, ignore_index=True)
            if stock_meta_updates
            else empty_flat_frame(STOCK_META_COLUMNS)
        )
        replacement_concepts = (
            pd.concat(concept_updates, axis=0, ignore_index=True)
            if concept_updates
            else empty_flat_frame(CONCEPT_COLUMNS)
        )
        updated_stock_meta = replace_rows_by_keys(
            frame=existing_stock_meta,
            key_column="full_symbol",
            keys_to_remove=stock_meta_keys_to_replace,
            replacement=replacement_stock_meta,
            columns=STOCK_META_COLUMNS,
        )
        updated_concepts = replace_rows_by_keys(
            frame=existing_concepts,
            key_column="full_symbol",
            keys_to_remove=concept_keys_to_replace,
            replacement=replacement_concepts,
            columns=CONCEPT_COLUMNS,
        )
        write_flat_frame(stocks_meta_path, updated_stock_meta, STOCK_META_COLUMNS)
        write_flat_frame(concepts_path, updated_concepts, CONCEPT_COLUMNS)
        write_json(registry_path, registry)

    print("")
    print(f"store_root: {store_root}")
    print(f"workers: {max(1, options.workers)}")
    if options.skip_store_metadata:
        print("store_metadata_written: false")
    else:
        print(f"stocks_meta: {stocks_meta_path}")
        print(f"stock_concepts: {concepts_path}")
        print(f"sync_registry: {registry_path}")
    print(f"success_count: {len(stocks) - len(failures)}")
    print(f"failure_count: {len(failures)}")
    return 0 if len(failures) < len(stocks) else 1


def parse_args(argv: list[str]) -> RunOptions:
    parser = argparse.ArgumentParser(description="Sync local parquet market-data store.")
    parser.add_argument("config_path", nargs="?", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--periods", help="Comma-separated periods to sync, e.g. daily,30m,5m")
    parser.add_argument("--stocks", help="Comma-separated stock selectors, e.g. 600519,sh600519")
    parser.add_argument("--start-date", help="Optional inclusive sync start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", help="Optional inclusive sync end date, format YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=1, help="Number of stocks to sync in parallel")
    parser.add_argument("--skip-concepts", action="store_true", help="Skip concept metadata sync")
    parser.add_argument("--skip-bars", action="store_true", help="Skip bar data sync")
    parser.add_argument(
        "--skip-store-metadata",
        action="store_true",
        help="Skip writing global metadata tables and sync registry; useful for queue workers",
    )
    parser.add_argument(
        "--stock-pool-only",
        action="store_true",
        help="Only refresh the filtered stock pool file and exit",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config_path = Path(args.config_path).resolve()
    if not config_path.exists():
        raise SyncError(f"配置文件不存在: {config_path}")

    selected_periods = parse_csv_arg(args.periods)
    selected_stocks_raw = parse_csv_arg(args.stocks)
    selected_stocks = (
        {normalize_stock_selector(item) for item in selected_stocks_raw}
        if selected_stocks_raw
        else None
    )
    sync_start_date = parse_date(args.start_date) if args.start_date else None
    sync_end_date = parse_date(args.end_date) if args.end_date else None
    if sync_start_date and sync_end_date and sync_start_date > sync_end_date:
        parser.error("--start-date must be on or before --end-date")

    return RunOptions(
        config_path=config_path,
        selected_periods=selected_periods,
        selected_stocks=selected_stocks,
        sync_start_date=sync_start_date,
        sync_end_date=sync_end_date,
        workers=args.workers,
        skip_concepts=args.skip_concepts,
        skip_bars=args.skip_bars,
        stock_pool_only=args.stock_pool_only,
        skip_store_metadata=args.skip_store_metadata,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv if argv is not None else sys.argv[1:])
    return sync_market_store(options)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
