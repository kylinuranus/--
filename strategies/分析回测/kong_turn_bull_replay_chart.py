from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from plotly.offline import get_plotlyjs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
DEFAULT_HALF_K_INPUT = (
    SCREENING_RESULTS_DIR
    / "weekly_bullish_kong_turn_bull_half_k_entry_detail_2026-04-01_to_2026-06-30.csv"
)
DEFAULT_RED_HOLD_LOW_INPUT = (
    SCREENING_RESULTS_DIR
    / "weekly_bullish_kong_turn_bull_nextday_red_holdlow_entry_detail_2026-04-01_to_2026-06-30.csv"
)
DEFAULT_FORWARD_INPUT = (
    SCREENING_RESULTS_DIR
    / "weekly_bullish_kong_turn_bull_forward_compare_detail_2026-04-01_to_2026-06-30.csv"
)
DAILY_COLUMNS = ["symbol", "market", "name", "open", "high", "low", "close", "volume"]
LATEST_SIGNAL_PATTERN = re.compile(r"^weekly_bullish_kong_turn_bull_(\d{4}-\d{2}-\d{2})\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an annotated local HTML replay chart for weekly-bullish "
            "kong-turn-bull signals and derived entry ideas."
        )
    )
    parser.add_argument(
        "--signal-input",
        type=Path,
        default=find_latest_daily_signal_file() or DEFAULT_FORWARD_INPUT,
        help="Signal CSV. Supports latest daily output or forward_compare_detail output.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local A-share parquet root.")
    parser.add_argument(
        "--half-k-input",
        type=Path,
        default=DEFAULT_HALF_K_INPUT if DEFAULT_HALF_K_INPUT.exists() else None,
        help="Optional 50%% signal-K entry detail CSV.",
    )
    parser.add_argument(
        "--red-hold-low-input",
        type=Path,
        default=DEFAULT_RED_HOLD_LOW_INPUT if DEFAULT_RED_HOLD_LOW_INPUT.exists() else None,
        help="Optional next-day red candle holding signal low entry detail CSV.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Optional code/file_symbol to include. Can be repeated or comma-separated, e.g. 601636 or sh601636.",
    )
    parser.add_argument("--start", default=None, help="Optional chart start date, e.g. 2026-04-01.")
    parser.add_argument("--end", default=None, help="Optional chart end date, e.g. 2026-06-30.")
    parser.add_argument(
        "--days-before",
        type=int,
        default=90,
        help="If start is omitted, chart starts this many calendar days before the first event per stock.",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=12,
        help="If end is omitted, chart ends this many calendar days after the last event per stock.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=160,
        help="Safety cap when no symbol is specified. Use 0 for no cap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCREENING_RESULTS_DIR / "replay_charts",
        help="Directory for the generated HTML dashboard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit HTML output path.",
    )
    return parser.parse_args()


def find_latest_daily_signal_file() -> Path | None:
    candidates: list[tuple[str, Path]] = []
    if not SCREENING_RESULTS_DIR.exists():
        return None

    for path in SCREENING_RESULTS_DIR.glob("weekly_bullish_kong_turn_bull_*.csv"):
        match = LATEST_SIGNAL_PATTERN.match(path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def normalize_code(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6) if digits else text


def compute_return(current: float | int | None, base: float | int | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base):
        return None
    base_float = float(base)
    if abs(base_float) <= 1e-9:
        return None
    return float(current) / base_float - 1.0


def infer_market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def resolve_file_symbol(raw_symbol: Any, data_root: Path) -> str:
    text = str(raw_symbol).strip().lower()
    if re.fullmatch(r"(sh|sz|bj)\d{6}", text):
        return text

    code = normalize_code(text)
    if not re.fullmatch(r"\d{6}", code):
        return text

    daily_root = data_root / "daily"
    for prefix in ("sh", "sz", "bj"):
        candidate = f"{prefix}{code}"
        if (daily_root / f"{candidate}.parquet").exists():
            return candidate
    return f"{infer_market_prefix(code)}{code}"


def normalize_signal_df(raw_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    date_column = first_existing_column(df, ["signal_date", "signal_time", "current_time"])
    if date_column is None:
        raise ValueError("Signal CSV must contain one of: signal_date, signal_time, current_time")

    file_symbol_column = first_existing_column(df, ["file_symbol"])
    symbol_column = first_existing_column(df, ["symbol", "symbol_daily", "symbol_weekly"])
    market_column = first_existing_column(df, ["market"])
    name_column = first_existing_column(df, ["name", "name_daily", "name_weekly"])

    open_column = first_existing_column(df, ["signal_open", "open", "current_open"])
    high_column = first_existing_column(df, ["signal_high", "high", "current_high"])
    low_column = first_existing_column(df, ["signal_low", "low", "current_low"])
    close_column = first_existing_column(df, ["signal_close", "close", "current_close"])
    pattern_column = first_existing_column(df, ["kong_turn_bull_pattern"])

    required_price_columns = [open_column, high_column, low_column, close_column]
    if any(column is None for column in required_price_columns):
        raise ValueError("Signal CSV must contain signal/open/high/low/close price columns")

    result = pd.DataFrame()
    if file_symbol_column:
        result["file_symbol"] = df[file_symbol_column].astype(str).str.strip().str.lower()
    elif symbol_column:
        if market_column:
            result["file_symbol"] = (
                df[market_column].astype(str).str.strip().str.lower()
                + df[symbol_column].map(normalize_code)
            )
        else:
            result["file_symbol"] = df[symbol_column].map(lambda value: resolve_file_symbol(value, data_root))
    else:
        raise ValueError("Signal CSV must contain file_symbol or symbol")

    result["file_symbol"] = result["file_symbol"].map(lambda value: resolve_file_symbol(value, data_root))
    result["symbol"] = df[symbol_column].map(normalize_code) if symbol_column else result["file_symbol"].str[-6:]
    result["name"] = df[name_column].astype(str) if name_column else result["file_symbol"]
    result["signal_date"] = pd.to_datetime(df[date_column]).dt.normalize()
    result["signal_month"] = result["signal_date"].dt.strftime("%Y-%m")
    result["signal_open"] = pd.to_numeric(df[open_column], errors="coerce")
    result["signal_high"] = pd.to_numeric(df[high_column], errors="coerce")
    result["signal_low"] = pd.to_numeric(df[low_column], errors="coerce")
    result["signal_close"] = pd.to_numeric(df[close_column], errors="coerce")
    result["kong_turn_bull_pattern"] = df[pattern_column].fillna("").astype(str) if pattern_column else ""

    if "three_day_kong_turn_bull" in df.columns:
        result["three_day_kong_turn_bull"] = normalize_bool(df["three_day_kong_turn_bull"])
    else:
        result["three_day_kong_turn_bull"] = result["kong_turn_bull_pattern"].isin(["three_day", "both"])

    if "two_day_kong_turn_bull" in df.columns:
        result["two_day_kong_turn_bull"] = normalize_bool(df["two_day_kong_turn_bull"])
    else:
        result["two_day_kong_turn_bull"] = result["kong_turn_bull_pattern"].isin(["two_day", "both"])

    for hold_day in range(1, 6):
        ret_column = f"ret_{hold_day}d"
        exit_column = f"exit_date_{hold_day}d"
        if ret_column in df.columns:
            result[ret_column] = pd.to_numeric(df[ret_column], errors="coerce")
        if exit_column in df.columns:
            result[exit_column] = pd.to_datetime(df[exit_column], errors="coerce").dt.normalize()

    result = result.dropna(subset=["signal_date", "signal_open", "signal_high", "signal_low", "signal_close"])
    return result.sort_values(["signal_date", "file_symbol"]).reset_index(drop=True)


def normalize_half_k_df(raw_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = normalize_signal_df(raw_df, data_root)
    raw = raw_df.copy()
    df["entry_date"] = pd.to_datetime(raw["entry_date"], errors="coerce").dt.normalize()
    df["entry_price"] = pd.to_numeric(raw["entry_price"], errors="coerce")
    df["entry_open"] = pd.to_numeric(raw["entry_open"], errors="coerce")
    df["entry_high"] = pd.to_numeric(raw["entry_high"], errors="coerce")
    df["entry_low"] = pd.to_numeric(raw["entry_low"], errors="coerce")
    df["entry_close"] = pd.to_numeric(raw["entry_close"], errors="coerce")

    for hold_day in range(1, 6):
        ret_column = f"ret_{hold_day}d"
        exit_column = f"exit_date_{hold_day}d"
        if ret_column in raw.columns:
            df[ret_column] = pd.to_numeric(raw[ret_column], errors="coerce")
        if exit_column in raw.columns:
            df[exit_column] = pd.to_datetime(raw[exit_column], errors="coerce").dt.normalize()

    return df.dropna(subset=["entry_date", "entry_price"]).sort_values(["entry_date", "file_symbol"]).reset_index(drop=True)


def normalize_red_hold_low_df(raw_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = normalize_signal_df(raw_df, data_root)
    raw = raw_df.copy()
    df["entry_date"] = pd.to_datetime(raw["entry_date"], errors="coerce").dt.normalize()
    df["entry_open"] = pd.to_numeric(raw["entry_open"], errors="coerce")
    df["entry_high"] = pd.to_numeric(raw["entry_high"], errors="coerce")
    df["entry_low"] = pd.to_numeric(raw["entry_low"], errors="coerce")
    df["entry_close"] = pd.to_numeric(raw["entry_close"], errors="coerce")
    df["exit_date"] = pd.to_datetime(raw["exit_date"], errors="coerce").dt.normalize() if "exit_date" in raw else pd.NaT
    df["exit_close"] = pd.to_numeric(raw["exit_close"], errors="coerce") if "exit_close" in raw else pd.NA
    df["next_close_return"] = (
        pd.to_numeric(raw["next_close_return"], errors="coerce") if "next_close_return" in raw else pd.NA
    )
    return df.dropna(subset=["entry_date", "entry_close"]).sort_values(["entry_date", "file_symbol"]).reset_index(drop=True)


def parse_symbol_filters(values: list[str], data_root: Path) -> list[str]:
    symbols: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                symbols.append(resolve_file_symbol(part, data_root))
    return list(dict.fromkeys(symbols))


def choose_symbols(signal_df: pd.DataFrame, requested: list[str], max_symbols: int) -> list[str]:
    if requested:
        return requested

    grouped = (
        signal_df.groupby("file_symbol", as_index=False)
        .agg(signal_count=("signal_date", "size"), latest_signal=("signal_date", "max"), name=("name", "last"))
        .sort_values(["latest_signal", "signal_count", "file_symbol"], ascending=[False, False, True])
    )
    symbols = grouped["file_symbol"].astype(str).tolist()
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    return symbols


def filter_entries_to_signals(entry_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    if entry_df.empty or signal_df.empty:
        return entry_df

    signal_keys = set(
        zip(
            signal_df["file_symbol"].astype(str),
            pd.to_datetime(signal_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        )
    )
    entry_keys = list(
        zip(
            entry_df["file_symbol"].astype(str),
            pd.to_datetime(entry_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        )
    )
    return entry_df.loc[[key in signal_keys for key in entry_keys]].copy()


def find_daily_position(daily_df: pd.DataFrame, signal_date: pd.Timestamp) -> int | None:
    dates = pd.to_datetime(daily_df.index).normalize()
    matches = [idx for idx, value in enumerate(dates) if pd.Timestamp(value).normalize() == signal_date]
    if not matches:
        return None
    return matches[0]


def signal_base_fields(signal: pd.Series) -> dict[str, Any]:
    return {
        "file_symbol": str(signal["file_symbol"]),
        "symbol": str(signal["symbol"]),
        "name": str(signal["name"]),
        "signal_date": pd.Timestamp(signal["signal_date"]).normalize(),
        "signal_month": str(signal["signal_month"]),
        "kong_turn_bull_pattern": str(signal.get("kong_turn_bull_pattern", "")),
        "three_day_kong_turn_bull": bool(signal.get("three_day_kong_turn_bull", False)),
        "two_day_kong_turn_bull": bool(signal.get("two_day_kong_turn_bull", False)),
        "signal_open": float(signal["signal_open"]),
        "signal_high": float(signal["signal_high"]),
        "signal_low": float(signal["signal_low"]),
        "signal_close": float(signal["signal_close"]),
    }


def derive_half_k_entries(signal_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    daily_cache: dict[str, pd.DataFrame] = {}
    for _, signal in signal_df.iterrows():
        file_symbol = str(signal["file_symbol"])
        daily_df = daily_cache.get(file_symbol)
        if daily_df is None:
            daily_df = load_daily_bars(file_symbol, data_root)
            daily_cache[file_symbol] = daily_df
        if daily_df.empty:
            continue

        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        signal_idx = find_daily_position(daily_df, signal_date)
        if signal_idx is None:
            continue

        entry_idx = signal_idx + 1
        if entry_idx >= len(daily_df):
            continue

        entry_bar = daily_df.iloc[entry_idx]
        entry_price = float(signal["signal_low"]) + (float(signal["signal_high"]) - float(signal["signal_low"])) * 0.5
        if not (float(entry_bar["low"]) <= entry_price <= float(entry_bar["high"])):
            continue

        daily_dates = pd.to_datetime(daily_df.index).normalize()
        row = {
            **signal_base_fields(signal),
            "entry_date": pd.Timestamp(daily_dates[entry_idx]).normalize(),
            "entry_price": float(entry_price),
            "entry_open": float(entry_bar["open"]),
            "entry_high": float(entry_bar["high"]),
            "entry_low": float(entry_bar["low"]),
            "entry_close": float(entry_bar["close"]),
        }
        for hold_day in range(1, 6):
            exit_idx = signal_idx + hold_day
            if exit_idx >= len(daily_df):
                row[f"exit_date_{hold_day}d"] = pd.NaT
                row[f"ret_{hold_day}d"] = pd.NA
                continue
            exit_bar = daily_df.iloc[exit_idx]
            row[f"exit_date_{hold_day}d"] = pd.Timestamp(daily_dates[exit_idx]).normalize()
            row[f"ret_{hold_day}d"] = compute_return(float(exit_bar["close"]), entry_price)
        rows.append(row)

    return pd.DataFrame(rows)


def derive_red_hold_low_entries(signal_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    daily_cache: dict[str, pd.DataFrame] = {}
    for _, signal in signal_df.iterrows():
        file_symbol = str(signal["file_symbol"])
        daily_df = daily_cache.get(file_symbol)
        if daily_df is None:
            daily_df = load_daily_bars(file_symbol, data_root)
            daily_cache[file_symbol] = daily_df
        if daily_df.empty:
            continue

        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        signal_idx = find_daily_position(daily_df, signal_date)
        if signal_idx is None:
            continue

        entry_idx = signal_idx + 1
        if entry_idx >= len(daily_df):
            continue

        daily_dates = pd.to_datetime(daily_df.index).normalize()
        entry_bar = daily_df.iloc[entry_idx]
        signal_low = float(signal["signal_low"])
        entry_open = float(entry_bar["open"])
        entry_high = float(entry_bar["high"])
        entry_low = float(entry_bar["low"])
        entry_close = float(entry_bar["close"])
        entry_day_red = entry_close < entry_open
        entry_day_holds_signal_low = entry_low >= signal_low
        if not (entry_day_red and entry_day_holds_signal_low):
            continue

        exit_idx = entry_idx + 1
        exit_date = pd.NaT
        exit_close: float | None = None
        next_close_return: float | None = None
        next_close_up: bool | None = None
        if exit_idx < len(daily_df):
            exit_bar = daily_df.iloc[exit_idx]
            exit_date = pd.Timestamp(daily_dates[exit_idx]).normalize()
            exit_close = float(exit_bar["close"])
            next_close_return = compute_return(exit_close, entry_close)
            next_close_up = exit_close > entry_close

        rows.append(
            {
                **signal_base_fields(signal),
                "entry_date": pd.Timestamp(daily_dates[entry_idx]).normalize(),
                "entry_open": entry_open,
                "entry_high": entry_high,
                "entry_low": entry_low,
                "entry_close": entry_close,
                "entry_day_red": True,
                "entry_day_holds_signal_low": True,
                "entry_low_vs_signal_low_ret": compute_return(entry_low, signal_low),
                "entry_close_vs_signal_close_ret": compute_return(entry_close, float(signal["signal_close"])),
                "exit_date": exit_date,
                "exit_close": exit_close,
                "next_close_return": next_close_return,
                "next_close_up": next_close_up,
            }
        )

    return pd.DataFrame(rows)


def merge_entry_frames(base_df: pd.DataFrame, derived_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return derived_df.copy()
    if derived_df.empty:
        return base_df.copy()

    combined = pd.concat([base_df, derived_df], ignore_index=True, sort=False)
    for column in ["signal_date", "entry_date"]:
        if column in combined.columns:
            combined[column] = pd.to_datetime(combined[column], errors="coerce").dt.normalize()
    return combined.drop_duplicates(["file_symbol", "signal_date", "entry_date"], keep="first").reset_index(drop=True)


def load_daily_bars(file_symbol: str, data_root: Path) -> pd.DataFrame:
    path = data_root / "daily" / f"{file_symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    available_columns = pd.read_parquet(path, columns=None).columns.tolist()
    columns = [column for column in DAILY_COLUMNS if column in available_columns]
    return pd.read_parquet(path, columns=columns).sort_index()


def pct_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def num_text(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def pattern_label(pattern: str, three_day: bool, two_day: bool) -> str:
    if three_day and two_day:
        return "双"
    if str(pattern) == "both":
        return "双"
    if two_day or str(pattern) == "two_day":
        return "2日"
    if three_day or str(pattern) == "three_day":
        return "3日"
    return str(pattern) or "信号"


def collect_event_dates(*frames: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for frame in frames:
        if frame.empty:
            continue
        for column in [
            "signal_date",
            "entry_date",
            "exit_date",
            "exit_date_1d",
            "exit_date_2d",
            "exit_date_3d",
            "exit_date_4d",
            "exit_date_5d",
        ]:
            if column not in frame.columns:
                continue
            values = pd.to_datetime(frame[column], errors="coerce").dropna()
            dates.extend(pd.Timestamp(value).normalize() for value in values)
    return dates


def event_in_range(frame: pd.DataFrame, date_columns: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(False, index=frame.index)
    for column in date_columns:
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        mask |= dates.between(start, end)
    return frame.loc[mask].copy()


def signal_hover(row: pd.Series) -> str:
    lines = [
        f"{date_text(row['signal_date'])} 信号日",
        f"形态: {row.get('kong_turn_bull_pattern', '-')}",
        f"O/H/L/C: {num_text(row['signal_open'])} / {num_text(row['signal_high'])} / "
        f"{num_text(row['signal_low'])} / {num_text(row['signal_close'])}",
    ]
    returns = []
    for hold_day in range(1, 6):
        column = f"ret_{hold_day}d"
        if column in row.index and not pd.isna(row[column]):
            returns.append(f"{hold_day}日 {pct_text(row[column])}")
    if returns:
        lines.append("信号收盘后收益: " + " | ".join(returns))
    return "<br>".join(lines)


def half_k_hover(row: pd.Series) -> str:
    lines = [
        f"{date_text(row['entry_date'])} 50%K买点",
        f"信号日: {date_text(row['signal_date'])}",
        f"买入价: {num_text(row['entry_price'])}",
        f"当日O/H/L/C: {num_text(row.get('entry_open'))} / {num_text(row.get('entry_high'))} / "
        f"{num_text(row.get('entry_low'))} / {num_text(row.get('entry_close'))}",
    ]
    returns = []
    for hold_day in range(1, 6):
        column = f"ret_{hold_day}d"
        if column in row.index and not pd.isna(row[column]):
            returns.append(f"{hold_day}日 {pct_text(row[column])}")
    if returns:
        lines.append("持仓收益: " + " | ".join(returns))
    return "<br>".join(lines)


def red_hold_low_hover(row: pd.Series) -> str:
    lines = [
        f"{date_text(row['entry_date'])} 阴线不破前低买点",
        f"信号日: {date_text(row['signal_date'])}",
        f"买入价(尾盘): {num_text(row['entry_close'])}",
        f"信号低点: {num_text(row['signal_low'])}",
        f"次日尾盘收益: {pct_text(row.get('next_close_return'))}",
    ]
    if "exit_date" in row.index and not pd.isna(row["exit_date"]):
        lines.append(f"卖出日/价: {date_text(row['exit_date'])} / {num_text(row.get('exit_close'))}")
    return "<br>".join(lines)


def build_signal_events(frame: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for _, row in frame.sort_values("signal_date").iterrows():
        three_day = bool(row.get("three_day_kong_turn_bull", False))
        two_day = bool(row.get("two_day_kong_turn_bull", False))
        pattern = str(row.get("kong_turn_bull_pattern", ""))
        signal_high = float(row["signal_high"])
        events.append(
            {
                "date": date_text(row["signal_date"]),
                "y": signal_high * 1.018,
                "price": float(row["signal_close"]),
                "label": pattern_label(pattern, three_day, two_day),
                "pattern": pattern,
                "three_day": three_day,
                "two_day": two_day,
                "hover": signal_hover(row),
            }
        )
    return events


def build_half_k_events(frame: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if frame.empty or "entry_date" not in frame.columns or "entry_price" not in frame.columns:
        return events
    for _, row in frame.sort_values("entry_date").iterrows():
        events.append(
            {
                "date": date_text(row["entry_date"]),
                "signal_date": date_text(row["signal_date"]),
                "y": float(row["entry_price"]),
                "label": "50%K",
                "hover": half_k_hover(row),
                "ret_1d": clean_json_value(row.get("ret_1d")),
                "ret_2d": clean_json_value(row.get("ret_2d")),
                "ret_3d": clean_json_value(row.get("ret_3d")),
                "ret_4d": clean_json_value(row.get("ret_4d")),
                "ret_5d": clean_json_value(row.get("ret_5d")),
            }
        )
    return events


def build_red_hold_low_events(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    if frame.empty or "entry_date" not in frame.columns or "entry_close" not in frame.columns:
        return entries, exits
    for _, row in frame.sort_values("entry_date").iterrows():
        entries.append(
            {
                "date": date_text(row["entry_date"]),
                "signal_date": date_text(row["signal_date"]),
                "y": float(row["entry_close"]),
                "label": "阴不破",
                "hover": red_hold_low_hover(row),
                "next_close_return": clean_json_value(row.get("next_close_return")),
            }
        )
        if "exit_date" in row.index and not pd.isna(row["exit_date"]) and not pd.isna(row.get("exit_close")):
            exits.append(
                {
                    "date": date_text(row["exit_date"]),
                    "y": float(row["exit_close"]),
                    "label": pct_text(row.get("next_close_return")),
                    "hover": red_hold_low_hover(row),
                }
            )
    return entries, exits


def clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): clean_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json(value) for value in obj]
    return clean_json_value(obj)


def build_symbol_payload(
    file_symbol: str,
    signal_df: pd.DataFrame,
    half_k_df: pd.DataFrame,
    red_hold_low_df: pd.DataFrame,
    data_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    signals = signal_df[signal_df["file_symbol"] == file_symbol].copy()
    half_k_entries = half_k_df[half_k_df["file_symbol"] == file_symbol].copy() if not half_k_df.empty else pd.DataFrame()
    red_entries = (
        red_hold_low_df[red_hold_low_df["file_symbol"] == file_symbol].copy()
        if not red_hold_low_df.empty
        else pd.DataFrame()
    )

    event_dates = collect_event_dates(signals, half_k_entries, red_entries)
    if not event_dates:
        return None

    start = pd.Timestamp(args.start).normalize() if args.start else min(event_dates) - pd.Timedelta(days=args.days_before)
    end = pd.Timestamp(args.end).normalize() if args.end else max(event_dates) + pd.Timedelta(days=args.days_after)

    half_k_entries = event_in_range(half_k_entries, ["signal_date", "entry_date"], start, end)
    red_entries = event_in_range(red_entries, ["signal_date", "entry_date", "exit_date"], start, end)
    signals = event_in_range(signals, ["signal_date"], start, end)

    daily_df = load_daily_bars(file_symbol, data_root)
    if daily_df.empty:
        return None

    daily_df = daily_df.loc[(pd.to_datetime(daily_df.index).normalize() >= start) & (pd.to_datetime(daily_df.index).normalize() <= end)]
    if daily_df.empty:
        return None

    date_index = pd.to_datetime(daily_df.index).normalize()
    name_series = daily_df["name"].dropna() if "name" in daily_df.columns else pd.Series(dtype=object)
    symbol_series = daily_df["symbol"].dropna() if "symbol" in daily_df.columns else pd.Series(dtype=object)
    display_name = str(name_series.iloc[-1]) if not name_series.empty else str(signals["name"].dropna().iloc[-1])
    code = normalize_code(symbol_series.iloc[-1]) if not symbol_series.empty else file_symbol[-6:]
    red_buy_events, red_exit_events = build_red_hold_low_events(red_entries)

    return {
        "file_symbol": file_symbol,
        "symbol": code,
        "name": display_name,
        "date_start": date_index[0].date().isoformat(),
        "date_end": date_index[-1].date().isoformat(),
        "dates": [value.date().isoformat() for value in date_index],
        "open": daily_df["open"].astype(float).round(4).tolist(),
        "high": daily_df["high"].astype(float).round(4).tolist(),
        "low": daily_df["low"].astype(float).round(4).tolist(),
        "close": daily_df["close"].astype(float).round(4).tolist(),
        "volume": (
            pd.to_numeric(daily_df["volume"], errors="coerce").fillna(0).astype(float).round(0).tolist()
            if "volume" in daily_df.columns
            else [0] * len(daily_df)
        ),
        "events": {
            "signals": build_signal_events(signals),
            "half_k": build_half_k_events(half_k_entries),
            "red_hold_low": red_buy_events,
            "red_hold_low_exit": red_exit_events,
        },
    }


def make_output_path(args: argparse.Namespace, symbols: list[str]) -> Path:
    if args.output:
        return args.output

    safe_stem = args.signal_input.stem
    if len(symbols) == 1:
        safe_stem = f"{safe_stem}_{symbols[0]}"
    return args.output_dir / f"{safe_stem}_replay.html"


def build_dashboard_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(clean_json(payload), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    plotly_js = get_plotlyjs()

    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>空转多策略复盘图</title>
  <style>
    :root {
      --bg: #f3efe6;
      --paper: rgba(255, 252, 244, 0.92);
      --ink: #1f2933;
      --muted: #667085;
      --line: rgba(62, 51, 38, 0.14);
      --green: #0f8b6f;
      --red: #c2412d;
      --blue: #2f80ed;
      --amber: #d97706;
      --magenta: #be185d;
      --shadow: 0 18px 60px rgba(64, 52, 35, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Microsoft YaHei", "LXGW WenKai", "Noto Serif SC", serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(15, 139, 111, 0.16), transparent 32rem),
        radial-gradient(circle at 92% 12%, rgba(217, 119, 6, 0.14), transparent 28rem),
        linear-gradient(135deg, #f9f5ea 0%, var(--bg) 46%, #e9efe7 100%);
    }

    .shell {
      width: min(1480px, calc(100vw - 36px));
      margin: 0 auto;
      padding: 24px 0 34px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 16px;
    }

    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }

    .title-card { padding: 22px 24px; }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(24px, 2.4vw, 38px);
      line-height: 1.12;
      letter-spacing: -0.04em;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }

    .meta-card {
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      padding: 13px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(31, 41, 51, 0.08);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }

    .metric strong {
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1.1fr) minmax(280px, 1.5fr) auto;
      gap: 12px;
      padding: 14px;
      margin-bottom: 16px;
      align-items: center;
    }

    input, select, button {
      width: 100%;
      border: 1px solid rgba(31, 41, 51, 0.16);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
      font: inherit;
      padding: 11px 12px;
      outline: none;
    }

    button {
      cursor: pointer;
      background: #20312e;
      color: #fff9ed;
      border: none;
      box-shadow: 0 10px 22px rgba(32, 49, 46, 0.22);
    }

    .button-row {
      display: flex;
      gap: 8px;
      min-width: 164px;
    }

    .checks {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }

    .checks label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }

    .checks input {
      width: auto;
      accent-color: var(--green);
    }

    .chart-card {
      padding: 12px 12px 4px;
      margin-bottom: 16px;
    }

    #chart {
      width: 100%;
      min-height: 720px;
    }

    .details {
      display: grid;
      grid-template-columns: 0.9fr 1.1fr;
      gap: 16px;
    }

    .panel {
      padding: 18px;
      min-height: 180px;
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }

    .legend {
      display: grid;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }

    .dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 7px;
      vertical-align: baseline;
    }

    .table-wrap {
      overflow: auto;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.52);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }

    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid rgba(31, 41, 51, 0.08);
      text-align: left;
    }

    th {
      position: sticky;
      top: 0;
      background: #f8f1e4;
      color: #475467;
      z-index: 1;
    }

    .empty {
      color: var(--muted);
      padding: 18px;
    }

    @media (max-width: 960px) {
      .hero, .details, .toolbar {
        grid-template-columns: 1fr;
      }

      #chart {
        min-height: 560px;
      }

      .button-row {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card title-card">
        <h1>空转多策略复盘图</h1>
        <p class="subtitle">
          用本地日 K 数据直接标记策略信号和买点。你可以按股票切换，看信号日、50%K 回踩买点、
          “次日收阴但不破前低”尾盘买点，以及对应的收益提示。
        </p>
      </div>
      <div class="card meta-card" id="metaCards"></div>
    </section>

    <section class="card toolbar">
      <input id="filterInput" placeholder="搜索代码或名称，例如 601636 / 旗滨" />
      <select id="symbolSelect"></select>
      <div class="button-row">
        <button id="prevBtn" type="button">上一只</button>
        <button id="nextBtn" type="button">下一只</button>
      </div>
      <div class="checks">
        <label><input id="showSignal" type="checkbox" checked />信号日</label>
        <label><input id="showHalfK" type="checkbox" checked />50%K买</label>
        <label><input id="showRed" type="checkbox" checked />阴不破买</label>
        <label><input id="showExit" type="checkbox" checked />卖点</label>
        <label><input id="showText" type="checkbox" checked />文字</label>
        <label><input id="showVolume" type="checkbox" checked />成交量</label>
      </div>
    </section>

    <section class="card chart-card">
      <div id="chart"></div>
    </section>

    <section class="details">
      <div class="card panel">
        <h2>标记说明</h2>
        <div class="legend">
          <div><span class="dot" style="background: var(--blue)"></span>蓝色三角：3日空转多信号。</div>
          <div><span class="dot" style="background: var(--amber)"></span>橙色三角：2日空转多信号。</div>
          <div><span class="dot" style="background: var(--green)"></span>绿色菱形：次日触及前一根信号 K 的 50% 位置买入。</div>
          <div><span class="dot" style="background: var(--red)"></span>红色圆点：次日收阴但不破信号 K 低点，尾盘买入。</div>
          <div><span class="dot" style="background: #64748b"></span>灰色叉号：阴不破方案的隔日尾盘卖点。</div>
        </div>
      </div>
      <div class="card panel">
        <h2 id="tableTitle">当前股票事件</h2>
        <div id="eventTable" class="table-wrap"></div>
      </div>
    </section>
  </div>

  <script>__PLOTLY_JS__</script>
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const allSymbols = payload.symbol_order;
    const symbolData = payload.symbols;
    const state = { visibleSymbols: [...allSymbols], current: allSymbols[0] };

    const el = (id) => document.getElementById(id);
    const checks = ["showSignal", "showHalfK", "showRed", "showExit", "showText", "showVolume"];

    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${(Number(value) * 100).toFixed(2)}%`;
    }

    function money(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(2);
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function signalColor(event) {
      if (event.three_day && event.two_day) return "#be185d";
      if (event.two_day || event.pattern === "two_day") return "#d97706";
      return "#2f80ed";
    }

    function populateMeta() {
      const meta = payload.meta;
      el("metaCards").innerHTML = `
        <div class="metric"><span>股票数</span><strong>${allSymbols.length}</strong></div>
        <div class="metric"><span>信号数</span><strong>${meta.signal_count}</strong></div>
        <div class="metric"><span>50%K买点</span><strong>${meta.half_k_count}</strong></div>
        <div class="metric"><span>阴不破买点</span><strong>${meta.red_hold_low_count}</strong></div>
      `;
    }

    function populateSelect() {
      const filter = el("filterInput").value.trim().toLowerCase();
      state.visibleSymbols = allSymbols.filter((fileSymbol) => {
        const item = symbolData[fileSymbol];
        const haystack = `${fileSymbol} ${item.symbol} ${item.name}`.toLowerCase();
        return haystack.includes(filter);
      });

      if (!state.visibleSymbols.includes(state.current)) {
        state.current = state.visibleSymbols[0] || allSymbols[0];
      }

      el("symbolSelect").innerHTML = state.visibleSymbols
        .map((fileSymbol) => {
          const item = symbolData[fileSymbol];
          const signals = item.events.signals.length;
          const half = item.events.half_k.length;
          const red = item.events.red_hold_low.length;
          const label = `${fileSymbol} ${item.name}  信号:${signals}  50%K:${half}  阴不破:${red}`;
          return `<option value="${escapeHtml(fileSymbol)}"${fileSymbol === state.current ? " selected" : ""}>${escapeHtml(label)}</option>`;
        })
        .join("");
    }

    function markerMode() {
      return el("showText").checked ? "markers+text" : "markers";
    }

    function scatterTrace(name, events, color, symbol, size, textPosition) {
      return {
        type: "scatter",
        mode: markerMode(),
        name,
        x: events.map((event) => event.date),
        y: events.map((event) => event.y),
        text: events.map((event) => event.label),
        textposition: textPosition,
        textfont: { size: 11, color },
        hoverinfo: "text",
        hovertext: events.map((event) => event.hover),
        marker: { color, symbol, size, line: { color: "#fffaf0", width: 1.5 } },
        yaxis: "y",
      };
    }

    function buildTraces(item) {
      const traces = [
        {
          type: "candlestick",
          name: "日K",
          x: item.dates,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          increasing: { line: { color: "#d33f2f", width: 1.2 }, fillcolor: "rgba(211,63,47,0.18)" },
          decreasing: { line: { color: "#0f8b6f", width: 1.2 }, fillcolor: "rgba(15,139,111,0.18)" },
          hoverlabel: { bgcolor: "#fffaf0" },
          yaxis: "y",
        },
      ];

      if (el("showVolume").checked) {
        traces.push({
          type: "bar",
          name: "成交量",
          x: item.dates,
          y: item.volume,
          marker: { color: "rgba(100,116,139,0.26)" },
          hovertemplate: "日期 %{x}<br>成交量 %{y:,}<extra></extra>",
          yaxis: "y2",
        });
      }

      if (el("showSignal").checked && item.events.signals.length) {
        const events = item.events.signals;
        traces.push({
          type: "scatter",
          mode: markerMode(),
          name: "空转多信号",
          x: events.map((event) => event.date),
          y: events.map((event) => event.y),
          text: events.map((event) => event.label),
          textposition: "top center",
          textfont: { size: 11, color: "#334155" },
          hoverinfo: "text",
          hovertext: events.map((event) => event.hover),
          marker: {
            color: events.map(signalColor),
            symbol: "triangle-up",
            size: 13,
            line: { color: "#fffaf0", width: 1.5 },
          },
          yaxis: "y",
        });
      }

      if (el("showHalfK").checked && item.events.half_k.length) {
        traces.push(scatterTrace("50%K买点", item.events.half_k, "#0f8b6f", "diamond", 12, "bottom center"));
      }

      if (el("showRed").checked && item.events.red_hold_low.length) {
        traces.push(scatterTrace("阴不破买点", item.events.red_hold_low, "#c2412d", "circle", 12, "top center"));
      }

      if (el("showExit").checked && item.events.red_hold_low_exit.length) {
        traces.push(scatterTrace("隔日尾盘卖点", item.events.red_hold_low_exit, "#64748b", "x", 11, "bottom center"));
      }

      return traces;
    }

    function buildLayout(item) {
      return {
        title: {
          text: `${item.file_symbol} ${item.name} | ${item.date_start} 至 ${item.date_end}`,
          x: 0.02,
          xanchor: "left",
          font: { size: 20, color: "#1f2933" },
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.35)",
        margin: { l: 56, r: 24, t: 58, b: 42 },
        hovermode: "x unified",
        showlegend: true,
        legend: { orientation: "h", y: 1.05, x: 0.36, bgcolor: "rgba(255,255,255,0.5)" },
        xaxis: {
          type: "category",
          rangeslider: { visible: false },
          showgrid: false,
          tickangle: -35,
          tickfont: { size: 11 },
        },
        yaxis: {
          title: "价格",
          domain: [0.22, 1],
          gridcolor: "rgba(31,41,51,0.08)",
          zeroline: false,
        },
        yaxis2: {
          title: "成交量",
          domain: [0, 0.16],
          showgrid: false,
          zeroline: false,
          showticklabels: false,
        },
      };
    }

    function eventRows(item) {
      const rows = [];
      item.events.signals.forEach((event) => {
        rows.push({
          type: "信号",
          date: event.date,
          price: event.price,
          label: event.label,
          ret: "",
        });
      });
      item.events.half_k.forEach((event) => {
        rows.push({
          type: "50%K买",
          date: event.date,
          price: event.y,
          label: `信号:${event.signal_date}`,
          ret: `1日 ${pct(event.ret_1d)} / 3日 ${pct(event.ret_3d)} / 5日 ${pct(event.ret_5d)}`,
        });
      });
      item.events.red_hold_low.forEach((event) => {
        rows.push({
          type: "阴不破买",
          date: event.date,
          price: event.y,
          label: `信号:${event.signal_date}`,
          ret: `隔日 ${pct(event.next_close_return)}`,
        });
      });
      return rows.sort((a, b) => String(a.date).localeCompare(String(b.date)));
    }

    function renderTable(item) {
      el("tableTitle").textContent = `${item.file_symbol} ${item.name} 事件`;
      const rows = eventRows(item);
      if (!rows.length) {
        el("eventTable").innerHTML = `<div class="empty">当前区间没有可显示事件。</div>`;
        return;
      }
      el("eventTable").innerHTML = `
        <table>
          <thead><tr><th>日期</th><th>类型</th><th>价格</th><th>说明</th><th>收益</th></tr></thead>
          <tbody>
            ${rows.map((row) => `
              <tr>
                <td>${escapeHtml(row.date)}</td>
                <td>${escapeHtml(row.type)}</td>
                <td>${escapeHtml(money(row.price))}</td>
                <td>${escapeHtml(row.label)}</td>
                <td>${escapeHtml(row.ret)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function render() {
      const item = symbolData[state.current];
      if (!item) return;
      el("symbolSelect").value = state.current;
      Plotly.react("chart", buildTraces(item), buildLayout(item), { responsive: true, displaylogo: false });
      renderTable(item);
    }

    function move(step) {
      const list = state.visibleSymbols.length ? state.visibleSymbols : allSymbols;
      const currentIndex = Math.max(0, list.indexOf(state.current));
      const nextIndex = (currentIndex + step + list.length) % list.length;
      state.current = list[nextIndex];
      populateSelect();
      render();
    }

    el("filterInput").addEventListener("input", () => {
      populateSelect();
      render();
    });
    el("symbolSelect").addEventListener("change", (event) => {
      state.current = event.target.value;
      render();
    });
    el("prevBtn").addEventListener("click", () => move(-1));
    el("nextBtn").addEventListener("click", () => move(1));
    checks.forEach((id) => el(id).addEventListener("change", render));

    document.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "SELECT"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    });

    populateMeta();
    populateSelect();
    render();
  </script>
</body>
</html>
"""

    return template.replace("__PLOTLY_JS__", plotly_js).replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    signal_raw = read_csv(args.signal_input)
    if signal_raw.empty:
        raise FileNotFoundError(f"No signal rows found in {args.signal_input}")

    signal_df = normalize_signal_df(signal_raw, args.data_root)
    half_k_df = normalize_half_k_df(read_csv(args.half_k_input), args.data_root) if args.half_k_input else pd.DataFrame()
    red_hold_low_df = (
        normalize_red_hold_low_df(read_csv(args.red_hold_low_input), args.data_root)
        if args.red_hold_low_input
        else pd.DataFrame()
    )
    half_k_df = merge_entry_frames(
        filter_entries_to_signals(half_k_df, signal_df),
        derive_half_k_entries(signal_df, args.data_root),
    )
    red_hold_low_df = merge_entry_frames(
        filter_entries_to_signals(red_hold_low_df, signal_df),
        derive_red_hold_low_entries(signal_df, args.data_root),
    )

    requested_symbols = parse_symbol_filters(args.symbol, args.data_root)
    selected_symbols = choose_symbols(signal_df, requested_symbols, args.max_symbols)
    signal_df = signal_df[signal_df["file_symbol"].isin(selected_symbols)].copy()

    if not signal_df.empty:
        selected_symbols = choose_symbols(signal_df, requested_symbols, args.max_symbols)
    if half_k_df.empty:
        half_k_df = pd.DataFrame(columns=["file_symbol"])
    if red_hold_low_df.empty:
        red_hold_low_df = pd.DataFrame(columns=["file_symbol"])

    symbols_payload: dict[str, Any] = {}
    missing_symbols: list[str] = []
    for file_symbol in selected_symbols:
        payload = build_symbol_payload(
            file_symbol=file_symbol,
            signal_df=signal_df,
            half_k_df=half_k_df,
            red_hold_low_df=red_hold_low_df,
            data_root=args.data_root,
            args=args,
        )
        if payload is None:
            missing_symbols.append(file_symbol)
            continue
        symbols_payload[file_symbol] = payload

    if not symbols_payload:
        raise RuntimeError("No chart payload could be built. Check symbol filters and local daily parquet files.")

    symbol_order = [symbol for symbol in selected_symbols if symbol in symbols_payload]
    payload = {
        "meta": {
            "signal_input": str(args.signal_input),
            "half_k_input": str(args.half_k_input) if args.half_k_input else "",
            "red_hold_low_input": str(args.red_hold_low_input) if args.red_hold_low_input else "",
            "data_root": str(args.data_root),
            "signal_count": int(len(signal_df)),
            "half_k_count": int(len(half_k_df[half_k_df["file_symbol"].isin(symbol_order)])) if "file_symbol" in half_k_df else 0,
            "red_hold_low_count": int(len(red_hold_low_df[red_hold_low_df["file_symbol"].isin(symbol_order)]))
            if "file_symbol" in red_hold_low_df
            else 0,
            "missing_symbols": missing_symbols,
        },
        "symbol_order": symbol_order,
        "symbols": symbols_payload,
    }

    output_path = make_output_path(args, symbol_order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_dashboard_html(payload), encoding="utf-8")

    print(f"output={output_path}")
    print(f"symbols={len(symbol_order)}")
    print(f"signals={len(signal_df)}")
    print(f"half_k_entries={payload['meta']['half_k_count']}")
    print(f"red_hold_low_entries={payload['meta']['red_hold_low_count']}")
    if missing_symbols:
        print(f"missing_symbols={','.join(missing_symbols[:20])}")


if __name__ == "__main__":
    main()
