from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.选股.默认选股 import run_default_selection


DATA_ROOT = Path(r"D:\a_stock_data")
HORIZONS = tuple(range(1, 6))
DAILY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
SIGNAL_COLUMNS = [
    "file_symbol",
    "symbol",
    "name",
    "name_weekly",
    "name_daily",
    "kong_turn_bull_pattern",
    "weekly_time",
    "weekly_open",
    "weekly_high",
    "weekly_low",
    "weekly_close",
    "signal_time",
    "open",
    "high",
    "low",
    "close",
    "current_open",
    "current_high",
    "current_low",
    "current_close",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按月或按日期区间逐交易日运行默认选股，并统计每只股票未来1至5个交易日相对信号日收盘价的"
            "收盘价、最高价、最低价涨跌幅。"
        )
    )
    parser.add_argument(
        "--month",
        help="月份，格式 YYYY-MM，例如 2026-08。与 --start-date/--end-date 三者选一。",
    )
    parser.add_argument(
        "--start-date",
        help="分析起始日期，格式 YYYY-MM-DD。需要与 --end-date 一起使用。",
    )
    parser.add_argument(
        "--end-date",
        help="分析结束日期，格式 YYYY-MM-DD。需要与 --start-date 一起使用。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="本地行情数据库根目录，默认 D:\\a_stock_data。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCREENING_RESULTS_DIR,
        help="报告和CSV输出目录。",
    )
    return parser.parse_args()


def parse_month(month_text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        month_start = pd.Timestamp(f"{month_text}-01").normalize()
    except Exception as exc:
        raise ValueError(f"月份格式必须是 YYYY-MM，收到: {month_text}") from exc

    if month_start.strftime("%Y-%m") != month_text:
        raise ValueError(f"月份格式必须是 YYYY-MM，收到: {month_text}")

    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    return month_start, month_end


def parse_date(date_text: str, argument_name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(date_text).normalize()
    except Exception as exc:
        raise ValueError(
            f"{argument_name} 格式必须是 YYYY-MM-DD，收到: {date_text}"
        ) from exc

    if parsed.strftime("%Y-%m-%d") != date_text:
        raise ValueError(
            f"{argument_name} 格式必须是 YYYY-MM-DD，收到: {date_text}"
        )
    return parsed


def resolve_period(args: argparse.Namespace) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
    str,
    str,
    str,
]:
    has_month = bool(args.month)
    has_range = bool(args.start_date or args.end_date)

    if has_month and has_range:
        raise ValueError("--month 不能与 --start-date/--end-date 同时使用。")

    if has_month:
        period_start, period_end = parse_month(args.month)
        suffix = args.month.replace("-", "")
        signal_dir_label = args.month
        period_text = args.month
        return period_start, period_end, suffix, signal_dir_label, period_text

    if not args.start_date or not args.end_date:
        raise ValueError(
            "请提供 --month，或同时提供 --start-date 和 --end-date。"
        )

    period_start = parse_date(args.start_date, "--start-date")
    period_end = parse_date(args.end_date, "--end-date")
    if period_end < period_start:
        raise ValueError("--end-date 不能早于 --start-date。")

    suffix = f"{period_start:%Y%m%d}_{period_end:%Y%m%d}"
    signal_dir_label = f"{period_start:%Y-%m-%d}_{period_end:%Y-%m-%d}"
    period_text = f"{period_start:%Y-%m-%d} 至 {period_end:%Y-%m-%d}"
    return period_start, period_end, suffix, signal_dir_label, period_text


def build_reference_time(signal_date: pd.Timestamp) -> pd.Timestamp:
    return signal_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)


def normalize_index(index: Iterable[object]) -> pd.DatetimeIndex:
    normalized = pd.to_datetime(index)
    if getattr(normalized, "tz", None) is not None:
        normalized = normalized.tz_localize(None)
    return normalized.normalize()


def normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    normalized = frame.copy()
    normalized.index = normalize_index(normalized.index)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()


def discover_trading_dates(
    data_root: Path,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    daily_root = data_root / "daily"
    if not daily_root.exists():
        raise FileNotFoundError(f"日线目录不存在: {daily_root}")

    dates: set[pd.Timestamp] = set()
    for parquet_path in sorted(daily_root.glob("*.parquet")):
        frame = pd.read_parquet(parquet_path, columns=["close"])
        if frame.empty:
            continue

        for trade_date in normalize_index(frame.index).unique():
            timestamp = pd.Timestamp(trade_date)
            if range_start <= timestamp <= range_end:
                dates.add(timestamp)

    return sorted(dates)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compute_return(current: object, base: object) -> float | None:
    current_value = as_float(current)
    base_value = as_float(base)
    if current_value is None or base_value is None or abs(base_value) <= 1e-12:
        return None
    return current_value / base_value - 1.0


def safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def safe_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def safe_min(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def safe_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def positive_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0).mean()) if not values.empty else None


def format_pct(value: object, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.{digits}f}%"


def format_number(value: object, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def format_count(value: object) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return str(int(number))


def pick_first(row: pd.Series, candidates: Iterable[str], default: object = pd.NA) -> object:
    for column in candidates:
        if column not in row.index:
            continue
        value = row[column]
        if as_float(value) is not None:
            return value
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return default


def pick_symbol(row: pd.Series) -> str:
    value = pick_first(row, ["symbol", "symbol_daily", "symbol_weekly"], default="")
    text = str(value).strip()
    if text and text.lower() != "nan":
        return text

    file_symbol = str(row.get("file_symbol", "")).strip()
    return file_symbol[-6:]


def pick_name(row: pd.Series) -> str:
    value = pick_first(row, ["name", "name_daily", "name_weekly"], default="")
    text = str(value).strip()
    if text and text.lower() != "nan":
        return text
    return str(row.get("file_symbol", "")).strip()


def load_daily_frame(
    file_symbol: str,
    data_root: Path,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    if file_symbol in cache:
        return cache[file_symbol]

    daily_path = data_root / "daily" / f"{file_symbol}.parquet"
    if not daily_path.exists():
        cache[file_symbol] = pd.DataFrame()
        return None

    frame = pd.read_parquet(daily_path, columns=DAILY_COLUMNS)
    normalized = normalize_daily_frame(frame)
    cache[file_symbol] = normalized
    return normalized


def get_daily_bar(frame: pd.DataFrame | None, trade_date: pd.Timestamp) -> pd.Series | None:
    if frame is None or frame.empty or trade_date not in frame.index:
        return None
    return frame.loc[trade_date]


def build_future_date_map(
    trading_dates: list[pd.Timestamp],
    signal_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[int, pd.Timestamp | None]]:
    index_by_date = {date: index for index, date in enumerate(trading_dates)}
    result: dict[pd.Timestamp, dict[int, pd.Timestamp | None]] = {}

    for signal_date in signal_dates:
        signal_index = index_by_date.get(signal_date)
        if signal_index is None:
            result[signal_date] = {horizon: None for horizon in HORIZONS}
            continue

        result[signal_date] = {
            horizon: (
                trading_dates[signal_index + horizon]
                if signal_index + horizon < len(trading_dates)
                else None
            )
            for horizon in HORIZONS
        }

    return result


def build_detail_row(
    signal_row: pd.Series,
    signal_date: pd.Timestamp,
    future_dates: dict[int, pd.Timestamp | None],
    data_root: Path,
    daily_cache: dict[str, pd.DataFrame],
) -> dict[str, object]:
    file_symbol = str(signal_row.get("file_symbol", "")).strip()
    daily_frame = load_daily_frame(file_symbol=file_symbol, data_root=data_root, cache=daily_cache)
    signal_bar = get_daily_bar(daily_frame, signal_date)

    signal_close = as_float(pick_first(signal_row, ["close", "current_close"]))
    if signal_close is None and signal_bar is not None:
        signal_close = as_float(signal_bar.get("close"))
    signal_open = as_float(pick_first(signal_row, ["open", "current_open"]))
    prev1_low = as_float(pick_first(signal_row, ["prev1_low"]))
    prev2_low = as_float(pick_first(signal_row, ["prev2_low"]))
    pattern_type = str(pick_first(signal_row, ["kong_turn_bull_pattern"], default="")).strip()
    prev1_low_gap_ret = compute_return(signal_close, prev1_low)
    prev2_low_gap_ret = compute_return(signal_close, prev2_low)
    if pattern_type == "two_day":
        reversal_low_ref = "前1根"
        reversal_low_gap_ret = prev1_low_gap_ret
    elif pattern_type in {"three_day", "both"}:
        reversal_low_ref = "前2根"
        reversal_low_gap_ret = prev2_low_gap_ret
    else:
        reversal_low_ref = pd.NA
        reversal_low_gap_ret = pd.NA

    result: dict[str, object] = {
        "signal_date": signal_date.date().isoformat(),
        "file_symbol": file_symbol,
        "symbol": pick_symbol(signal_row),
        "name": pick_name(signal_row),
        "signal_close": signal_close,
        "signal_open": signal_open,
        "signal_high": as_float(pick_first(signal_row, ["high", "current_high"])),
        "signal_low": as_float(pick_first(signal_row, ["low", "current_low"])),
        "kong_turn_bull_pattern": pattern_type,
        "signal_candle_ret": compute_return(signal_close, signal_open),
        "prev1_low": prev1_low,
        "prev2_low": prev2_low,
        "prev1_low_gap_ret": prev1_low_gap_ret,
        "prev2_low_gap_ret": prev2_low_gap_ret,
        "reversal_low_ref": reversal_low_ref,
        "reversal_low_gap_ret": reversal_low_gap_ret,
        "weekly_time": pick_first(signal_row, ["weekly_time"]),
        "weekly_open": as_float(pick_first(signal_row, ["weekly_open"])),
        "weekly_high": as_float(pick_first(signal_row, ["weekly_high"])),
        "weekly_low": as_float(pick_first(signal_row, ["weekly_low"])),
        "weekly_close": as_float(pick_first(signal_row, ["weekly_close"])),
        "signal_bar_available": signal_bar is not None,
    }

    if signal_close is None:
        for horizon in HORIZONS:
            result[f"day{horizon}_date"] = (
                future_dates[horizon].date().isoformat()
                if future_dates[horizon] is not None
                else pd.NA
            )
            result[f"day{horizon}_status"] = "missing_signal_close"
            for field in ("close", "high", "low", "close_ret", "high_ret", "low_ret"):
                result[f"day{horizon}_{field}"] = pd.NA
        return result

    for horizon in HORIZONS:
        future_date = future_dates[horizon]
        result[f"day{horizon}_date"] = (
            future_date.date().isoformat() if future_date is not None else pd.NA
        )

        if future_date is None:
            result[f"day{horizon}_status"] = "future_date_unavailable"
            for field in ("close", "high", "low", "close_ret", "high_ret", "low_ret"):
                result[f"day{horizon}_{field}"] = pd.NA
            continue

        future_bar = get_daily_bar(daily_frame, future_date)
        if future_bar is None:
            result[f"day{horizon}_status"] = "stock_bar_unavailable"
            for field in ("close", "high", "low", "close_ret", "high_ret", "low_ret"):
                result[f"day{horizon}_{field}"] = pd.NA
            continue

        future_close = as_float(future_bar.get("close"))
        future_high = as_float(future_bar.get("high"))
        future_low = as_float(future_bar.get("low"))
        result[f"day{horizon}_status"] = "ok"
        result[f"day{horizon}_close"] = future_close
        result[f"day{horizon}_high"] = future_high
        result[f"day{horizon}_low"] = future_low
        result[f"day{horizon}_close_ret"] = compute_return(future_close, signal_close)
        result[f"day{horizon}_high_ret"] = compute_return(future_high, signal_close)
        result[f"day{horizon}_low_ret"] = compute_return(future_low, signal_close)

    return result


def summarize_horizon(detail_df: pd.DataFrame, horizon: int) -> dict[str, object]:
    close_column = f"day{horizon}_close_ret"
    high_column = f"day{horizon}_high_ret"
    low_column = f"day{horizon}_low_ret"
    status_column = f"day{horizon}_status"

    close_values = (
        pd.to_numeric(detail_df[close_column], errors="coerce")
        if close_column in detail_df.columns
        else pd.Series(dtype=float)
    )
    high_values = (
        pd.to_numeric(detail_df[high_column], errors="coerce")
        if high_column in detail_df.columns
        else pd.Series(dtype=float)
    )
    low_values = (
        pd.to_numeric(detail_df[low_column], errors="coerce")
        if low_column in detail_df.columns
        else pd.Series(dtype=float)
    )
    status_values = (
        detail_df[status_column]
        if status_column in detail_df.columns
        else pd.Series(dtype="object")
    )
    valid_count = int(close_values.notna().sum())

    return {
        "horizon": horizon,
        "sample_count": int(len(detail_df)),
        "valid_count": valid_count,
        "missing_count": int(len(detail_df) - valid_count),
        "status_future_date_unavailable": int((status_values == "future_date_unavailable").sum()),
        "status_stock_bar_unavailable": int((status_values == "stock_bar_unavailable").sum()),
        "close_avg_ret": safe_mean(close_values),
        "close_median_ret": safe_median(close_values),
        "close_win_rate": positive_rate(close_values),
        "close_avg_gain": safe_mean(close_values[close_values > 0]),
        "close_avg_loss": safe_mean(close_values[close_values <= 0]),
        "close_best_ret": safe_max(close_values),
        "close_worst_ret": safe_min(close_values),
        "high_avg_ret": safe_mean(high_values),
        "high_median_ret": safe_median(high_values),
        "high_above_signal_close_rate": positive_rate(high_values),
        "high_best_ret": safe_max(high_values),
        "high_worst_ret": safe_min(high_values),
        "low_avg_ret": safe_mean(low_values),
        "low_median_ret": safe_median(low_values),
        "low_above_signal_close_rate": positive_rate(low_values),
        "low_best_ret": safe_max(low_values),
        "low_worst_ret": safe_min(low_values),
    }


def build_daily_summary_row(
    signal_date: pd.Timestamp,
    selection_summary: dict[str, object],
    signal_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {
        "signal_date": signal_date.date().isoformat(),
        "weekly_pool_count": selection_summary.get("weekly_pool_count"),
        "kong_turn_bull_count": selection_summary.get("kong_turn_bull_count"),
        "monthly_bearish_excluded_count": selection_summary.get("monthly_bearish_excluded_count"),
        "post_monthly_count": selection_summary.get("post_monthly_count"),
        "abnormal_move_included_count": selection_summary.get("abnormal_move_included_count"),
        "final_count": selection_summary.get("final_count", len(signal_df)),
        "detail_row_count": int(len(detail_df)),
    }

    for horizon in HORIZONS:
        stats = summarize_horizon(detail_df, horizon)
        prefix = f"day{horizon}_"
        for key, value in stats.items():
            if key in {"horizon", "sample_count"}:
                continue
            result[f"{prefix}{key}"] = value

    return result


def build_stock_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (file_symbol, symbol, name), group in detail_df.groupby(
        ["file_symbol", "symbol", "name"],
        dropna=False,
        sort=True,
    ):
        row: dict[str, object] = {
            "file_symbol": file_symbol,
            "symbol": symbol,
            "name": name,
            "signal_count": int(len(group)),
            "first_signal_date": str(group["signal_date"].min()),
            "last_signal_date": str(group["signal_date"].max()),
        }

        for horizon in HORIZONS:
            for metric in ("close_ret", "high_ret", "low_ret"):
                column = f"day{horizon}_{metric}"
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"day{horizon}_{metric}_avg"] = safe_mean(values)
                row[f"day{horizon}_{metric}_median"] = safe_median(values)
                row[f"day{horizon}_{metric}_valid_count"] = int(values.notna().sum())
                if metric == "close_ret":
                    row[f"day{horizon}_close_win_rate"] = positive_rate(values)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["signal_count", "file_symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_wide_detail_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        columns = [
            "信号日",
            "代码",
            "名称",
            "反包类型",
            "信号日K线涨幅",
            "反包参考低点",
            "反包基准低点涨幅",
            "1D收盘涨跌幅",
            "1D最高涨幅",
            "1D最低涨幅",
            "2D收盘涨跌幅",
            "2D最高涨幅",
            "2D最低涨幅",
            "3D收盘涨跌幅",
            "3D最高涨幅",
            "3D最低涨幅",
            "4D收盘涨跌幅",
            "4D最高涨幅",
            "4D最低涨幅",
            "5D收盘涨跌幅",
            "5D最高涨幅",
            "5D最低涨幅",
            "5日里最高涨幅",
            "5日里最低涨幅",
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for _, row in detail_df.iterrows():
        code = str(row.get("symbol", "")).strip()
        if code.lower() in {"", "nan", "none", "<na>"}:
            file_symbol = str(row.get("file_symbol", "")).strip()
            code = file_symbol[-6:] if file_symbol else ""
        elif code.isdigit():
            code = code.zfill(6)

        record: dict[str, object] = {
            "信号日": row["signal_date"],
            "代码": code,
            "名称": row["name"],
            "反包类型": row.get("kong_turn_bull_pattern"),
            "信号日K线涨幅": format_pct(row.get("signal_candle_ret")),
            "反包参考低点": row.get("reversal_low_ref"),
            "反包基准低点涨幅": format_pct(row.get("reversal_low_gap_ret")),
        }
        high_values: list[float] = []
        low_values: list[float] = []

        for horizon in HORIZONS:
            close_ret = row.get(f"day{horizon}_close_ret")
            high_ret = row.get(f"day{horizon}_high_ret")
            low_ret = row.get(f"day{horizon}_low_ret")

            record[f"{horizon}D收盘涨跌幅"] = format_pct(close_ret)
            record[f"{horizon}D最高涨幅"] = format_pct(high_ret)
            record[f"{horizon}D最低涨幅"] = format_pct(low_ret)

            if pd.notna(high_ret):
                high_values.append(float(high_ret))
            if pd.notna(low_ret):
                low_values.append(float(low_ret))

        record["5日里最高涨幅"] = format_pct(max(high_values) if high_values else None)
        record["5日里最低涨幅"] = format_pct(min(low_values) if low_values else None)
        rows.append(record)

    columns = [
        "信号日",
        "代码",
        "名称",
        "反包类型",
        "信号日K线涨幅",
        "反包参考低点",
        "反包基准低点涨幅",
        "1D收盘涨跌幅",
        "1D最高涨幅",
        "1D最低涨幅",
        "2D收盘涨跌幅",
        "2D最高涨幅",
        "2D最低涨幅",
        "3D收盘涨跌幅",
        "3D最高涨幅",
        "3D最低涨幅",
        "4D收盘涨跌幅",
        "4D最高涨幅",
        "4D最低涨幅",
        "5D收盘涨跌幅",
        "5D最高涨幅",
        "5D最低涨幅",
        "5日里最高涨幅",
        "5日里最低涨幅",
    ]
    return pd.DataFrame(rows, columns=columns)


def render_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
    formatters: dict[str, object] | None = None,
) -> str:
    if frame.empty:
        return "暂无数据。"

    actual_headers = headers or columns
    formatter_map = formatters or {}
    lines = [
        "| " + " | ".join(actual_headers) + " |",
        "| " + " | ".join("---" for _ in actual_headers) + " |",
    ]

    for _, row in frame[columns].iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            formatter = formatter_map.get(column)
            if formatter is not None:
                cell = str(formatter(value))
            elif pd.isna(value):
                cell = "-"
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def pct_formatter(value: object) -> str:
    return format_pct(value)


def count_formatter(value: object) -> str:
    return format_count(value)


def build_stock_detail_sections(detail_df: pd.DataFrame) -> list[str]:
    if detail_df.empty:
        return ["暂无逐股票明细。", ""]

    lines: list[str] = []
    grouped = detail_df.groupby(
        ["file_symbol", "symbol", "name"],
        dropna=False,
        sort=True,
    )

    for index, ((file_symbol, symbol, name), stock_df) in enumerate(grouped, start=1):
        lines.append(f"### {index}. {name}（{file_symbol}）")
        lines.append("")
        signal_dates = stock_df["signal_date"].astype(str).tolist()
        patterns = [
            str(value)
            for value in stock_df["kong_turn_bull_pattern"].dropna().unique().tolist()
            if str(value).strip()
        ]
        lines.append(f"- 分析区间入选次数：{len(stock_df)}")
        lines.append(f"- 入选信号日：{'、'.join(signal_dates)}")
        if patterns:
            lines.append(f"- 空转多形态：{'、'.join(patterns)}")

        signal_candle_values = pd.to_numeric(stock_df["signal_candle_ret"], errors="coerce")
        selected_gap_values = pd.to_numeric(stock_df["reversal_low_gap_ret"], errors="coerce")
        lines.append(f"- 信号日K线平均涨幅：{format_pct(safe_mean(signal_candle_values))}")
        lines.append(f"- 反包基准低点平均涨幅：{format_pct(safe_mean(selected_gap_values))}")
        first_day_values = pd.to_numeric(stock_df["day1_close_ret"], errors="coerce")
        fifth_day_values = pd.to_numeric(stock_df["day5_close_ret"], errors="coerce")
        lines.append(f"- 未来1日收盘平均涨跌幅：{format_pct(safe_mean(first_day_values))}")
        lines.append(f"- 未来5日收盘平均涨跌幅：{format_pct(safe_mean(fifth_day_values))}")
        lines.append("")

        for signal_row in stock_df.sort_values("signal_date").itertuples(index=False):
            lines.append(
                f"#### 信号日 {signal_row.signal_date} "
                f"（信号日收盘价 {format_number(signal_row.signal_close)}）"
            )
            lines.append("")
            lines.append(f"- 反包类型：{signal_row.kong_turn_bull_pattern}")
            lines.append(f"- 信号日K线涨幅：{format_pct(signal_row.signal_candle_ret)}")
            lines.append(f"- 反包参考低点：{signal_row.reversal_low_ref}")
            lines.append(f"- 反包基准低点涨幅：{format_pct(signal_row.reversal_low_gap_ret)}")
            lines.append("")
            rows: list[dict[str, object]] = []
            for horizon in HORIZONS:
                rows.append(
                    {
                        "horizon": f"{horizon}D",
                        "future_date": getattr(signal_row, f"day{horizon}_date"),
                        "status": getattr(signal_row, f"day{horizon}_status"),
                        "close": getattr(signal_row, f"day{horizon}_close"),
                        "close_ret": getattr(signal_row, f"day{horizon}_close_ret"),
                        "high": getattr(signal_row, f"day{horizon}_high"),
                        "high_ret": getattr(signal_row, f"day{horizon}_high_ret"),
                        "low": getattr(signal_row, f"day{horizon}_low"),
                        "low_ret": getattr(signal_row, f"day{horizon}_low_ret"),
                    }
                )

            signal_detail_table = pd.DataFrame(rows)
            lines.append(
                render_table(
                    signal_detail_table,
                    [
                        "horizon",
                        "future_date",
                        "close",
                        "close_ret",
                        "high",
                        "high_ret",
                        "low",
                        "low_ret",
                        "status",
                    ],
                    headers=[
                        "未来交易日",
                        "日期",
                        "收盘价",
                        "收盘涨跌幅",
                        "盘中最高价",
                        "最高涨跌幅",
                        "盘中最低价",
                        "最低涨跌幅",
                        "数据状态",
                    ],
                    formatters={
                        "close": format_number,
                        "close_ret": pct_formatter,
                        "high": format_number,
                        "high_ret": pct_formatter,
                        "low": format_number,
                        "low_ret": pct_formatter,
                    },
                )
            )
            lines.append("")

    return lines


def build_report(
    period_text: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    trading_dates: list[pd.Timestamp],
    period_trading_dates: list[pd.Timestamp],
    signal_daily_df: pd.DataFrame,
    horizon_summary_df: pd.DataFrame,
    stock_summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    latest_local_date: pd.Timestamp,
    output_paths: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append(f"# {period_text} 默认选股未来1至5个交易日表现分析报告")
    lines.append("")
    lines.append("## 一、回测口径")
    lines.append("")
    lines.append("- 选股入口：`strategies/选股/默认选股.py` 的 `run_default_selection`。")
    lines.append("- 每个信号日均按项目默认流程执行：周线多头母池 → 当日空转多 → 剔除月线空头 → 异动条件保留。")
    lines.append(
        f"- 信号日：{period_start.date().isoformat()} 至 {period_end.date().isoformat()} "
        "之间的每一个本地日线交易日，按 Asia/Shanghai 解释。"
    )
    lines.append("- 信号价格：当天日线收盘价。")
    lines.append("- 未来第 N 个交易日：按全市场日线交易日序列向后第 N 个交易日，不按自然日计算。")
    lines.append("- 收盘价涨跌幅：`未来第N日收盘价 / 信号日收盘价 - 1`。")
    lines.append("- 最高价涨跌幅：`未来第N日最高价 / 信号日收盘价 - 1`。")
    lines.append("- 最低价涨跌幅：`未来第N日最低价 / 信号日收盘价 - 1`。")
    lines.append("- 某只股票在未来交易日没有日线K线时记为缺失，不向后跳过该交易日，也不按0收益计算。")
    lines.append("")

    lines.append("## 二、数据覆盖与限制")
    lines.append("")
    lines.append(f"- 请求区间：{period_start.date().isoformat()} 至 {period_end.date().isoformat()}")
    lines.append(f"- 本地日线最新日期：{latest_local_date.date().isoformat()}")
    lines.append(f"- 分析区间本地交易日数量：{len(period_trading_dates)}")
    lines.append(f"- 实际执行默认选股的信号日数量：{len(signal_daily_df)}")
    lines.append(f"- 全市场交易日序列数量（含未来可用日期）：{len(trading_dates)}")
    if latest_local_date < period_end:
        lines.append(
            f"- 注意：本地数据库截至 {latest_local_date.date().isoformat()}，"
            f"因此区间末及最近几个信号日的第3至第5个交易日数据可能尚未发生或尚未同步。"
        )
    lines.append("- 缺失样本仅从对应 horizon 的有效样本统计中排除，不会影响其他 horizon。")
    lines.append("")

    total_signal_count = int(signal_daily_df["final_count"].sum()) if not signal_daily_df.empty else 0
    unique_stock_count = int(detail_df["file_symbol"].nunique()) if not detail_df.empty else 0
    signal_days_with_picks = int(signal_daily_df["final_count"].gt(0).sum()) if not signal_daily_df.empty else 0

    lines.append("## 三、总体概览")
    lines.append("")
    lines.append(f"- 分析区间累计选股样本数：{total_signal_count}")
    lines.append(f"- 有选股结果的交易日数量：{signal_days_with_picks}")
    lines.append(f"- 触发过信号的去重股票数量：{unique_stock_count}")
    lines.append("- 统计对象：每个“信号日 × 股票”样本。")
    lines.append("")

    horizon_columns = [
        "horizon",
        "valid_count",
        "missing_count",
        "close_avg_ret",
        "close_median_ret",
        "close_win_rate",
        "high_avg_ret",
        "high_median_ret",
        "high_above_signal_close_rate",
        "low_avg_ret",
        "low_median_ret",
        "low_above_signal_close_rate",
    ]
    lines.append("### 1至5日总体统计")
    lines.append("")
    lines.append(
        render_table(
            horizon_summary_df,
            horizon_columns,
            headers=[
                "未来日",
                "有效样本",
                "缺失样本",
                "收盘均值",
                "收盘中位数",
                "收盘上涨率",
                "最高均值",
                "最高中位数",
                "最高价高于信号收盘率",
                "最低均值",
                "最低中位数",
                "最低价高于信号收盘率",
            ],
            formatters={
                "horizon": lambda value: f"{int(value)}D",
                "valid_count": count_formatter,
                "missing_count": count_formatter,
                "close_avg_ret": pct_formatter,
                "close_median_ret": pct_formatter,
                "close_win_rate": pct_formatter,
                "high_avg_ret": pct_formatter,
                "high_median_ret": pct_formatter,
                "high_above_signal_close_rate": pct_formatter,
                "low_avg_ret": pct_formatter,
                "low_median_ret": pct_formatter,
                "low_above_signal_close_rate": pct_formatter,
            },
        )
    )
    lines.append("")
    lines.append("其中，收盘上涨率表示未来第N日收盘价高于信号日收盘价的比例；最高价和最低价的比例分别表示该日最高价、最低价仍高于信号日收盘价的比例。")
    lines.append("")

    lines.append("## 四、逐日选股结果与远期表现")
    lines.append("")
    daily_selection_columns = [
        "signal_date",
        "weekly_pool_count",
        "kong_turn_bull_count",
        "monthly_bearish_excluded_count",
        "abnormal_move_included_count",
        "final_count",
    ]
    lines.append("### 每个交易日的选股数量")
    lines.append("")
    lines.append(
        render_table(
            signal_daily_df,
            daily_selection_columns,
            headers=["信号日", "周线母池", "空转多", "月线剔除", "异动保留", "最终选股"],
            formatters={
                "weekly_pool_count": count_formatter,
                "kong_turn_bull_count": count_formatter,
                "monthly_bearish_excluded_count": count_formatter,
                "abnormal_move_included_count": count_formatter,
                "final_count": count_formatter,
            },
        )
    )
    lines.append("")

    daily_performance_rows: list[dict[str, object]] = []
    for _, row in signal_daily_df.iterrows():
        for horizon in HORIZONS:
            daily_performance_rows.append(
                {
                    "signal_date": row["signal_date"],
                    "horizon": horizon,
                    "sample_count": row.get(f"day{horizon}_valid_count"),
                    "missing_count": row.get(f"day{horizon}_missing_count"),
                    "close_avg_ret": row.get(f"day{horizon}_close_avg_ret"),
                    "close_median_ret": row.get(f"day{horizon}_close_median_ret"),
                    "close_win_rate": row.get(f"day{horizon}_close_win_rate"),
                    "high_avg_ret": row.get(f"day{horizon}_high_avg_ret"),
                    "low_avg_ret": row.get(f"day{horizon}_low_avg_ret"),
                }
            )
    daily_performance_df = pd.DataFrame(daily_performance_rows)
    lines.append("### 每个信号日的1至5日表现")
    lines.append("")
    lines.append(
        render_table(
            daily_performance_df,
            [
                "signal_date",
                "horizon",
                "sample_count",
                "missing_count",
                "close_avg_ret",
                "close_median_ret",
                "close_win_rate",
                "high_avg_ret",
                "low_avg_ret",
            ],
            headers=["信号日", "未来日", "有效样本", "缺失", "收盘均值", "收盘中位数", "收盘上涨率", "最高均值", "最低均值"],
            formatters={
                "horizon": lambda value: f"{int(value)}D",
                "sample_count": count_formatter,
                "missing_count": count_formatter,
                "close_avg_ret": pct_formatter,
                "close_median_ret": pct_formatter,
                "close_win_rate": pct_formatter,
                "high_avg_ret": pct_formatter,
                "low_avg_ret": pct_formatter,
            },
        )
    )
    lines.append("")

    lines.append("## 五、股票级汇总")
    lines.append("")
    lines.append(
        "以下按股票汇总其在分析区间内多次入选后的平均表现；随后直接展开每只股票的"
        "每个信号日及未来1至5个交易日明细。"
    )
    lines.append("")
    stock_report_columns = [
        "file_symbol",
        "name",
        "signal_count",
        "day1_close_ret_avg",
        "day1_close_win_rate",
        "day1_high_ret_avg",
        "day1_low_ret_avg",
        "day5_close_ret_avg",
        "day5_close_win_rate",
        "day5_high_ret_avg",
        "day5_low_ret_avg",
    ]
    stock_report_df = stock_summary_df.copy()
    if not stock_report_df.empty:
        stock_report_df = stock_report_df.sort_values(
            ["day1_close_ret_avg", "signal_count"],
            ascending=[False, False],
            na_position="last",
        ).head(30)
    lines.append("### 按1日收盘平均涨跌幅排序的股票")
    lines.append("")
    lines.append(
        render_table(
            stock_report_df,
            stock_report_columns,
            headers=["代码", "名称", "入选次数", "1D收盘均值", "1D胜率", "1D最高均值", "1D最低均值", "5D收盘均值", "5D胜率", "5D最高均值", "5D最低均值"],
            formatters={
                "signal_count": count_formatter,
                "day1_close_ret_avg": pct_formatter,
                "day1_close_win_rate": pct_formatter,
                "day1_high_ret_avg": pct_formatter,
                "day1_low_ret_avg": pct_formatter,
                "day5_close_ret_avg": pct_formatter,
                "day5_close_win_rate": pct_formatter,
                "day5_high_ret_avg": pct_formatter,
                "day5_low_ret_avg": pct_formatter,
            },
        )
    )
    lines.append("")

    lines.append("## 六、逐股票逐信号日明细")
    lines.append("")
    lines.append("每只股票按信号日分别列出未来第1至第5个交易日。涨跌幅均以对应信号日收盘价为基准；“盘中最高价”和“盘中最低价”分别使用未来交易日的最高价和最低价计算。")
    lines.append("")
    lines.extend(build_stock_detail_sections(detail_df))

    lines.append("## 七、极值样本")
    lines.append("")
    if not detail_df.empty:
        for horizon in (1, 5):
            close_column = f"day{horizon}_close_ret"
            high_column = f"day{horizon}_high_ret"
            low_column = f"day{horizon}_low_ret"
            valid_detail = detail_df.dropna(subset=[close_column]).copy()
            if valid_detail.empty:
                continue

            winners = valid_detail.sort_values(close_column, ascending=False).head(10)
            losers = valid_detail.sort_values(close_column, ascending=True).head(10)
            lines.append(f"### 未来{horizon}个交易日收盘涨幅前十")
            lines.append("")
            lines.append(
                render_table(
                    winners,
                    ["signal_date", "file_symbol", "name", close_column, high_column, low_column, f"day{horizon}_date"],
                    headers=["信号日", "代码", "名称", "收盘涨跌幅", "最高涨跌幅", "最低涨跌幅", "未来日期"],
                    formatters={
                        close_column: pct_formatter,
                        high_column: pct_formatter,
                        low_column: pct_formatter,
                    },
                )
            )
            lines.append("")
            lines.append(f"### 未来{horizon}个交易日收盘跌幅前十")
            lines.append("")
            lines.append(
                render_table(
                    losers,
                    ["signal_date", "file_symbol", "name", close_column, high_column, low_column, f"day{horizon}_date"],
                    headers=["信号日", "代码", "名称", "收盘涨跌幅", "最高涨跌幅", "最低涨跌幅", "未来日期"],
                    formatters={
                        close_column: pct_formatter,
                        high_column: pct_formatter,
                        low_column: pct_formatter,
                    },
                )
            )
            lines.append("")

    lines.append("## 八、报告说明")
    lines.append("")
    lines.append("- 本 Markdown 正文已经直接包含每只入选股票、每个信号日以及未来第1至第5个交易日的明细。")
    lines.append("- 新增字段采用单选口径：two_day 对应前1根低点，three_day 和 both 对应前2根低点。")
    lines.append("- 每个未来交易日明细包含日期、收盘价及收盘涨跌幅、盘中最高价及最高涨跌幅、盘中最低价及最低涨跌幅。")
    lines.append("- CSV 文件仅作为程序内部的可复核中间结果，不是阅读本报告的必要文件。")
    lines.append(f"- 本报告文件：`{output_paths['report'].name}`")
    lines.append("")
    lines.append("## 九、结论使用边界")
    lines.append("")
    lines.append("- 本报告是基于历史日线收盘、最高、最低价的统计分析，不包含手续费、滑点、涨跌停成交限制和实际持仓权重。")
    lines.append("- 同一只股票在多个交易日重复入选时，逐次入选视为独立信号样本；股票级汇总再进行合并。")
    lines.append("- 当前数据库尚未覆盖的未来交易日不会被填充为0收益，因此越靠近数据截止日的信号，5日样本数会自然减少。")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    period_start, period_end, suffix, signal_dir_label, period_text = resolve_period(args)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trading_dates = discover_trading_dates(
        data_root=data_root,
        range_start=period_start,
        range_end=period_end + pd.Timedelta(days=31),
    )
    period_trading_dates = [
        trade_date for trade_date in trading_dates if period_start <= trade_date <= period_end
    ]
    if not period_trading_dates:
        raise ValueError(
            f"本地数据库中没有找到 {period_start.date().isoformat()} 至 "
            f"{period_end.date().isoformat()} 的交易日。"
        )

    future_date_map = build_future_date_map(
        trading_dates=trading_dates,
        signal_dates=period_trading_dates,
    )

    signal_dir = output_dir / f"default_selection_forward_signal_pool_{signal_dir_label}"
    signal_dir.mkdir(parents=True, exist_ok=True)

    daily_cache: dict[str, pd.DataFrame] = {}
    detail_rows: list[dict[str, object]] = []
    daily_summary_rows: list[dict[str, object]] = []

    for index, signal_date in enumerate(period_trading_dates, start=1):
        print(
            f"[{index}/{len(period_trading_dates)}] "
            f"运行默认选股 {signal_date.date().isoformat()}",
            flush=True,
        )
        output_path = signal_dir / f"default_selection_{signal_date.date().isoformat()}.csv"
        signal_df, selection_summary = run_default_selection(
            reference_time=build_reference_time(signal_date),
            output_path=output_path,
        )
        if signal_df.empty:
            signal_df = pd.DataFrame(columns=SIGNAL_COLUMNS)
            signal_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        signal_detail_rows: list[dict[str, object]] = []
        future_dates = future_date_map[signal_date]
        for _, signal_row in signal_df.iterrows():
            detail_row = build_detail_row(
                signal_row=signal_row,
                signal_date=signal_date,
                future_dates=future_dates,
                data_root=data_root,
                daily_cache=daily_cache,
            )
            signal_detail_rows.append(detail_row)
            detail_rows.append(detail_row)

        signal_detail_df = pd.DataFrame(signal_detail_rows)
        daily_summary_rows.append(
            build_daily_summary_row(
                signal_date=signal_date,
                selection_summary=selection_summary,
                signal_df=signal_df,
                detail_df=signal_detail_df,
            )
        )

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ["signal_date", "file_symbol"],
            ascending=[True, True],
        ).reset_index(drop=True)

    daily_summary_df = pd.DataFrame(daily_summary_rows).sort_values("signal_date").reset_index(drop=True)
    horizon_summary_df = pd.DataFrame(
        [summarize_horizon(detail_df, horizon) for horizon in HORIZONS]
    )
    stock_summary_df = build_stock_summary(detail_df)
    wide_detail_df = build_wide_detail_df(detail_df)

    output_paths = {
        "signal_dir": signal_dir,
        "detail": output_dir / f"default_selection_forward_{suffix}_detail.csv",
        "daily_summary": output_dir / f"default_selection_forward_{suffix}_daily_summary.csv",
        "horizon_summary": output_dir / f"default_selection_forward_{suffix}_horizon_summary.csv",
        "stock_summary": output_dir / f"default_selection_forward_{suffix}_stock_summary.csv",
        "wide": output_dir / f"default_selection_forward_{suffix}_wide.csv",
        "report": output_dir / f"default_selection_forward_{suffix}_report.md",
    }

    detail_df.to_csv(output_paths["detail"], index=False, encoding="utf-8-sig")
    daily_summary_df.to_csv(output_paths["daily_summary"], index=False, encoding="utf-8-sig")
    horizon_summary_df.to_csv(output_paths["horizon_summary"], index=False, encoding="utf-8-sig")
    stock_summary_df.to_csv(output_paths["stock_summary"], index=False, encoding="utf-8-sig")
    wide_detail_df.to_csv(output_paths["wide"], index=False, encoding="utf-8-sig")
    output_paths["report"].write_text(
        build_report(
            period_text=period_text,
            period_start=period_start,
            period_end=period_end,
            trading_dates=trading_dates,
            period_trading_dates=period_trading_dates,
            signal_daily_df=daily_summary_df,
            horizon_summary_df=horizon_summary_df,
            stock_summary_df=stock_summary_df,
            detail_df=detail_df,
            latest_local_date=trading_dates[-1],
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )

    print(f"period={period_text}")
    print(f"period_trading_days={len(period_trading_dates)}")
    print(f"total_signal_rows={len(detail_df)}")
    print(f"unique_symbols={detail_df['file_symbol'].nunique() if not detail_df.empty else 0}")
    print(f"latest_local_daily_date={trading_dates[-1].date().isoformat()}")
    print(f"detail_path={output_paths['detail']}")
    print(f"daily_summary_path={output_paths['daily_summary']}")
    print(f"horizon_summary_path={output_paths['horizon_summary']}")
    print(f"stock_summary_path={output_paths['stock_summary']}")
    print(f"wide_path={output_paths['wide']}")
    print(f"report_path={output_paths['report']}")
    print(horizon_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
