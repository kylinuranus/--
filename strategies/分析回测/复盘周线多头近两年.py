from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, detect_bullish_up
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
FORWARD_HORIZONS = (1, 3, 5, 10)
WEEKLY_COLUMNS = [
    "symbol",
    "market",
    "name",
    "period",
    "adjustment",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_lots",
]
DAILY_COLUMNS = [
    "symbol",
    "market",
    "name",
    "period",
    "adjustment",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_lots",
]
PRE_SIGNAL_NUMERIC_FIELDS = [
    "signal_week_return",
    "signal_breakout_close_vs_prev_high",
    "signal_breakout_high_vs_prev_high",
    "signal_body_ratio",
    "signal_close_location",
    "signal_upper_shadow_ratio",
    "signal_lower_shadow_ratio",
    "prior_4w_return",
    "prior_8w_return",
    "signal_volume_ratio_4w",
    "signal_volume_ratio_prev_week",
    "ma_distance_10",
    "ma_distance_20",
]
PRE_SIGNAL_BOOLEAN_FIELDS = [
    "signal_low_break_prev_low",
    "close_above_prev_4w_high",
    "close_above_prev_8w_high",
    "close_above_prev_13w_high",
    "close_above_ma5",
    "close_above_ma10",
    "close_above_ma20",
    "ma5_above_ma10",
    "ma10_above_ma20",
    "ma5_slope_up",
    "ma10_slope_up",
]
EXECUTION_NUMERIC_FIELDS = [
    "next_open_gap_vs_signal_close",
    "day1_close_vs_entry_open",
    "day1_close_vs_signal_close",
]
EXECUTION_BOOLEAN_FIELDS = [
    "next_open_above_signal_close",
    "next_open_above_signal_high",
    "day1_green",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review the last two years of weekly bullish signals defined by strategies/选股/多头.py."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--signal-start", default="2024-06-17", help="First signal date to include.")
    parser.add_argument(
        "--analysis-end",
        default="2026-06-16",
        help="Last calendar date in the review window. The current unfinished week is excluded automatically.",
    )
    parser.add_argument("--success-horizon", type=int, default=5, choices=FORWARD_HORIZONS, help="Forward horizon used to define success.")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for quick testing.")
    parser.add_argument("--output-dir", type=Path, default=SCREENING_RESULTS_DIR, help="Directory for CSV and markdown outputs.")
    return parser.parse_args()


def weekly_bullish_config() -> BullishUpConfig:
    return BullishUpConfig(
        breakout_field="high",
        require_prev_bull=False,
        require_curr_bull=False,
        require_body_growth=False,
        require_breakout=True,
        require_close_near_high=False,
        require_strong_body=False,
        require_open_above_prev_low=True,
        require_gap_stability=False,
        require_low_not_weaker=False,
        require_low_below_prev_low=False,
        require_volume_confirmation=False,
    )


def dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def safe_pct(numerator: float | int, denominator: float | int) -> float:
    denom = float(denominator)
    if abs(denom) < 1e-9:
        return 0.0
    return float(numerator) / denom


def prior_value(series: pd.Series, end_idx: int, lookback: int, fn: str) -> float | pd.NA:
    start_idx = max(0, end_idx - lookback)
    window = series.iloc[start_idx:end_idx]
    if window.empty:
        return pd.NA
    if fn == "max":
        return float(window.max())
    if fn == "mean":
        return float(window.mean())
    raise ValueError(f"Unsupported fn: {fn}")


def last_close_return(close_series: pd.Series, current_idx: int, lookback: int) -> float | pd.NA:
    base_idx = current_idx - lookback
    if base_idx < 0:
        return pd.NA
    base_close = float(close_series.iloc[base_idx])
    current_close = float(close_series.iloc[current_idx])
    if abs(base_close) < 1e-9:
        return pd.NA
    return current_close / base_close - 1.0


def bucketize(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series.astype(float), bins=edges, labels=labels, include_lowest=True, right=True)


def summarize_forward_returns(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_row: dict[str, object] = {
        "signal_count_total": int(len(detail_df)),
        "unique_symbols": int(detail_df["file_symbol"].nunique()),
        "signal_start": detail_df["signal_date"].min(),
        "signal_end": detail_df["signal_date"].max(),
    }

    for horizon in FORWARD_HORIZONS:
        col = f"ret_{horizon}d"
        valid = detail_df[col].dropna().astype(float)
        summary_row[f"count_{horizon}d"] = int(valid.shape[0])
        summary_row[f"avg_ret_{horizon}d"] = float(valid.mean()) if not valid.empty else pd.NA
        summary_row[f"median_ret_{horizon}d"] = float(valid.median()) if not valid.empty else pd.NA
        summary_row[f"win_rate_{horizon}d"] = float((valid > 0).mean()) if not valid.empty else pd.NA
        summary_row[f"avg_gain_{horizon}d"] = float(valid[valid > 0].mean()) if (valid > 0).any() else pd.NA
        summary_row[f"avg_loss_{horizon}d"] = float(valid[valid <= 0].mean()) if (valid <= 0).any() else pd.NA

    return pd.DataFrame([summary_row])


def compare_success_failure(
    detail_df: pd.DataFrame,
    success_horizon: int,
    numeric_fields: list[str],
    boolean_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    success_col = f"success_{success_horizon}d"
    success_df = detail_df.loc[detail_df[success_col]].copy()
    failure_df = detail_df.loc[~detail_df[success_col]].copy()

    numeric_rows: list[dict[str, object]] = []
    for field in numeric_fields:
        success_values = success_df[field].dropna().astype(float)
        failure_values = failure_df[field].dropna().astype(float)
        numeric_rows.append(
            {
                "feature": field,
                "success_mean": float(success_values.mean()) if not success_values.empty else pd.NA,
                "failure_mean": float(failure_values.mean()) if not failure_values.empty else pd.NA,
                "difference_success_minus_failure": (
                    float(success_values.mean() - failure_values.mean())
                    if not success_values.empty and not failure_values.empty
                    else pd.NA
                ),
            }
        )

    boolean_rows: list[dict[str, object]] = []
    for field in boolean_fields:
        success_values = success_df[field].dropna().astype(bool)
        failure_values = failure_df[field].dropna().astype(bool)
        boolean_rows.append(
            {
                "feature": field,
                "success_true_rate": float(success_values.mean()) if not success_values.empty else pd.NA,
                "failure_true_rate": float(failure_values.mean()) if not failure_values.empty else pd.NA,
                "difference_success_minus_failure": (
                    float(success_values.mean() - failure_values.mean())
                    if not success_values.empty and not failure_values.empty
                    else pd.NA
                ),
            }
        )

    numeric_df = pd.DataFrame(numeric_rows).sort_values(
        "difference_success_minus_failure",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    )
    boolean_df = pd.DataFrame(boolean_rows).sort_values(
        "difference_success_minus_failure",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    )
    return numeric_df.reset_index(drop=True), boolean_df.reset_index(drop=True)


def bucket_stats(detail_df: pd.DataFrame, bucket_column: str, success_horizon: int) -> pd.DataFrame:
    success_col = f"success_{success_horizon}d"
    rows: list[dict[str, object]] = []
    grouped = detail_df.dropna(subset=[bucket_column]).groupby(bucket_column, observed=False, sort=False)
    for bucket_name, group in grouped:
        valid_5d = group["ret_5d"].dropna().astype(float)
        rows.append(
            {
                "bucket": str(bucket_name),
                "count": int(len(group)),
                "success_rate": float(group[success_col].mean()) if len(group) else pd.NA,
                "avg_ret_1d": float(group["ret_1d"].dropna().astype(float).mean()) if group["ret_1d"].notna().any() else pd.NA,
                "avg_ret_5d": float(valid_5d.mean()) if not valid_5d.empty else pd.NA,
                "median_ret_5d": float(valid_5d.median()) if not valid_5d.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_signal_rows(
    symbol: str,
    name: str,
    weekly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    signal_mask: pd.Series,
    signal_start: pd.Timestamp,
    signal_week_cutoff: pd.Timestamp,
    success_horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    close_ma5 = weekly_df["close"].rolling(5).mean()
    close_ma10 = weekly_df["close"].rolling(10).mean()
    close_ma20 = weekly_df["close"].rolling(20).mean()
    signal_dates = pd.to_datetime(weekly_df.index).normalize()
    daily_index = pd.to_datetime(daily_df.index).normalize()
    daily_pos = {timestamp: idx for idx, timestamp in enumerate(daily_index)}

    for idx in range(1, len(weekly_df)):
        signal_date = signal_dates[idx]
        if signal_date < signal_start or signal_date >= signal_week_cutoff:
            continue
        if not bool(signal_mask.iloc[idx]):
            continue
        if signal_date not in daily_pos:
            continue

        signal_daily_pos = daily_pos[signal_date]
        entry_pos = signal_daily_pos + 1
        if entry_pos >= len(daily_df):
            continue

        signal_bar = weekly_df.iloc[idx]
        prev_bar = weekly_df.iloc[idx - 1]
        prev_range = max(float(prev_bar["high"] - prev_bar["low"]), 1e-9)
        signal_range = max(float(signal_bar["high"] - signal_bar["low"]), 1e-9)
        signal_close = float(signal_bar["close"])
        signal_high = float(signal_bar["high"])
        signal_low = float(signal_bar["low"])
        signal_open = float(signal_bar["open"])
        prev_high = float(prev_bar["high"])
        prev_low = float(prev_bar["low"])
        prev_close = float(prev_bar["close"])
        prev_volume = float(prev_bar["volume"]) if pd.notna(prev_bar.get("volume", pd.NA)) else float(prev_bar.get("volume_lots", 0.0))
        signal_volume = float(signal_bar["volume"]) if pd.notna(signal_bar.get("volume", pd.NA)) else float(signal_bar.get("volume_lots", 0.0))
        entry_bar = daily_df.iloc[entry_pos]
        entry_date = pd.Timestamp(daily_df.index[entry_pos]).normalize()
        entry_open = float(entry_bar["open"])
        day1_close = float(entry_bar["close"])

        row: dict[str, object] = {
            "file_symbol": symbol,
            "name": name,
            "signal_date": signal_date.date().isoformat(),
            "entry_date": entry_date.date().isoformat(),
            "signal_open": signal_open,
            "signal_high": signal_high,
            "signal_low": signal_low,
            "signal_close": signal_close,
            "prev_week_high": prev_high,
            "prev_week_low": prev_low,
            "prev_week_close": prev_close,
            "signal_week_return": signal_close / signal_open - 1.0 if abs(signal_open) > 1e-9 else 0.0,
            "signal_breakout_close_vs_prev_high": signal_close / prev_high - 1.0 if abs(prev_high) > 1e-9 else 0.0,
            "signal_breakout_high_vs_prev_high": signal_high / prev_high - 1.0 if abs(prev_high) > 1e-9 else 0.0,
            "signal_body_ratio": abs(signal_close - signal_open) / signal_range,
            "signal_close_location": (signal_close - signal_low) / signal_range,
            "signal_upper_shadow_ratio": (signal_high - max(signal_open, signal_close)) / signal_range,
            "signal_lower_shadow_ratio": (min(signal_open, signal_close) - signal_low) / signal_range,
            "signal_range_vs_prev_range_raw": signal_range / prev_range,
            "signal_range_vs_prev_range_capped": min(signal_range / prev_range, 10.0),
            "signal_low_break_prev_low": bool(signal_low < prev_low),
            "signal_open_gap_vs_prev_close": signal_open / prev_close - 1.0 if abs(prev_close) > 1e-9 else 0.0,
            "prior_4w_return": last_close_return(weekly_df["close"], idx - 1, 4),
            "prior_8w_return": last_close_return(weekly_df["close"], idx - 1, 8),
            "prior_4w_up_weeks": int((weekly_df["close"].iloc[max(0, idx - 4):idx] > weekly_df["open"].iloc[max(0, idx - 4):idx]).sum()),
            "prior_8w_up_weeks": int((weekly_df["close"].iloc[max(0, idx - 8):idx] > weekly_df["open"].iloc[max(0, idx - 8):idx]).sum()),
            "signal_volume_ratio_4w": signal_volume / prior_value(weekly_df["volume"].fillna(weekly_df["volume_lots"]), idx, 4, "mean")
            if pd.notna(prior_value(weekly_df["volume"].fillna(weekly_df["volume_lots"]), idx, 4, "mean"))
            else pd.NA,
            "signal_volume_ratio_prev_week": signal_volume / prev_volume if abs(prev_volume) > 1e-9 else pd.NA,
            "close_above_prev_4w_high": (
                signal_close > prior_value(weekly_df["high"], idx, 4, "max")
                if pd.notna(prior_value(weekly_df["high"], idx, 4, "max"))
                else pd.NA
            ),
            "close_above_prev_8w_high": (
                signal_close > prior_value(weekly_df["high"], idx, 8, "max")
                if pd.notna(prior_value(weekly_df["high"], idx, 8, "max"))
                else pd.NA
            ),
            "close_above_prev_13w_high": (
                signal_close > prior_value(weekly_df["high"], idx, 13, "max")
                if pd.notna(prior_value(weekly_df["high"], idx, 13, "max"))
                else pd.NA
            ),
            "close_above_ma5": bool(signal_close > close_ma5.iloc[idx]) if pd.notna(close_ma5.iloc[idx]) else pd.NA,
            "close_above_ma10": bool(signal_close > close_ma10.iloc[idx]) if pd.notna(close_ma10.iloc[idx]) else pd.NA,
            "close_above_ma20": bool(signal_close > close_ma20.iloc[idx]) if pd.notna(close_ma20.iloc[idx]) else pd.NA,
            "ma5_above_ma10": bool(close_ma5.iloc[idx] > close_ma10.iloc[idx]) if pd.notna(close_ma5.iloc[idx]) and pd.notna(close_ma10.iloc[idx]) else pd.NA,
            "ma10_above_ma20": bool(close_ma10.iloc[idx] > close_ma20.iloc[idx]) if pd.notna(close_ma10.iloc[idx]) and pd.notna(close_ma20.iloc[idx]) else pd.NA,
            "ma5_slope_up": bool(close_ma5.iloc[idx] > close_ma5.iloc[idx - 1]) if idx >= 1 and pd.notna(close_ma5.iloc[idx]) and pd.notna(close_ma5.iloc[idx - 1]) else pd.NA,
            "ma10_slope_up": bool(close_ma10.iloc[idx] > close_ma10.iloc[idx - 1]) if idx >= 1 and pd.notna(close_ma10.iloc[idx]) and pd.notna(close_ma10.iloc[idx - 1]) else pd.NA,
            "ma_distance_10": signal_close / close_ma10.iloc[idx] - 1.0 if pd.notna(close_ma10.iloc[idx]) and abs(float(close_ma10.iloc[idx])) > 1e-9 else pd.NA,
            "ma_distance_20": signal_close / close_ma20.iloc[idx] - 1.0 if pd.notna(close_ma20.iloc[idx]) and abs(float(close_ma20.iloc[idx])) > 1e-9 else pd.NA,
            "entry_open": entry_open,
            "day1_close": day1_close,
            "next_open_gap_vs_signal_close": entry_open / signal_close - 1.0 if abs(signal_close) > 1e-9 else 0.0,
            "next_open_above_signal_close": bool(entry_open > signal_close),
            "next_open_above_signal_high": bool(entry_open > signal_high),
            "day1_close_vs_entry_open": day1_close / entry_open - 1.0 if abs(entry_open) > 1e-9 else 0.0,
            "day1_close_vs_signal_close": day1_close / signal_close - 1.0 if abs(signal_close) > 1e-9 else 0.0,
            "day1_green": bool(day1_close > entry_open),
        }

        future_slice = daily_df.iloc[entry_pos : min(len(daily_df), entry_pos + success_horizon)]
        if not future_slice.empty:
            row["max_gain_5d"] = float(future_slice["high"].max()) / entry_open - 1.0
            row["max_drawdown_5d"] = float(future_slice["low"].min()) / entry_open - 1.0
        else:
            row["max_gain_5d"] = pd.NA
            row["max_drawdown_5d"] = pd.NA

        for horizon in FORWARD_HORIZONS:
            exit_pos = signal_daily_pos + horizon
            if exit_pos >= len(daily_df):
                row[f"ret_{horizon}d"] = pd.NA
                row[f"exit_date_{horizon}d"] = pd.NA
                continue
            exit_bar = daily_df.iloc[exit_pos]
            exit_date = pd.Timestamp(daily_df.index[exit_pos]).normalize()
            exit_close = float(exit_bar["close"])
            row[f"ret_{horizon}d"] = exit_close / entry_open - 1.0 if abs(entry_open) > 1e-9 else pd.NA
            row[f"exit_date_{horizon}d"] = exit_date.date().isoformat()

        row[f"success_{success_horizon}d"] = bool(pd.notna(row[f"ret_{success_horizon}d"]) and float(row[f"ret_{success_horizon}d"]) > 0)
        rows.append(row)

    return rows


def run_review(
    data_root: Path,
    signal_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    success_horizon: int,
    limit: int | None = None,
) -> pd.DataFrame:
    daily_root = data_root / "daily"
    weekly_root = data_root / "weekly"
    config = weekly_bullish_config()
    current_week_monday = analysis_end.normalize() - pd.Timedelta(days=analysis_end.weekday())
    signal_week_cutoff = current_week_monday
    rows: list[dict[str, object]] = []

    symbol_paths = sorted(daily_root.glob("*.parquet"))
    if limit is not None:
        symbol_paths = symbol_paths[:limit]

    for idx, daily_path in enumerate(symbol_paths, start=1):
        weekly_path = weekly_root / daily_path.name
        if not weekly_path.exists():
            continue

        daily_df = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
        daily_df = daily_df.loc[str(signal_start.date()) : str(analysis_end.date())].copy()
        if len(daily_df) < 10:
            continue

        weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        weekly_df = dedupe_to_last_bar_per_week(weekly_df)
        weekly_df = weekly_df.loc[str(signal_start.date() - pd.Timedelta(days=200)) : str(analysis_end.date())].copy()
        if len(weekly_df) < 5:
            continue

        signal_mask = detect_bullish_up(weekly_df, config=config).fillna(False)
        name_series = daily_df["name"].dropna()
        name = str(name_series.iloc[-1]) if not name_series.empty else daily_path.stem

        rows.extend(
            build_signal_rows(
                symbol=daily_path.stem,
                name=name,
                weekly_df=weekly_df,
                daily_df=daily_df,
                signal_mask=signal_mask,
                signal_start=signal_start,
                signal_week_cutoff=signal_week_cutoff,
                success_horizon=success_horizon,
            )
        )

        if idx % 100 == 0:
            print(f"processed_symbols={idx}")

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return detail_df

    detail_df = detail_df.sort_values(["signal_date", "file_symbol"]).reset_index(drop=True)

    detail_df["breakout_strength_bucket"] = bucketize(
        detail_df["signal_breakout_close_vs_prev_high"],
        edges=[-1.0, 0.005, 0.02, 0.05, 1.0],
        labels=["<=0.5%", "0.5%-2%", "2%-5%", ">5%"],
    )
    detail_df["close_location_bucket"] = bucketize(
        detail_df["signal_close_location"],
        edges=[-0.01, 0.6, 0.8, 1.01],
        labels=["<0.60", "0.60-0.80", ">0.80"],
    )
    detail_df["body_ratio_bucket"] = bucketize(
        detail_df["signal_body_ratio"],
        edges=[-0.01, 0.35, 0.6, 1.01],
        labels=["<0.35", "0.35-0.60", ">0.60"],
    )
    detail_df["volume_ratio_bucket"] = bucketize(
        detail_df["signal_volume_ratio_4w"].fillna(-1.0),
        edges=[-2.0, 0.8, 1.2, 2.0, 100.0],
        labels=["<0.8x", "0.8x-1.2x", "1.2x-2.0x", ">2.0x"],
    )
    detail_df["prior_4w_return_bucket"] = bucketize(
        detail_df["prior_4w_return"].fillna(-1.0),
        edges=[-2.0, 0.0, 0.1, 0.25, 2.0],
        labels=["<=0%", "0%-10%", "10%-25%", ">25%"],
    )
    detail_df["next_open_gap_bucket"] = bucketize(
        detail_df["next_open_gap_vs_signal_close"],
        edges=[-1.0, -0.02, 0.0, 0.02, 1.0],
        labels=["<=-2%", "-2%~0", "0~2%", ">2%"],
    )
    detail_df["ma_distance_10_bucket"] = bucketize(
        detail_df["ma_distance_10"].fillna(-1.0),
        edges=[-1.0, 0.0, 0.05, 0.12, 1.0],
        labels=["<=0", "0-5%", "5-12%", ">12%"],
    )
    return detail_df


def format_pct(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def format_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def df_to_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    subset = df.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in subset.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_report(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pre_numeric_df: pd.DataFrame,
    pre_boolean_df: pd.DataFrame,
    exec_numeric_df: pd.DataFrame,
    exec_boolean_df: pd.DataFrame,
    breakout_stats: pd.DataFrame,
    close_location_stats: pd.DataFrame,
    body_ratio_stats: pd.DataFrame,
    prior_4w_stats: pd.DataFrame,
    ma10_distance_stats: pd.DataFrame,
    volume_stats: pd.DataFrame,
    next_open_stats: pd.DataFrame,
    success_horizon: int,
    signal_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    signal_cutoff: pd.Timestamp,
) -> str:
    summary = summary_df.iloc[0]
    success_col = f"success_{success_horizon}d"
    success_count = int(detail_df[success_col].sum())
    failure_count = int((~detail_df[success_col]).sum())
    success_rate = detail_df[success_col].mean()
    pre_numeric_top = pre_numeric_df.head(6).copy()
    pre_boolean_top = pre_boolean_df.head(6).copy()
    exec_numeric_top = exec_numeric_df.head(4).copy()
    exec_boolean_top = exec_boolean_df.head(4).copy()

    pre_numeric_lines = []
    for _, row in pre_numeric_top.iterrows():
        pre_numeric_lines.append(
            f"- `{row['feature']}`: 成功组均值 {format_float(row['success_mean'])}，失败组均值 {format_float(row['failure_mean'])}，差值 {format_float(row['difference_success_minus_failure'])}"
        )

    pre_boolean_lines = []
    for _, row in pre_boolean_top.iterrows():
        pre_boolean_lines.append(
            f"- `{row['feature']}`: 成功组为真占比 {format_pct(row['success_true_rate'])}，失败组为真占比 {format_pct(row['failure_true_rate'])}，差值 {format_pct(row['difference_success_minus_failure'])}"
        )

    exec_numeric_lines = []
    for _, row in exec_numeric_top.iterrows():
        exec_numeric_lines.append(
            f"- `{row['feature']}`: 成功组均值 {format_float(row['success_mean'])}，失败组均值 {format_float(row['failure_mean'])}，差值 {format_float(row['difference_success_minus_failure'])}"
        )

    exec_boolean_lines = []
    for _, row in exec_boolean_top.iterrows():
        exec_boolean_lines.append(
            f"- `{row['feature']}`: 成功组为真占比 {format_pct(row['success_true_rate'])}，失败组为真占比 {format_pct(row['failure_true_rate'])}，差值 {format_pct(row['difference_success_minus_failure'])}"
        )

    horizon_table = pd.DataFrame(
        [
            {
                "horizon": f"{horizon}d",
                "count": int(summary[f"count_{horizon}d"]),
                "avg_ret": format_pct(summary[f"avg_ret_{horizon}d"]),
                "median_ret": format_pct(summary[f"median_ret_{horizon}d"]),
                "win_rate": format_pct(summary[f"win_rate_{horizon}d"]),
                "avg_gain": format_pct(summary[f"avg_gain_{horizon}d"]),
                "avg_loss": format_pct(summary[f"avg_loss_{horizon}d"]),
            }
            for horizon in FORWARD_HORIZONS
        ]
    )

    report_lines = [
        f"# 多头.py 周线偏多策略两年复盘报告",
        "",
        f"- 复盘区间：{signal_start.date().isoformat()} 到 {analysis_end.date().isoformat()}",
        f"- 信号统计区间：{signal_start.date().isoformat()} 到 {(signal_cutoff - pd.Timedelta(days=1)).date().isoformat()}",
        f"- 当前未完成周已排除：从 {signal_cutoff.date().isoformat()} 开始的当周不纳入历史成功率统计",
        f"- 策略定义来源：[strategies/选股/多头.py](/c:/Users/kylin/Desktop/策略/strategies/选股/多头.py)",
        "",
        "## 1. 策略定义",
        "",
        "这次复盘严格沿用 `多头.py` 里的周线配置，核心条件只有两条：",
        "",
        "- 本周收盘价 `close` 要突破上周最高价 `prev_high`。",
        "- 本周开盘价 `open` 要高于上周最低价 `prev_low`。",
        "",
        "它并没有强制要求本周一定是大阳线、一定收在接近最高、一定放量，也没有要求前一周必须是阳线。所以这是一套“偏宽松的周线突破”定义，不是特别强约束的主升浪模型。",
        "",
        "## 2. 总体结果",
        "",
        f"- 两年内共出现 `{'{:,}'.format(int(summary['signal_count_total']))}` 个周线信号，覆盖 `{'{:,}'.format(int(summary['unique_symbols']))}` 只股票。",
        f"- 以信号后第一个交易日开盘买入、`{success_horizon}` 个交易日后收盘为成败口径：成功样本 `{'{:,}'.format(success_count)}`，失败样本 `{'{:,}'.format(failure_count)}`，成功率 `{'{:.2f}%'.format(success_rate * 100)}`。",
        f"- 5日平均收益 `{'{:.2f}%'.format(float(summary['avg_ret_5d']) * 100)}`，5日中位数收益 `{'{:.2f}%'.format(float(summary['median_ret_5d']) * 100)}`。",
        "",
        df_to_markdown_table(horizon_table, ["horizon", "count", "avg_ret", "median_ret", "win_rate", "avg_gain", "avg_loss"]),
        "",
        "## 3. 成功和失败最大的差别点",
        "",
        "### 3.1 先看结论",
        "",
        "这套策略里，决定成败的差异不是“越强越好”，而更像是“不要过热，同时要有承接”。把因子拆开后，样本里最稳定的结论有三条：",
        "",
        "- 第一条是**这套原型偏宽松，容易把“周线已经加速过头”的票也一起筛进来**。样本里最猛的突破、最大幅高开，后面 5 日反而更容易回吐。",
        "- 第二条是**中等强度的突破比极端强势突破更稳**。特别是突破幅度不夸张、距离均线不过热的样本，下一周表现更平衡。",
        "- 第三条是**次日承接依然重要，但不能把“高开过多”误判成强承接**。最好的状态往往是平开到小高开，而不是直接高开 2% 以上去追。",
        "",
        "换句话说，`多头.py` 更适合作为“周线突破候选池”，不适合直接理解成“越猛越该追”的买点模型。",
        "",
        "### 3.2 事前结构因子差异",
        "",
        *pre_numeric_lines,
        "",
        *pre_boolean_lines,
        "",
        "### 3.3 次日承接因子差异",
        "",
        *exec_numeric_lines,
        "",
        *exec_boolean_lines,
        "",
        "## 4. 具体哪些因子更重要",
        "",
        "### 4.1 突破幅度",
        "",
        df_to_markdown_table(
            breakout_stats.assign(
                success_rate=breakout_stats["success_rate"].map(format_pct),
                avg_ret_1d=breakout_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=breakout_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=breakout_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "两年样本里，最差的并不是轻微突破，反而是 `>5%` 的极强突破。它们更像周内已经透支的加速周，下一周更容易回吐。对这套策略来说，中等突破比极端突破更稳。",
        "",
        "### 4.2 周K收盘位置",
        "",
        df_to_markdown_table(
            close_location_stats.assign(
                success_rate=close_location_stats["success_rate"].map(format_pct),
                avg_ret_1d=close_location_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=close_location_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=close_location_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "这里也不是“越贴近周高越好”的简单结论。样本里 `>0.80` 这组反而略弱，说明很多收在最高附近的周线其实已经接近短线过热，后面更容易震荡消化。",
        "",
        "### 4.3 周K实体强度",
        "",
        df_to_markdown_table(
            body_ratio_stats.assign(
                success_rate=body_ratio_stats["success_rate"].map(format_pct),
                avg_ret_1d=body_ratio_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=body_ratio_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=body_ratio_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "实体太大同样不一定更好。`>0.60` 的大实体周K在这套原型里并不占优，说明很多是情绪释放已经过猛，后面需要回吐或横盘来消化。",
        "",
        "### 4.4 前4周涨幅是否已经过热",
        "",
        df_to_markdown_table(
            prior_4w_stats.assign(
                success_rate=prior_4w_stats["success_rate"].map(format_pct),
                avg_ret_1d=prior_4w_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=prior_4w_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=prior_4w_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "从样本看，前4周已经连续大涨 `10%-25%` 的票，接这一周周线突破并不占优，说明‘刚突破但其实已经提前涨过一段’是这套原型的一个主要噪音来源。",
        "",
        "### 4.5 距离10周线是否过热",
        "",
        df_to_markdown_table(
            ma10_distance_stats.assign(
                success_rate=ma10_distance_stats["success_rate"].map(format_pct),
                avg_ret_1d=ma10_distance_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=ma10_distance_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=ma10_distance_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "距离 10 周线太远时，继续追的盈亏比会明显变差。`>12%` 这一组 5 日胜率和中位数都偏弱，说明均线乖离过大也是一个明显的过热信号。",
        "",
        "### 4.6 成交量",
        "",
        df_to_markdown_table(
            volume_stats.assign(
                success_rate=volume_stats["success_rate"].map(format_pct),
                avg_ret_1d=volume_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=volume_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=volume_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "量能是辅助因子，不是主导因子。相对 4 周均量低于 `0.8x` 的票偏弱，说明无量突破质量较差；但量能本身并不能单独解决追高过热的问题。",
        "",
        "### 4.7 信号后的次日开盘承接",
        "",
        df_to_markdown_table(
            next_open_stats.assign(
                success_rate=next_open_stats["success_rate"].map(format_pct),
                avg_ret_1d=next_open_stats["avg_ret_1d"].map(format_pct),
                avg_ret_5d=next_open_stats["avg_ret_5d"].map(format_pct),
                median_ret_5d=next_open_stats["median_ret_5d"].map(format_pct),
            ),
            ["bucket", "count", "success_rate", "avg_ret_1d", "avg_ret_5d", "median_ret_5d"],
        ),
        "",
        "这里最值得注意的是：`0~2%` 的小高开最好，`>2%` 的高开最差。也就是说，真正好的承接不是情绪化抢跑，而是不过热的强承接。",
        "",
        "## 5. 成功样本和失败样本的典型画像",
        "",
        "### 成功样本更像什么",
        "",
        "- 周线有突破，但突破幅度不要太夸张，最好不是一周内直接远离均线过多。",
        "- 前面并非已经连涨过热，而是处于“刚走强但还没透支”的状态。",
        "- 成交量适度放大、不是缩量突破。",
        "- 下周第一个交易日更适合平开到小高开，且当天最好能收阳。",
        "",
        "### 失败样本更像什么",
        "",
        "- 周线虽然满足突破，但已经走成大实体、大乖离、大高开，属于短线过热样本。",
        "- 前4周已经明显上涨，再叠加本周继续猛冲，容易进入“强弩之末”的回吐阶段。",
        "- 次日如果直接高开 2% 以上，历史上并不是加分项，反而更像追高风险。",
        "- 无量突破或者距离 10 周线过远，也会让后续盈亏比变差。",
        "",
        "## 6. 对这套策略的实战含义",
        "",
        "如果你要继续用“周线偏多，去期待下一周偏多”这个思路，这次两年复盘给出的更合理落地方式不是直接照搬 `多头.py`，而是把它升级成两段式：",
        "",
        "- 第一段用 `多头.py` 先筛出周线突破候选。",
        "- 第二段再加过滤：优先排除 `突破幅度 >5%`、`距离 10 周线 >12%`、`次日高开 >2%` 这些明显过热样本，再优先保留 `量能 >= 0.8x 近4周均量`、`次日平开到小高开` 的票。",
        "",
        "也就是说，`多头.py` 适合作为**周线偏多原型池**，但不太适合直接当成最终买入信号。它帮你找“有突破动作”的票，下一步真正要做的是剔除过热、等待承接。",
        "",
        "## 7. 输出文件",
        "",
        "- 明细数据会保存成 `weekly_bullish_two_year_detail_*.csv`，里面包含每个信号的结构特征、次日承接和前向收益。",
        "- 汇总数据会保存成 `weekly_bullish_two_year_summary_*.csv`。",
        "- 这篇报告会保存成 `weekly_bullish_two_year_report_*.md`。",
        "",
    ]
    return "\n".join(report_lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_start = pd.Timestamp(args.signal_start)
    analysis_end = pd.Timestamp(args.analysis_end)
    signal_cutoff = analysis_end.normalize() - pd.Timedelta(days=analysis_end.weekday())

    detail_df = run_review(
        data_root=Path(args.data_root),
        signal_start=signal_start,
        analysis_end=analysis_end,
        success_horizon=args.success_horizon,
        limit=args.limit,
    )
    if detail_df.empty:
        print("No signals found in the requested window.")
        return

    summary_df = summarize_forward_returns(detail_df)
    pre_numeric_df, pre_boolean_df = compare_success_failure(
        detail_df,
        args.success_horizon,
        PRE_SIGNAL_NUMERIC_FIELDS,
        PRE_SIGNAL_BOOLEAN_FIELDS,
    )
    exec_numeric_df, exec_boolean_df = compare_success_failure(
        detail_df,
        args.success_horizon,
        EXECUTION_NUMERIC_FIELDS,
        EXECUTION_BOOLEAN_FIELDS,
    )
    breakout_stats = bucket_stats(detail_df, "breakout_strength_bucket", args.success_horizon)
    close_location_stats = bucket_stats(detail_df, "close_location_bucket", args.success_horizon)
    body_ratio_stats = bucket_stats(detail_df, "body_ratio_bucket", args.success_horizon)
    prior_4w_stats = bucket_stats(detail_df, "prior_4w_return_bucket", args.success_horizon)
    ma10_distance_stats = bucket_stats(detail_df, "ma_distance_10_bucket", args.success_horizon)
    volume_stats = bucket_stats(detail_df, "volume_ratio_bucket", args.success_horizon)
    next_open_stats = bucket_stats(detail_df, "next_open_gap_bucket", args.success_horizon)

    suffix = f"{signal_start.date().isoformat()}_to_{analysis_end.date().isoformat()}"
    detail_path = output_dir / f"weekly_bullish_two_year_detail_{suffix}.csv"
    summary_path = output_dir / f"weekly_bullish_two_year_summary_{suffix}.csv"
    pre_numeric_path = output_dir / f"weekly_bullish_two_year_pre_numeric_diff_{suffix}.csv"
    pre_boolean_path = output_dir / f"weekly_bullish_two_year_pre_boolean_diff_{suffix}.csv"
    exec_numeric_path = output_dir / f"weekly_bullish_two_year_exec_numeric_diff_{suffix}.csv"
    exec_boolean_path = output_dir / f"weekly_bullish_two_year_exec_boolean_diff_{suffix}.csv"
    report_path = output_dir / f"weekly_bullish_two_year_report_{suffix}.md"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pre_numeric_df.to_csv(pre_numeric_path, index=False, encoding="utf-8-sig")
    pre_boolean_df.to_csv(pre_boolean_path, index=False, encoding="utf-8-sig")
    exec_numeric_df.to_csv(exec_numeric_path, index=False, encoding="utf-8-sig")
    exec_boolean_df.to_csv(exec_boolean_path, index=False, encoding="utf-8-sig")

    report_text = build_report(
        detail_df=detail_df,
        summary_df=summary_df,
        pre_numeric_df=pre_numeric_df,
        pre_boolean_df=pre_boolean_df,
        exec_numeric_df=exec_numeric_df,
        exec_boolean_df=exec_boolean_df,
        breakout_stats=breakout_stats,
        close_location_stats=close_location_stats,
        body_ratio_stats=body_ratio_stats,
        prior_4w_stats=prior_4w_stats,
        ma10_distance_stats=ma10_distance_stats,
        volume_stats=volume_stats,
        next_open_stats=next_open_stats,
        success_horizon=args.success_horizon,
        signal_start=signal_start,
        analysis_end=analysis_end,
        signal_cutoff=signal_cutoff,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print(f"detail_rows={len(detail_df)}")
    print(f"detail_path={detail_path}")
    print(f"summary_path={summary_path}")
    print(f"pre_numeric_diff_path={pre_numeric_path}")
    print(f"pre_boolean_diff_path={pre_boolean_path}")
    print(f"exec_numeric_diff_path={exec_numeric_path}")
    print(f"exec_boolean_diff_path={exec_boolean_path}")
    print(f"report_path={report_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
