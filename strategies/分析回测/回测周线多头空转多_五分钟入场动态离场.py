from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import analyze_kong_turn_bull, detect_bullish_up
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.分析回测.回测周线多头空转多_五分钟入场 import (
    DAILY_COLUMNS,
    DATA_ROOT,
    M5_COLUMNS,
    WEEKLY_COLUMNS,
    build_weekly_pool_map,
    bucketize_feature,
    compute_return,
    dedupe_to_last_bar_per_week,
    expectancy,
    first_monday_of_week,
    format_num,
    format_pct,
    kelly_fraction,
    kong_turn_bull_config,
    locate_first_5m_bar_for_day,
    max_drawdown_from_returns,
    normalize_daily_index,
    profit_factor,
    safe_mean,
    safe_median,
    safe_std,
    safe_sum,
    summarize_return_series,
    weekly_bullish_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest: weekly bullish pool -> daily kong-turn-bull -> next-day first 5m entry "
            "with dynamic exit rules."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--signal-start", default="2026-01-01", help="First signal date to include.")
    parser.add_argument("--analysis-end", default=None, help="Last calendar date to include.")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for quick testing.")
    parser.add_argument("--output-dir", type=Path, default=SCREENING_RESULTS_DIR, help="Directory for outputs.")
    return parser.parse_args()


def get_limit_up_ratio(file_symbol: str, name: str | None) -> float:
    symbol = file_symbol.lower()
    upper_name = str(name).upper() if name is not None else ""
    if "ST" in upper_name:
        return 0.05
    if symbol.startswith("bj"):
        return 0.30
    if symbol.startswith("sh688") or symbol.startswith("sz300") or symbol.startswith("sz301"):
        return 0.20
    return 0.10


def is_limit_up_day(
    file_symbol: str,
    name: str | None,
    close_price: float | int | None,
    high_price: float | int | None,
    prev_close: float | int | None,
) -> bool:
    limit_ratio = get_limit_up_ratio(file_symbol=file_symbol, name=name)
    close_ret = compute_return(close_price, prev_close)
    if close_ret is None:
        return False
    if close_ret < limit_ratio - 0.0025:
        return False
    if close_price is None or high_price is None or pd.isna(close_price) or pd.isna(high_price):
        return False
    return abs(float(high_price) - float(close_price)) <= max(0.01, abs(float(close_price)) * 0.001)


def determine_dynamic_exit(
    file_symbol: str,
    name: str,
    signal_limit_up: bool,
    entry_day_idx: int,
    daily_df: pd.DataFrame,
    m5_df: pd.DataFrame,
) -> dict[str, object]:
    result = {
        "exit_date": pd.NA,
        "exit_time": pd.NA,
        "exit_price": pd.NA,
        "exit_reason": pd.NA,
        "exit_first_5m_open": pd.NA,
        "exit_first_5m_close": pd.NA,
        "exit_first_5m_green": pd.NA,
        "exit_day_is_limit_up": pd.NA,
        "strategy_return": pd.NA,
        "holding_days_after_entry": pd.NA,
        "gross_max_gain": pd.NA,
        "gross_max_drawdown": pd.NA,
    }

    sellable_idx = entry_day_idx + 1
    if sellable_idx >= len(daily_df):
        return result

    if signal_limit_up:
        for exit_idx in range(sellable_idx, len(daily_df)):
            exit_bar = daily_df.iloc[exit_idx]
            prev_close = daily_df.iloc[exit_idx - 1]["close"] if exit_idx - 1 >= 0 else pd.NA
            exit_is_limit_up = is_limit_up_day(
                file_symbol=file_symbol,
                name=name,
                close_price=exit_bar["close"],
                high_price=exit_bar["high"],
                prev_close=prev_close,
            )
            if exit_is_limit_up:
                continue

            exit_date = pd.Timestamp(daily_df.index[exit_idx]).normalize()
            hold_slice = daily_df.iloc[entry_day_idx : exit_idx + 1]
            result.update(
                {
                    "exit_date": exit_date.date().isoformat(),
                    "exit_time": f"{exit_date.date().isoformat()} 15:00:00",
                    "exit_price": float(exit_bar["close"]),
                    "exit_reason": "signal_limit_up_hold_until_first_non_limit_close",
                    "exit_day_is_limit_up": False,
                    "strategy_return": compute_return(exit_bar["close"], daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
                    "holding_days_after_entry": int(exit_idx - entry_day_idx),
                    "gross_max_gain": compute_return(hold_slice["high"].max(), daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
                    "gross_max_drawdown": compute_return(hold_slice["low"].min(), daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
                }
            )
            return result
        return result

    exit_idx = sellable_idx
    exit_date = pd.Timestamp(daily_df.index[exit_idx]).normalize()
    exit_bar = daily_df.iloc[exit_idx]
    first_5m_bar = locate_first_5m_bar_for_day(m5_df, exit_date)
    if first_5m_bar is None:
        return result

    first_5m_open = float(first_5m_bar["open"])
    first_5m_close = float(first_5m_bar["close"])
    first_5m_green = first_5m_close > first_5m_open
    if first_5m_green:
        exit_price = float(exit_bar["close"])
        exit_time = f"{exit_date.date().isoformat()} 15:00:00"
        exit_reason = "sellable_day_close_after_first_5m_bull"
    else:
        exit_price = first_5m_close
        exit_time = pd.Timestamp(first_5m_bar.name).isoformat(sep=" ")
        exit_reason = "sellable_day_first_5m_bear_close"

    hold_slice = daily_df.iloc[entry_day_idx : exit_idx + 1]
    result.update(
        {
            "exit_date": exit_date.date().isoformat(),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_first_5m_open": first_5m_open,
            "exit_first_5m_close": first_5m_close,
            "exit_first_5m_green": bool(first_5m_green),
            "exit_day_is_limit_up": is_limit_up_day(
                file_symbol=file_symbol,
                name=name,
                close_price=exit_bar["close"],
                high_price=exit_bar["high"],
                prev_close=daily_df.iloc[exit_idx - 1]["close"] if exit_idx - 1 >= 0 else pd.NA,
            ),
            "strategy_return": compute_return(exit_price, daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
            "holding_days_after_entry": int(exit_idx - entry_day_idx),
            "gross_max_gain": compute_return(hold_slice["high"].max(), daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
            "gross_max_drawdown": compute_return(hold_slice["low"].min(), daily_df.iloc[entry_day_idx]["entry_price_proxy"]),
        }
    )
    return result


def build_trade_row(
    file_symbol: str,
    name: str,
    signal_date: pd.Timestamp,
    signal_idx: int,
    signal_bar: pd.Series,
    signal_analysis: pd.Series,
    weekly_info: dict[str, object],
    daily_df: pd.DataFrame,
    m5_df: pd.DataFrame,
) -> dict[str, object] | None:
    entry_day_idx = signal_idx + 1
    if entry_day_idx >= len(daily_df):
        return None

    entry_date = pd.Timestamp(daily_df.index[entry_day_idx]).normalize()
    entry_day_bar = daily_df.iloc[entry_day_idx]
    first_5m_bar = locate_first_5m_bar_for_day(m5_df, entry_date)
    if first_5m_bar is None:
        return None

    prev_day_high = float(signal_bar["high"])
    prev_day_close = float(signal_bar["close"])
    entry_bar_open = float(first_5m_bar["open"])
    entry_bar_close = float(first_5m_bar["close"])
    entry_bar_high = float(first_5m_bar["high"])
    entry_bar_low = float(first_5m_bar["low"])
    entry_bar_time = pd.Timestamp(first_5m_bar.name)
    entry_5m_return = compute_return(entry_bar_close, prev_day_close)
    if entry_5m_return is None:
        return None

    passed_entry = (
        entry_bar_close > entry_bar_open
        and entry_5m_return <= 0.04
        and entry_bar_close > prev_day_high
    )
    if not passed_entry:
        return None

    signal_prev_close = daily_df.iloc[signal_idx - 1]["close"] if signal_idx - 1 >= 0 else pd.NA
    signal_limit_up = is_limit_up_day(
        file_symbol=file_symbol,
        name=name,
        close_price=signal_bar["close"],
        high_price=signal_bar["high"],
        prev_close=signal_prev_close,
    )

    # Reuse the daily frame to carry the entry price into the exit helper without threading extra args.
    daily_work_df = daily_df.copy()
    daily_work_df["entry_price_proxy"] = pd.NA
    daily_work_df.iloc[entry_day_idx, daily_work_df.columns.get_loc("entry_price_proxy")] = entry_bar_close

    row: dict[str, object] = {
        "file_symbol": file_symbol,
        "name": name,
        "signal_week": first_monday_of_week(signal_date).date().isoformat(),
        "signal_date": signal_date.date().isoformat(),
        "entry_date": entry_date.date().isoformat(),
        "entry_time": entry_bar_time.isoformat(sep=" "),
        "entry_price": entry_bar_close,
        "entry_bar_open": entry_bar_open,
        "entry_bar_close": entry_bar_close,
        "entry_bar_high": entry_bar_high,
        "entry_bar_low": entry_bar_low,
        "entry_5m_return_vs_signal_close": entry_5m_return,
        "entry_5m_bar_green": True,
        "entry_break_prev_day_high": True,
        "entry_day_open": float(entry_day_bar["open"]),
        "entry_day_high": float(entry_day_bar["high"]),
        "entry_day_low": float(entry_day_bar["low"]),
        "entry_day_close": float(entry_day_bar["close"]),
        "entry_day_close_return": compute_return(entry_day_bar["close"], entry_bar_close),
        "entry_daily_gap_vs_signal_close": compute_return(entry_day_bar["open"], prev_day_close),
        "signal_open": float(signal_bar["open"]),
        "signal_high": prev_day_high,
        "signal_low": float(signal_bar["low"]),
        "signal_close": prev_day_close,
        "signal_day_return": compute_return(signal_bar["close"], signal_bar["open"]),
        "signal_limit_up": bool(signal_limit_up),
        "signal_volume": float(signal_bar["volume"]) if pd.notna(signal_bar.get("volume", pd.NA)) else pd.NA,
        "kong_turn_bull_pattern": signal_analysis.get("kong_turn_bull_pattern"),
        "prev2_time": pd.Timestamp(signal_analysis["prev2_time"]).date().isoformat() if pd.notna(signal_analysis["prev2_time"]) else pd.NA,
        "prev1_time": pd.Timestamp(signal_analysis["prev1_time"]).date().isoformat() if pd.notna(signal_analysis["prev1_time"]) else pd.NA,
        "current_time": pd.Timestamp(signal_analysis["current_time"]).date().isoformat() if pd.notna(signal_analysis["current_time"]) else pd.NA,
        "prev2_low": float(signal_analysis["prev2_low"]) if pd.notna(signal_analysis["prev2_low"]) else pd.NA,
        "prev1_low": float(signal_analysis["prev1_low"]) if pd.notna(signal_analysis["prev1_low"]) else pd.NA,
        "prev1_high": float(signal_analysis["prev1_high"]) if pd.notna(signal_analysis["prev1_high"]) else pd.NA,
        "three_day_kong_turn_bull": bool(signal_analysis["three_day_kong_turn_bull"]),
        "two_day_kong_turn_bull": bool(signal_analysis["two_day_kong_turn_bull"]),
        "today_bullish_close": bool(signal_analysis["today_bullish_close"]),
        "today_close_breaks_yesterday_high": bool(signal_analysis["today_close_breaks_yesterday_high"]),
        "today_higher_high": bool(signal_analysis["today_higher_high"]),
        "today_lower_low": bool(signal_analysis["today_lower_low"]),
        "yesterday_lower_low": bool(signal_analysis["yesterday_lower_low"]),
        **weekly_info,
    }

    row.update(
        determine_dynamic_exit(
            file_symbol=file_symbol,
            name=name,
            signal_limit_up=bool(signal_limit_up),
            entry_day_idx=entry_day_idx,
            daily_df=daily_work_df,
            m5_df=m5_df,
        )
    )
    row["strategy_win"] = bool(row["strategy_return"] > 0) if pd.notna(row["strategy_return"]) else pd.NA
    return row


def run_backtest(
    data_root: Path,
    signal_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    weekly_root = data_root / "weekly"
    daily_root = data_root / "daily"
    m5_root = data_root / "5m"

    weekly_cfg = weekly_bullish_config()
    kong_cfg = kong_turn_bull_config()
    symbol_paths = sorted(daily_root.glob("*.parquet"))
    if limit is not None:
        symbol_paths = symbol_paths[:limit]

    weekly_pool_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    processed_symbols = 0
    missing_weekly = 0
    missing_5m = 0
    skipped_short_daily = 0

    for daily_path in symbol_paths:
        file_symbol = daily_path.stem
        weekly_path = weekly_root / daily_path.name
        m5_path = m5_root / daily_path.name
        if not weekly_path.exists():
            missing_weekly += 1
            continue
        if not m5_path.exists():
            missing_5m += 1
            continue

        daily_df = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
        if daily_df.empty:
            skipped_short_daily += 1
            continue
        daily_df = daily_df.loc[: str(analysis_end.date())].copy()
        daily_df = daily_df.loc[str(signal_start.date() - pd.Timedelta(days=30)) :]
        if len(daily_df) < 10:
            skipped_short_daily += 1
            continue

        weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        weekly_df = dedupe_to_last_bar_per_week(weekly_df)
        if weekly_df.empty:
            continue
        weekly_df = weekly_df.loc[: str(analysis_end.date())].copy()

        m5_df = pd.read_parquet(m5_path, columns=M5_COLUMNS).sort_index()
        if m5_df.empty:
            continue
        m5_df = m5_df.loc[(m5_df.index >= signal_start) & (m5_df.index < analysis_end + pd.Timedelta(days=10))].copy()
        if m5_df.empty:
            continue

        weekly_signal_mask = detect_bullish_up(weekly_df, config=weekly_cfg).fillna(False)
        weekly_pool_map = build_weekly_pool_map(
            weekly_df=weekly_df,
            signal_mask=weekly_signal_mask,
            analysis_end=analysis_end,
        )
        if not weekly_pool_map:
            processed_symbols += 1
            continue

        daily_df = daily_df.loc[daily_df.index >= min(weekly_pool_map.keys())].copy()
        if daily_df.empty:
            processed_symbols += 1
            continue

        daily_analysis = analyze_kong_turn_bull(daily_df, config=kong_cfg)
        daily_dates = normalize_daily_index(daily_df)
        name_series = daily_df["name"].dropna()
        name = str(name_series.iloc[-1]) if not name_series.empty else file_symbol

        for idx in range(len(daily_df)):
            signal_date = daily_dates[idx]
            if signal_date < signal_start or signal_date > analysis_end:
                continue

            week_start = first_monday_of_week(signal_date)
            weekly_info = weekly_pool_map.get(week_start)
            if weekly_info is None:
                continue

            pool_key = week_start.date().isoformat()
            weekly_pool_rows.append(
                {
                    "signal_week": pool_key,
                    "file_symbol": file_symbol,
                    "name": name,
                    **weekly_info,
                }
            )

            analysis_row = daily_analysis.iloc[idx]
            if not bool(analysis_row["kong_turn_bull"]):
                continue

            signal_bar = daily_df.iloc[idx]
            signal_rows.append(
                {
                    "signal_week": pool_key,
                    "signal_date": signal_date.date().isoformat(),
                    "file_symbol": file_symbol,
                    "name": name,
                    "kong_turn_bull_pattern": analysis_row.get("kong_turn_bull_pattern"),
                    "signal_open": float(signal_bar["open"]),
                    "signal_high": float(signal_bar["high"]),
                    "signal_low": float(signal_bar["low"]),
                    "signal_close": float(signal_bar["close"]),
                    **weekly_info,
                }
            )

            trade_row = build_trade_row(
                file_symbol=file_symbol,
                name=name,
                signal_date=signal_date,
                signal_idx=idx,
                signal_bar=signal_bar,
                signal_analysis=analysis_row,
                weekly_info=weekly_info,
                daily_df=daily_df,
                m5_df=m5_df,
            )
            if trade_row is not None:
                trade_rows.append(trade_row)

        processed_symbols += 1
        if processed_symbols % 200 == 0:
            print(f"processed_symbols={processed_symbols}")

    weekly_pool_df = pd.DataFrame(weekly_pool_rows)
    if not weekly_pool_df.empty:
        weekly_pool_df = weekly_pool_df.drop_duplicates(["signal_week", "file_symbol"]).sort_values(["signal_week", "file_symbol"]).reset_index(drop=True)

    signal_df = pd.DataFrame(signal_rows)
    if not signal_df.empty:
        signal_df = signal_df.sort_values(["signal_date", "file_symbol"]).reset_index(drop=True)

    trade_df = pd.DataFrame(trade_rows)
    if not trade_df.empty:
        trade_df = trade_df.sort_values(["signal_date", "entry_time", "file_symbol"]).reset_index(drop=True)

    daily_summary_df = summarize_by_date(trade_df)
    weekly_summary_df = summarize_by_week(trade_df)

    meta = {
        "processed_symbols": processed_symbols,
        "missing_weekly": missing_weekly,
        "missing_5m": missing_5m,
        "skipped_short_daily": skipped_short_daily,
        "signal_start": signal_start.date().isoformat(),
        "analysis_end": analysis_end.date().isoformat(),
    }
    return weekly_pool_df, signal_df, trade_df, daily_summary_df, weekly_summary_df, meta


def summarize_by_date(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        row = {
            "signal_date": signal_date,
            "trade_count": int(len(group)),
            "unique_symbols": int(group["file_symbol"].nunique()),
            "avg_entry_5m_return_vs_signal_close": safe_mean(group["entry_5m_return_vs_signal_close"]),
            "signal_limit_up_count": int(group["signal_limit_up"].fillna(False).sum()),
        }
        stats = summarize_return_series(group["strategy_return"])
        for key, value in stats.items():
            row[f"strategy_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def summarize_by_week(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for signal_week, group in trade_df.groupby("signal_week", sort=True):
        row = {
            "signal_week": signal_week,
            "trade_count": int(len(group)),
            "unique_symbols": int(group["file_symbol"].nunique()),
            "trade_days": int(group["signal_date"].nunique()),
            "avg_entry_5m_return_vs_signal_close": safe_mean(group["entry_5m_return_vs_signal_close"]),
            "signal_limit_up_count": int(group["signal_limit_up"].fillna(False).sum()),
        }
        stats = summarize_return_series(group["strategy_return"])
        for key, value in stats.items():
            row[f"strategy_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_week").reset_index(drop=True)


def summarize_overall(trade_df: pd.DataFrame, signal_df: pd.DataFrame, weekly_pool_df: pd.DataFrame) -> pd.DataFrame:
    base = {
        "weekly_pool_weeks": int(weekly_pool_df["signal_week"].nunique()) if not weekly_pool_df.empty else 0,
        "weekly_pool_rows": int(len(weekly_pool_df)) if not weekly_pool_df.empty else 0,
        "weekly_pool_unique_symbols": int(weekly_pool_df["file_symbol"].nunique()) if not weekly_pool_df.empty else 0,
        "daily_signal_rows": int(len(signal_df)) if not signal_df.empty else 0,
        "daily_signal_unique_symbols": int(signal_df["file_symbol"].nunique()) if not signal_df.empty else 0,
        "entry_trade_rows": int(len(trade_df)) if not trade_df.empty else 0,
        "entry_trade_unique_symbols": int(trade_df["file_symbol"].nunique()) if not trade_df.empty else 0,
        "signal_limit_up_trade_count": int(trade_df["signal_limit_up"].fillna(False).sum()) if not trade_df.empty else 0,
    }
    row = {"holding_period": "dynamic_exit", **base}
    row.update(summarize_return_series(trade_df["strategy_return"] if not trade_df.empty else pd.Series(dtype=float)))
    return pd.DataFrame([row])


def feature_bucket_analysis(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    work_df = trade_df.copy()
    work_df["entry_5m_return_bucket"] = bucketize_feature(
        work_df["entry_5m_return_vs_signal_close"],
        edges=[-1.0, 0.005, 0.015, 0.025, 0.04],
        labels=["<=0.5%", "0.5%-1.5%", "1.5%-2.5%", "2.5%-4%"],
    )
    work_df["weekly_breakout_bucket"] = bucketize_feature(
        work_df["weekly_breakout_close_vs_prev_high"],
        edges=[-1.0, 0.01, 0.03, 0.06, 1.0],
        labels=["<=1%", "1%-3%", "3%-6%", ">6%"],
    )
    work_df["signal_day_return_bucket"] = bucketize_feature(
        work_df["signal_day_return"],
        edges=[-1.0, 0.0, 0.03, 0.06, 1.0],
        labels=["<=0%", "0%-3%", "3%-6%", ">6%"],
    )

    result_frames: list[pd.DataFrame] = []
    for feature in [
        "kong_turn_bull_pattern",
        "entry_5m_return_bucket",
        "weekly_breakout_bucket",
        "signal_day_return_bucket",
        "three_day_kong_turn_bull",
        "two_day_kong_turn_bull",
        "signal_limit_up",
    ]:
        rows: list[dict[str, object]] = []
        grouped = work_df.dropna(subset=[feature]).groupby(feature, observed=False, sort=False)
        for value, group in grouped:
            stats = summarize_return_series(group["strategy_return"])
            rows.append({"feature": feature, "bucket": str(value), **stats})
        if rows:
            result_frames.append(pd.DataFrame(rows))

    if not result_frames:
        return pd.DataFrame()
    return pd.concat(result_frames, ignore_index=True).sort_values(
        ["feature", "sample_count", "avg_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def compare_filtered_vs_all(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    filters: list[tuple[str, pd.Series]] = [
        ("all_trades", pd.Series(True, index=trade_df.index)),
        ("entry_5m_le_2pct", pd.to_numeric(trade_df["entry_5m_return_vs_signal_close"], errors="coerce") <= 0.02),
        ("entry_5m_le_1pct", pd.to_numeric(trade_df["entry_5m_return_vs_signal_close"], errors="coerce") <= 0.01),
        ("three_day_only", trade_df["three_day_kong_turn_bull"].fillna(False)),
        ("two_day_only", trade_df["two_day_kong_turn_bull"].fillna(False)),
        ("pattern_both", trade_df["kong_turn_bull_pattern"].astype(str) == "both"),
        ("weekly_breakout_le_3pct", pd.to_numeric(trade_df["weekly_breakout_close_vs_prev_high"], errors="coerce") <= 0.03),
        ("signal_day_return_le_3pct", pd.to_numeric(trade_df["signal_day_return"], errors="coerce") <= 0.03),
        ("signal_close_break_yes", trade_df["today_close_breaks_yesterday_high"].fillna(False)),
        ("signal_limit_up_yes", trade_df["signal_limit_up"].fillna(False)),
    ]

    rows: list[dict[str, object]] = []
    for label, mask in filters:
        filtered = trade_df.loc[mask.fillna(False)].copy()
        stats = summarize_return_series(filtered["strategy_return"] if not filtered.empty else pd.Series(dtype=float))
        rows.append(
            {
                "filter_name": label,
                "sample_count": int(len(filtered)),
                "sample_ratio": float(len(filtered) / len(trade_df)) if len(trade_df) else pd.NA,
                "win_rate": stats["win_rate"],
                "avg_return": stats["avg_return"],
                "expectancy": stats["expectancy"],
                "profit_factor": stats["profit_factor"],
            }
        )
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def top_trade_samples(trade_df: pd.DataFrame, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trade_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    work_df = trade_df.dropna(subset=["strategy_return"]).copy()
    if work_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    winners = work_df.sort_values("strategy_return", ascending=False).head(top_n).reset_index(drop=True)
    losers = work_df.sort_values("strategy_return", ascending=True).head(top_n).reset_index(drop=True)
    return winners, losers


def build_report(
    trade_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    weekly_pool_df: pd.DataFrame,
    daily_summary_df: pd.DataFrame,
    weekly_summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    filtered_compare_df: pd.DataFrame,
    meta: dict[str, object],
    output_paths: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# 周线多头 + 日线空转多 + 5分钟买点 + 动态卖出 回测报告")
    lines.append("")
    lines.append("## 回测口径")
    lines.append("")
    lines.append(f"- 回测区间：{meta['signal_start']} 到 {meta['analysis_end']}")
    lines.append("- 买点：信号次日第一根 5 分钟收阳、涨幅不超过 4%、且收盘站上信号日最高价时，在该 5 分钟收盘价买入。")
    lines.append("- 常规卖法：买入后的第一个可卖交易日，若首根 5 分钟收阴，则在该 5 分钟收盘价卖出；若首根 5 分钟收阳，则持有到当日收盘卖出。")
    lines.append("- 特殊卖法：如果信号日本身是涨停，则跳过上面的首根 5 分钟判断，至少持有到下一个可卖日收盘；之后一直持有到第一次非涨停日，并在该日收盘卖出。")
    lines.append("- 涨停判断：按常见 A 股涨停幅度近似处理，主板 10%，创业板/科创板 20%，北交所 30%，ST 5%。")
    lines.append("")

    row = overall_df.iloc[0] if not overall_df.empty else None
    lines.append("## 总体结果")
    lines.append("")
    if row is None:
        lines.append("- 没有有效交易样本。")
        lines.append("")
    else:
        pf = "inf" if row["profit_factor"] == math.inf else format_num(row["profit_factor"])
        lines.append(f"- 周线母池样本：{int(row['weekly_pool_rows'])}")
        lines.append(f"- 日线空转多信号：{int(row['daily_signal_rows'])}")
        lines.append(f"- 真正触发买入的交易：{int(row['entry_trade_rows'])}")
        lines.append(f"- 其中信号日涨停样本：{int(row['signal_limit_up_trade_count'])}")
        lines.append(f"- 胜率：{format_pct(row['win_rate'])}")
        lines.append(f"- 平均涨跌幅：{format_pct(row['avg_return'])}")
        lines.append(f"- 中位数涨跌幅：{format_pct(row['median_return'])}")
        lines.append(f"- 平均盈利：{format_pct(row['avg_gain'])}")
        lines.append(f"- 平均亏损：{format_pct(row['avg_loss'])}")
        lines.append(f"- 期望：{format_pct(row['expectancy'])}")
        lines.append(f"- Profit Factor：{pf}")
        lines.append(f"- Kelly 粗估：{format_pct(row['kelly_fraction'])}")
        lines.append(f"- 单笔最大收益：{format_pct(row['max_return'])}")
        lines.append(f"- 单笔最大亏损：{format_pct(row['min_return'])}")
        lines.append(f"- 顺序资金曲线最大回撤：{format_pct(row['max_drawdown_seq'])}")
        lines.append("")

    if not weekly_summary_df.empty:
        best_week = weekly_summary_df.sort_values("strategy_avg_return", ascending=False).iloc[0]
        worst_week = weekly_summary_df.sort_values("strategy_avg_return", ascending=True).iloc[0]
        lines.append("## 每周情况")
        lines.append("")
        lines.append(
            f"- 最强周：{best_week['signal_week']}，{int(best_week['trade_count'])} 笔，胜率 {format_pct(best_week['strategy_win_rate'])}，平均 {format_pct(best_week['strategy_avg_return'])}。"
        )
        lines.append(
            f"- 最弱周：{worst_week['signal_week']}，{int(worst_week['trade_count'])} 笔，胜率 {format_pct(worst_week['strategy_win_rate'])}，平均 {format_pct(worst_week['strategy_avg_return'])}。"
        )
        lines.append("")

    if not daily_summary_df.empty:
        best_day = daily_summary_df.sort_values("strategy_avg_return", ascending=False).iloc[0]
        worst_day = daily_summary_df.sort_values("strategy_avg_return", ascending=True).iloc[0]
        lines.append("## 每日情况")
        lines.append("")
        lines.append(
            f"- 最强信号日：{best_day['signal_date']}，{int(best_day['trade_count'])} 笔，胜率 {format_pct(best_day['strategy_win_rate'])}，平均 {format_pct(best_day['strategy_avg_return'])}。"
        )
        lines.append(
            f"- 最弱信号日：{worst_day['signal_date']}，{int(worst_day['trade_count'])} 笔，胜率 {format_pct(worst_day['strategy_win_rate'])}，平均 {format_pct(worst_day['strategy_avg_return'])}。"
        )
        lines.append("")

    if not filtered_compare_df.empty:
        top_filters = filtered_compare_df.sort_values(["expectancy", "sample_count"], ascending=[False, False]).head(8)
        lines.append("## 可优化方向")
        lines.append("")
        lines.append("- 过滤条件里相对更好的组合：")
        for filter_row in top_filters.itertuples(index=False):
            pf_text = "inf" if filter_row.profit_factor == math.inf else format_num(filter_row.profit_factor)
            lines.append(
                f"- {filter_row.filter_name} | 样本 {filter_row.sample_count} | 胜率 {format_pct(filter_row.win_rate)} | 平均 {format_pct(filter_row.avg_return)} | 期望 {format_pct(filter_row.expectancy)} | PF {pf_text}"
            )
        lines.append("")

    if not feature_df.empty:
        lines.append("## 分层表现")
        lines.append("")
        work_df = feature_df[feature_df["sample_count"] >= 10].copy()
        if work_df.empty:
            work_df = feature_df.copy()
        best_rows = work_df.sort_values(["expectancy", "sample_count"], ascending=[False, False]).head(12)
        for feature_row in best_rows.itertuples(index=False):
            pf_text = "inf" if feature_row.profit_factor == math.inf else format_num(feature_row.profit_factor)
            lines.append(
                f"- {feature_row.feature} = {feature_row.bucket} | 样本 {feature_row.sample_count} | 胜率 {format_pct(feature_row.win_rate)} | 平均 {format_pct(feature_row.avg_return)} | 期望 {format_pct(feature_row.expectancy)} | PF {pf_text}"
            )
        lines.append("")

    if not trade_df.empty:
        winners, losers = top_trade_samples(trade_df, top_n=10)
        lines.append("## 强弱样本")
        lines.append("")
        lines.append("### 最强样本")
        lines.append("")
        for sample in winners.itertuples(index=False):
            lines.append(
                f"- {sample.signal_date} | {sample.file_symbol} {sample.name} | 入场 {sample.entry_time} | 离场 {sample.exit_time} | 理由 {sample.exit_reason} | 收益 {format_pct(sample.strategy_return)}"
            )
        lines.append("")
        lines.append("### 最弱样本")
        lines.append("")
        for sample in losers.itertuples(index=False):
            lines.append(
                f"- {sample.signal_date} | {sample.file_symbol} {sample.name} | 入场 {sample.entry_time} | 离场 {sample.exit_time} | 理由 {sample.exit_reason} | 收益 {format_pct(sample.strategy_return)}"
            )
        lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_start = pd.Timestamp(args.signal_start).normalize()
    if args.analysis_end is None:
        sample_daily = next(sorted((Path(args.data_root) / "daily").glob("*.parquet")))
        sample_df = pd.read_parquet(sample_daily, columns=["close"]).sort_index()
        analysis_end = pd.Timestamp(sample_df.index.max()).normalize()
    else:
        analysis_end = pd.Timestamp(args.analysis_end).normalize()

    weekly_pool_df, signal_df, trade_df, daily_summary_df, weekly_summary_df, meta = run_backtest(
        data_root=Path(args.data_root),
        signal_start=signal_start,
        analysis_end=analysis_end,
        limit=args.limit,
    )

    overall_df = summarize_overall(trade_df, signal_df, weekly_pool_df)
    feature_df = feature_bucket_analysis(trade_df)
    filtered_compare_df = compare_filtered_vs_all(trade_df)

    suffix = f"{signal_start.date().isoformat()}_to_{analysis_end.date().isoformat()}"
    output_paths = {
        "weekly_pool": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_weekly_pool_{suffix}.csv",
        "signals": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_signals_{suffix}.csv",
        "trades": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_trades_{suffix}.csv",
        "daily_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_daily_summary_{suffix}.csv",
        "weekly_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_weekly_summary_{suffix}.csv",
        "overall_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_overall_summary_{suffix}.csv",
        "feature_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_feature_summary_{suffix}.csv",
        "filtered_compare": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_filtered_compare_{suffix}.csv",
        "report": output_dir / f"weekly_bullish_kong_turn_bull_5m_dynamic_exit_report_{suffix}.md",
    }

    weekly_pool_df.to_csv(output_paths["weekly_pool"], index=False, encoding="utf-8-sig")
    signal_df.to_csv(output_paths["signals"], index=False, encoding="utf-8-sig")
    trade_df.to_csv(output_paths["trades"], index=False, encoding="utf-8-sig")
    daily_summary_df.to_csv(output_paths["daily_summary"], index=False, encoding="utf-8-sig")
    weekly_summary_df.to_csv(output_paths["weekly_summary"], index=False, encoding="utf-8-sig")
    overall_df.to_csv(output_paths["overall_summary"], index=False, encoding="utf-8-sig")
    feature_df.to_csv(output_paths["feature_summary"], index=False, encoding="utf-8-sig")
    filtered_compare_df.to_csv(output_paths["filtered_compare"], index=False, encoding="utf-8-sig")
    output_paths["report"].write_text(
        build_report(
            trade_df=trade_df,
            signal_df=signal_df,
            weekly_pool_df=weekly_pool_df,
            daily_summary_df=daily_summary_df,
            weekly_summary_df=weekly_summary_df,
            overall_df=overall_df,
            feature_df=feature_df,
            filtered_compare_df=filtered_compare_df,
            meta=meta,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )

    print(f"weekly_pool_rows={len(weekly_pool_df)}")
    print(f"daily_signal_rows={len(signal_df)}")
    print(f"trade_rows={len(trade_df)}")
    print(f"report_path={output_paths['report']}")
    if not overall_df.empty:
        print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
