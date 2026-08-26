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

from patterns import BullishUpConfig, KongTurnBullConfig, analyze_kong_turn_bull, detect_bullish_up
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
WEEKLY_ROOT = DATA_ROOT / "weekly"
DAILY_ROOT = DATA_ROOT / "daily"
M5_ROOT = DATA_ROOT / "5m"

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
M5_COLUMNS = [
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
    "amount",
    "turnover",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest: weekly bullish pool -> daily kong-turn-bull -> next-day first 5m bullish "
            "breakout confirmation entry."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--signal-start", default="2026-01-01", help="First signal date to include.")
    parser.add_argument(
        "--analysis-end",
        default=None,
        help="Last calendar date to include. Default uses latest available daily bar.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for quick testing.")
    parser.add_argument("--output-dir", type=Path, default=SCREENING_RESULTS_DIR, help="Directory for CSV/report outputs.")
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


def kong_turn_bull_config() -> KongTurnBullConfig:
    return KongTurnBullConfig(
        require_yesterday_lower_low=True,
        require_today_higher_high=True,
        require_three_day_bullish_close=True,
        enable_three_day_pattern=True,
        enable_two_day_close_breakout=True,
    )


def dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def safe_median(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def safe_std(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.std(ddof=0))


def safe_sum(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.sum())


def compute_return(current: float | int | None, base: float | int | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base):
        return None
    base_float = float(base)
    if abs(base_float) <= 1e-9:
        return None
    return float(current) / base_float - 1.0


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def format_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def expectancy(series: pd.Series) -> tuple[float | None, float | None, float | None, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None, None, None, None
    win_mask = numeric > 0
    loss_mask = numeric <= 0
    win_rate = float(win_mask.mean())
    avg_gain = float(numeric.loc[win_mask].mean()) if win_mask.any() else None
    avg_loss = float(numeric.loc[loss_mask].mean()) if loss_mask.any() else None
    exp = 0.0
    if avg_gain is not None:
        exp += win_rate * avg_gain
    if avg_loss is not None:
        exp += (1.0 - win_rate) * avg_loss
    return win_rate, avg_gain, avg_loss, exp


def profit_factor(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    gross_profit = float(numeric[numeric > 0].sum()) if (numeric > 0).any() else 0.0
    gross_loss = float((-numeric[numeric < 0]).sum()) if (numeric < 0).any() else 0.0
    if gross_loss <= 1e-12:
        if gross_profit > 0:
            return math.inf
        return None
    return gross_profit / gross_loss


def kelly_fraction(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    pos = numeric[numeric > 0]
    neg = -numeric[numeric < 0]
    if pos.empty or neg.empty:
        return None
    win_rate = float((numeric > 0).mean())
    loss_rate = 1.0 - win_rate
    avg_gain = float(pos.mean())
    avg_loss = float(neg.mean())
    if avg_loss <= 1e-12:
        return None
    odds = avg_gain / avg_loss
    if odds <= 1e-12:
        return None
    return win_rate - loss_rate / odds


def max_drawdown_from_returns(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    equity = (1.0 + numeric).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def first_monday_of_week(timestamp: pd.Timestamp) -> pd.Timestamp:
    normalized = pd.Timestamp(timestamp).normalize()
    return normalized - pd.Timedelta(days=normalized.weekday())


def build_weekly_pool_map(
    weekly_df: pd.DataFrame,
    signal_mask: pd.Series,
    analysis_end: pd.Timestamp,
) -> dict[pd.Timestamp, dict[str, object]]:
    result: dict[pd.Timestamp, dict[str, object]] = {}
    if weekly_df.empty:
        return result

    dates = pd.to_datetime(weekly_df.index).normalize()
    latest_allowed = first_monday_of_week(analysis_end)
    for idx in range(2, len(weekly_df)):
        week_start = dates[idx]
        signal_idx = idx - 1
        if week_start > latest_allowed:
            continue
        if not bool(signal_mask.iloc[signal_idx]):
            continue
        bar = weekly_df.iloc[signal_idx]
        prev_bar = weekly_df.iloc[signal_idx - 1]
        result[week_start] = {
            "pool_effective_week_start": week_start.date().isoformat(),
            "weekly_signal_time": dates[signal_idx].date().isoformat(),
            "weekly_time": week_start.date().isoformat(),
            "weekly_open": float(bar["open"]),
            "weekly_high": float(bar["high"]),
            "weekly_low": float(bar["low"]),
            "weekly_close": float(bar["close"]),
            "weekly_prev_high": float(prev_bar["high"]),
            "weekly_prev_low": float(prev_bar["low"]),
            "weekly_prev_close": float(prev_bar["close"]),
            "weekly_breakout_close_vs_prev_high": compute_return(bar["close"], prev_bar["high"]),
            "weekly_breakout_high_vs_prev_high": compute_return(bar["high"], prev_bar["high"]),
            "weekly_week_return": compute_return(bar["close"], bar["open"]),
        }
    return result


def normalize_daily_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(df.index).normalize()


def locate_first_5m_bar_for_day(df: pd.DataFrame, trade_date: pd.Timestamp) -> pd.Series | None:
    same_day = df.loc[(df.index >= trade_date) & (df.index < trade_date + pd.Timedelta(days=1))]
    if same_day.empty:
        return None
    return same_day.iloc[0]


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
    next_day_idx = signal_idx + 1
    if next_day_idx >= len(daily_df):
        return None

    next_trade_date = pd.Timestamp(daily_df.index[next_day_idx]).normalize()
    next_bar = daily_df.iloc[next_day_idx]

    first_5m_bar = locate_first_5m_bar_for_day(m5_df, next_trade_date)
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

    entry_price = entry_bar_close

    row: dict[str, object] = {
        "file_symbol": file_symbol,
        "name": name,
        "signal_week": first_monday_of_week(signal_date).date().isoformat(),
        "signal_date": signal_date.date().isoformat(),
        "next_trade_date": next_trade_date.date().isoformat(),
        "entry_time": entry_bar_time.isoformat(sep=" "),
        "entry_price": entry_price,
        "entry_bar_open": entry_bar_open,
        "entry_bar_close": entry_bar_close,
        "entry_bar_high": entry_bar_high,
        "entry_bar_low": entry_bar_low,
        "entry_5m_return_vs_signal_close": entry_5m_return,
        "entry_5m_bar_green": True,
        "entry_break_prev_day_high": True,
        "entry_daily_gap_vs_signal_close": compute_return(next_bar["open"], prev_day_close),
        "signal_open": float(signal_bar["open"]),
        "signal_high": prev_day_high,
        "signal_low": float(signal_bar["low"]),
        "signal_close": prev_day_close,
        "signal_day_return": compute_return(signal_bar["close"], signal_bar["open"]),
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

    day_after_entry_ret = compute_return(next_bar["close"], entry_price)
    row["entry_day_close"] = float(next_bar["close"])
    row["entry_day_high"] = float(next_bar["high"])
    row["entry_day_low"] = float(next_bar["low"])
    row["entry_day_return_close"] = day_after_entry_ret
    row["entry_day_max_gain"] = compute_return(next_bar["high"], entry_price)
    row["entry_day_max_drawdown"] = compute_return(next_bar["low"], entry_price)

    for hold_days in (1, 2):
        exit_idx = next_day_idx + hold_days
        if exit_idx >= len(daily_df):
            row[f"hold_{hold_days}d_exit_date"] = pd.NA
            row[f"hold_{hold_days}d_exit_close"] = pd.NA
            row[f"hold_{hold_days}d_return"] = pd.NA
            row[f"hold_{hold_days}d_win"] = pd.NA
            row[f"hold_{hold_days}d_max_gain"] = pd.NA
            row[f"hold_{hold_days}d_max_drawdown"] = pd.NA
            continue

        exit_date = pd.Timestamp(daily_df.index[exit_idx]).normalize()
        exit_bar = daily_df.iloc[exit_idx]
        hold_slice = daily_df.iloc[next_day_idx : exit_idx + 1]
        row[f"hold_{hold_days}d_exit_date"] = exit_date.date().isoformat()
        row[f"hold_{hold_days}d_exit_close"] = float(exit_bar["close"])
        hold_ret = compute_return(exit_bar["close"], entry_price)
        row[f"hold_{hold_days}d_return"] = hold_ret
        row[f"hold_{hold_days}d_win"] = bool(hold_ret > 0) if hold_ret is not None else pd.NA
        row[f"hold_{hold_days}d_max_gain"] = compute_return(hold_slice["high"].max(), entry_price)
        row[f"hold_{hold_days}d_max_drawdown"] = compute_return(hold_slice["low"].min(), entry_price)

    return row


def run_backtest(
    data_root: Path,
    signal_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
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
        m5_df = m5_df.loc[(m5_df.index >= signal_start) & (m5_df.index < analysis_end + pd.Timedelta(days=5))].copy()
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

            if signal_date + pd.Timedelta(days=1) > analysis_end + pd.Timedelta(days=5):
                continue

            pool_key = (week_start.date().isoformat(), file_symbol)
            weekly_pool_rows.append(
                {
                    "signal_week": pool_key[0],
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
                    "signal_week": week_start.date().isoformat(),
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


def summarize_return_series(series: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "sample_count": 0,
            "win_count": 0,
            "win_rate": pd.NA,
            "avg_return": pd.NA,
            "median_return": pd.NA,
            "avg_gain": pd.NA,
            "avg_loss": pd.NA,
            "expectancy": pd.NA,
            "profit_factor": pd.NA,
            "std_return": pd.NA,
            "max_return": pd.NA,
            "min_return": pd.NA,
            "sum_return": pd.NA,
            "kelly_fraction": pd.NA,
            "max_drawdown_seq": pd.NA,
        }
    win_rate, avg_gain, avg_loss, exp = expectancy(numeric)
    return {
        "sample_count": int(len(numeric)),
        "win_count": int((numeric > 0).sum()),
        "win_rate": win_rate,
        "avg_return": safe_mean(numeric),
        "median_return": safe_median(numeric),
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "expectancy": exp,
        "profit_factor": profit_factor(numeric),
        "std_return": safe_std(numeric),
        "max_return": float(numeric.max()),
        "min_return": float(numeric.min()),
        "sum_return": safe_sum(numeric),
        "kelly_fraction": kelly_fraction(numeric),
        "max_drawdown_seq": max_drawdown_from_returns(numeric),
    }


def summarize_by_date(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = trade_df.groupby("signal_date", sort=True)
    for signal_date, group in grouped:
        row: dict[str, object] = {
            "signal_date": signal_date,
            "trade_count": int(len(group)),
            "unique_symbols": int(group["file_symbol"].nunique()),
            "avg_entry_5m_return_vs_signal_close": safe_mean(group["entry_5m_return_vs_signal_close"]),
            "avg_signal_day_return": safe_mean(group["signal_day_return"]),
            "avg_weekly_breakout_close_vs_prev_high": safe_mean(group["weekly_breakout_close_vs_prev_high"]),
        }
        for hold_days in (1, 2):
            stats = summarize_return_series(group[f"hold_{hold_days}d_return"])
            for key, value in stats.items():
                row[f"hold_{hold_days}d_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def summarize_by_week(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = trade_df.groupby("signal_week", sort=True)
    for signal_week, group in grouped:
        row: dict[str, object] = {
            "signal_week": signal_week,
            "trade_count": int(len(group)),
            "unique_symbols": int(group["file_symbol"].nunique()),
            "trade_days": int(group["signal_date"].nunique()),
            "avg_entry_5m_return_vs_signal_close": safe_mean(group["entry_5m_return_vs_signal_close"]),
            "avg_signal_day_return": safe_mean(group["signal_day_return"]),
            "avg_weekly_breakout_close_vs_prev_high": safe_mean(group["weekly_breakout_close_vs_prev_high"]),
        }
        for hold_days in (1, 2):
            stats = summarize_return_series(group[f"hold_{hold_days}d_return"])
            for key, value in stats.items():
                row[f"hold_{hold_days}d_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_week").reset_index(drop=True)


def summarize_overall(trade_df: pd.DataFrame, signal_df: pd.DataFrame, weekly_pool_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = {
        "weekly_pool_weeks": int(weekly_pool_df["signal_week"].nunique()) if not weekly_pool_df.empty else 0,
        "weekly_pool_rows": int(len(weekly_pool_df)),
        "weekly_pool_unique_symbols": int(weekly_pool_df["file_symbol"].nunique()) if not weekly_pool_df.empty else 0,
        "daily_signal_rows": int(len(signal_df)),
        "daily_signal_unique_symbols": int(signal_df["file_symbol"].nunique()) if not signal_df.empty else 0,
        "entry_trade_rows": int(len(trade_df)),
        "entry_trade_unique_symbols": int(trade_df["file_symbol"].nunique()) if not trade_df.empty else 0,
    }
    for hold_days in (1, 2):
        row = {"holding_period": f"hold_{hold_days}d", **base}
        stats = summarize_return_series(trade_df[f"hold_{hold_days}d_return"] if not trade_df.empty else pd.Series(dtype=float))
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def bucketize_feature(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.cut(numeric, bins=edges, labels=labels, include_lowest=True, right=True)


def feature_bucket_analysis(trade_df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
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

    analyses: list[pd.DataFrame] = []
    for feature in [
        "kong_turn_bull_pattern",
        "entry_5m_return_bucket",
        "weekly_breakout_bucket",
        "signal_day_return_bucket",
        "three_day_kong_turn_bull",
        "two_day_kong_turn_bull",
    ]:
        grouped = work_df.dropna(subset=[feature]).groupby(feature, observed=False, sort=False)
        rows: list[dict[str, object]] = []
        for value, group in grouped:
            stats = summarize_return_series(group[f"hold_{hold_days}d_return"])
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(value),
                    **stats,
                }
            )
        if rows:
            analyses.append(pd.DataFrame(rows))

    if not analyses:
        return pd.DataFrame()
    result = pd.concat(analyses, ignore_index=True)
    return result.sort_values(["feature", "sample_count", "avg_return"], ascending=[True, False, False]).reset_index(drop=True)


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
    ]

    rows: list[dict[str, object]] = []
    for label, mask in filters:
        filtered = trade_df.loc[mask.fillna(False)].copy()
        row: dict[str, object] = {
            "filter_name": label,
            "sample_count": int(len(filtered)),
            "sample_ratio": float(len(filtered) / len(trade_df)) if len(trade_df) else pd.NA,
        }
        for hold_days in (1, 2):
            stats = summarize_return_series(filtered[f"hold_{hold_days}d_return"] if not filtered.empty else pd.Series(dtype=float))
            row[f"hold_{hold_days}d_win_rate"] = stats["win_rate"]
            row[f"hold_{hold_days}d_avg_return"] = stats["avg_return"]
            row[f"hold_{hold_days}d_expectancy"] = stats["expectancy"]
            row[f"hold_{hold_days}d_profit_factor"] = stats["profit_factor"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def top_trade_samples(trade_df: pd.DataFrame, hold_days: int, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trade_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    col = f"hold_{hold_days}d_return"
    work_df = trade_df.dropna(subset=[col]).copy()
    if work_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    winners = work_df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
    losers = work_df.sort_values(col, ascending=True).head(top_n).reset_index(drop=True)
    return winners, losers


def build_report(
    trade_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    weekly_pool_df: pd.DataFrame,
    daily_summary_df: pd.DataFrame,
    weekly_summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    hold_1d_feature_df: pd.DataFrame,
    hold_2d_feature_df: pd.DataFrame,
    filtered_compare_df: pd.DataFrame,
    meta: dict[str, object],
    output_paths: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# 周线多头 + 日线空转多 + 隔日首根5分钟确认 回测报告")
    lines.append("")
    lines.append("## 回测口径")
    lines.append("")
    lines.append(f"- 回测区间：{meta['signal_start']} 到 {meta['analysis_end']}")
    lines.append("- 第一步：每周母池使用现有 `BullishUpConfig` 的周线多头定义。")
    lines.append("- 第二步：只在当周母池内筛选日线 `空转多`。")
    lines.append("- 第三步：观察信号次日第一根 5 分钟K线，也就是 `09:35` 这根收盘时。")
    lines.append("- 买入条件：首根5分钟收阳、相对信号日收盘涨幅不超过 4%、且 5 分钟收盘价站上信号日最高价。")
    lines.append("- 买入价格口径：用这根 5 分钟K线的收盘价买入。")
    lines.append("- 持有 1 天口径：买入后持有 1 个完整交易日，到下一交易日收盘卖出。")
    lines.append("- 持有 2 天口径：买入后持有 2 个完整交易日，到第二个交易日收盘卖出。")
    lines.append("")

    lines.append("## 样本概览")
    lines.append("")
    lines.append(f"- 周线母池周数：{int(weekly_pool_df['signal_week'].nunique()) if not weekly_pool_df.empty else 0}")
    lines.append(f"- 周线母池总样本：{len(weekly_pool_df)}")
    lines.append(f"- 日线空转多信号数：{len(signal_df)}")
    lines.append(f"- 触发 5 分钟买点的交易数：{len(trade_df)}")
    lines.append(f"- 覆盖股票数：{int(trade_df['file_symbol'].nunique()) if not trade_df.empty else 0}")
    lines.append(f"- 已处理股票文件：{meta['processed_symbols']}")
    lines.append(f"- 缺少周线文件：{meta['missing_weekly']}")
    lines.append(f"- 缺少 5 分钟文件：{meta['missing_5m']}")
    lines.append("")

    lines.append("## 总体结果")
    lines.append("")
    for row in overall_df.itertuples(index=False):
        lines.append(f"### {row.holding_period}")
        lines.append("")
        lines.append(f"- 样本数：{row.sample_count}")
        lines.append(f"- 胜率：{format_pct(row.win_rate)}")
        lines.append(f"- 平均涨跌幅：{format_pct(row.avg_return)}")
        lines.append(f"- 中位数涨跌幅：{format_pct(row.median_return)}")
        lines.append(f"- 平均盈利：{format_pct(row.avg_gain)}")
        lines.append(f"- 平均亏损：{format_pct(row.avg_loss)}")
        lines.append(f"- 期望：{format_pct(row.expectancy)}")
        pf = "inf" if row.profit_factor == math.inf else format_num(row.profit_factor)
        lines.append(f"- Profit Factor：{pf}")
        lines.append(f"- Kelly 粗估：{format_pct(row.kelly_fraction)}")
        lines.append(f"- 单笔最大收益：{format_pct(row.max_return)}")
        lines.append(f"- 单笔最大亏损：{format_pct(row.min_return)}")
        lines.append(f"- 顺序资金曲线最大回撤：{format_pct(row.max_drawdown_seq)}")
        lines.append("")

    if not daily_summary_df.empty:
        best_day_1d = daily_summary_df.sort_values("hold_1d_avg_return", ascending=False).iloc[0]
        worst_day_1d = daily_summary_df.sort_values("hold_1d_avg_return", ascending=True).iloc[0]
        best_day_2d = daily_summary_df.sort_values("hold_2d_avg_return", ascending=False).iloc[0]
        worst_day_2d = daily_summary_df.sort_values("hold_2d_avg_return", ascending=True).iloc[0]
        lines.append("## 每日观察")
        lines.append("")
        lines.append(
            f"- `hold_1d` 最强信号日：{best_day_1d['signal_date']}，{int(best_day_1d['trade_count'])} 笔，平均涨跌幅 {format_pct(best_day_1d['hold_1d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_1d` 最弱信号日：{worst_day_1d['signal_date']}，{int(worst_day_1d['trade_count'])} 笔，平均涨跌幅 {format_pct(worst_day_1d['hold_1d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_2d` 最强信号日：{best_day_2d['signal_date']}，{int(best_day_2d['trade_count'])} 笔，平均涨跌幅 {format_pct(best_day_2d['hold_2d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_2d` 最弱信号日：{worst_day_2d['signal_date']}，{int(worst_day_2d['trade_count'])} 笔，平均涨跌幅 {format_pct(worst_day_2d['hold_2d_avg_return'])}。"
        )
        lines.append("")

    if not weekly_summary_df.empty:
        best_week_1d = weekly_summary_df.sort_values("hold_1d_avg_return", ascending=False).iloc[0]
        worst_week_1d = weekly_summary_df.sort_values("hold_1d_avg_return", ascending=True).iloc[0]
        best_week_2d = weekly_summary_df.sort_values("hold_2d_avg_return", ascending=False).iloc[0]
        worst_week_2d = weekly_summary_df.sort_values("hold_2d_avg_return", ascending=True).iloc[0]
        lines.append("## 每周观察")
        lines.append("")
        lines.append(
            f"- `hold_1d` 最强周：{best_week_1d['signal_week']}，{int(best_week_1d['trade_count'])} 笔，平均涨跌幅 {format_pct(best_week_1d['hold_1d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_1d` 最弱周：{worst_week_1d['signal_week']}，{int(worst_week_1d['trade_count'])} 笔，平均涨跌幅 {format_pct(worst_week_1d['hold_1d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_2d` 最强周：{best_week_2d['signal_week']}，{int(best_week_2d['trade_count'])} 笔，平均涨跌幅 {format_pct(best_week_2d['hold_2d_avg_return'])}。"
        )
        lines.append(
            f"- `hold_2d` 最弱周：{worst_week_2d['signal_week']}，{int(worst_week_2d['trade_count'])} 笔，平均涨跌幅 {format_pct(worst_week_2d['hold_2d_avg_return'])}。"
        )
        lines.append("")

    lines.append("## 优化线索")
    lines.append("")
    if not filtered_compare_df.empty:
        compare_sorted_1d = filtered_compare_df.sort_values(["hold_1d_expectancy", "sample_count"], ascending=[False, False]).head(6)
        compare_sorted_2d = filtered_compare_df.sort_values(["hold_2d_expectancy", "sample_count"], ascending=[False, False]).head(6)
        lines.append("### 过滤条件对比")
        lines.append("")
        lines.append("- 对 `hold_1d` 期望值较友好的过滤：")
        for row in compare_sorted_1d.itertuples(index=False):
            lines.append(
                f"- {row.filter_name} | 样本 {row.sample_count} | 胜率 {format_pct(row.hold_1d_win_rate)} | 平均 {format_pct(row.hold_1d_avg_return)} | 期望 {format_pct(row.hold_1d_expectancy)}"
            )
        lines.append("")
        lines.append("- 对 `hold_2d` 期望值较友好的过滤：")
        for row in compare_sorted_2d.itertuples(index=False):
            lines.append(
                f"- {row.filter_name} | 样本 {row.sample_count} | 胜率 {format_pct(row.hold_2d_win_rate)} | 平均 {format_pct(row.hold_2d_avg_return)} | 期望 {format_pct(row.hold_2d_expectancy)}"
            )
        lines.append("")

    def append_feature_insights(title: str, df: pd.DataFrame, hold_days: int) -> None:
        lines.append(title)
        lines.append("")
        if df.empty:
            lines.append("- 没有足够样本。")
            lines.append("")
            return
        work_df = df[df["sample_count"] >= 5].copy()
        if work_df.empty:
            work_df = df.copy()
        best_rows = work_df.sort_values(["expectancy", "sample_count"], ascending=[False, False]).head(8)
        for row in best_rows.itertuples(index=False):
            pf_value = "inf" if row.profit_factor == math.inf else format_num(row.profit_factor)
            lines.append(
                f"- {row.feature} = {row.bucket} | 样本 {row.sample_count} | 胜率 {format_pct(row.win_rate)} | 平均 {format_pct(row.avg_return)} | 期望 {format_pct(row.expectancy)} | PF {pf_value}"
            )
        lines.append("")

    append_feature_insights("### `hold_1d` 分层表现", hold_1d_feature_df, 1)
    append_feature_insights("### `hold_2d` 分层表现", hold_2d_feature_df, 2)

    if not trade_df.empty:
        winners_1d, losers_1d = top_trade_samples(trade_df, 1, top_n=10)
        winners_2d, losers_2d = top_trade_samples(trade_df, 2, top_n=10)

        def append_top_samples(title: str, df: pd.DataFrame, col: str) -> None:
            lines.append(title)
            lines.append("")
            if df.empty:
                lines.append("- 没有足够样本。")
                lines.append("")
                return
            for row in df.itertuples(index=False):
                lines.append(
                    f"- {row.signal_date} | {row.file_symbol} {row.name} | 入场 {row.entry_time} | 形态 {row.kong_turn_bull_pattern} | 收益 {format_pct(getattr(row, col))}"
                )
            lines.append("")

        append_top_samples("## `hold_1d` 最强样本", winners_1d, "hold_1d_return")
        append_top_samples("## `hold_1d` 最弱样本", losers_1d, "hold_1d_return")
        append_top_samples("## `hold_2d` 最强样本", winners_2d, "hold_2d_return")
        append_top_samples("## `hold_2d` 最弱样本", losers_2d, "hold_2d_return")

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
    hold_1d_feature_df = feature_bucket_analysis(trade_df, hold_days=1)
    hold_2d_feature_df = feature_bucket_analysis(trade_df, hold_days=2)
    filtered_compare_df = compare_filtered_vs_all(trade_df)

    suffix = f"{signal_start.date().isoformat()}_to_{analysis_end.date().isoformat()}"
    output_paths = {
        "weekly_pool": output_dir / f"weekly_bullish_kong_turn_bull_5m_weekly_pool_{suffix}.csv",
        "signals": output_dir / f"weekly_bullish_kong_turn_bull_5m_signals_{suffix}.csv",
        "trades": output_dir / f"weekly_bullish_kong_turn_bull_5m_trades_{suffix}.csv",
        "daily_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_daily_summary_{suffix}.csv",
        "weekly_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_weekly_summary_{suffix}.csv",
        "overall_summary": output_dir / f"weekly_bullish_kong_turn_bull_5m_overall_summary_{suffix}.csv",
        "hold_1d_feature": output_dir / f"weekly_bullish_kong_turn_bull_5m_hold1d_feature_{suffix}.csv",
        "hold_2d_feature": output_dir / f"weekly_bullish_kong_turn_bull_5m_hold2d_feature_{suffix}.csv",
        "filtered_compare": output_dir / f"weekly_bullish_kong_turn_bull_5m_filtered_compare_{suffix}.csv",
        "report": output_dir / f"weekly_bullish_kong_turn_bull_5m_report_{suffix}.md",
    }

    weekly_pool_df.to_csv(output_paths["weekly_pool"], index=False, encoding="utf-8-sig")
    signal_df.to_csv(output_paths["signals"], index=False, encoding="utf-8-sig")
    trade_df.to_csv(output_paths["trades"], index=False, encoding="utf-8-sig")
    daily_summary_df.to_csv(output_paths["daily_summary"], index=False, encoding="utf-8-sig")
    weekly_summary_df.to_csv(output_paths["weekly_summary"], index=False, encoding="utf-8-sig")
    overall_df.to_csv(output_paths["overall_summary"], index=False, encoding="utf-8-sig")
    hold_1d_feature_df.to_csv(output_paths["hold_1d_feature"], index=False, encoding="utf-8-sig")
    hold_2d_feature_df.to_csv(output_paths["hold_2d_feature"], index=False, encoding="utf-8-sig")
    filtered_compare_df.to_csv(output_paths["filtered_compare"], index=False, encoding="utf-8-sig")

    report_text = build_report(
        trade_df=trade_df,
        signal_df=signal_df,
        weekly_pool_df=weekly_pool_df,
        daily_summary_df=daily_summary_df,
        weekly_summary_df=weekly_summary_df,
        overall_df=overall_df,
        hold_1d_feature_df=hold_1d_feature_df,
        hold_2d_feature_df=hold_2d_feature_df,
        filtered_compare_df=filtered_compare_df,
        meta=meta,
        output_paths=output_paths,
    )
    output_paths["report"].write_text(report_text, encoding="utf-8")

    print(f"weekly_pool_rows={len(weekly_pool_df)}")
    print(f"daily_signal_rows={len(signal_df)}")
    print(f"trade_rows={len(trade_df)}")
    print(f"report_path={output_paths['report']}")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
