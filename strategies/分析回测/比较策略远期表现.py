from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import (
    BullishUpConfig,
    DowPullbackConfig,
    StrongWeakStrongConfig,
    analyze_dow_pullback,
    detect_bullish_up,
    detect_strong_weak_strong,
)
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
DEFAULT_OUTPUT_DIR = SCREENING_RESULTS_DIR
FORWARD_HORIZONS = (1, 3, 5, 10)
WEEKLY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
DAILY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
M30_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare forward returns of weekly bullish, 30m Dow pullback, combined, and strong-weak-strong strategies."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for quick testing.")
    parser.add_argument(
        "--common-start",
        default="2021-06-08",
        help="Common start date used for the comparison universe.",
    )
    parser.add_argument(
        "--signal-end",
        default="2026-06-05",
        help="Last signal date to include. 30m data in this repo currently stops at 2026-06-05.",
    )
    return parser.parse_args()


def dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


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


def dow_pullback_config() -> DowPullbackConfig:
    return DowPullbackConfig(
        swing_window=3,
        min_pullback_bars=2,
        min_pullback_retracement=0.10,
        max_pullback_retracement=0.60,
        require_price_in_lower_half=False,
    )


def strong_weak_strong_config() -> StrongWeakStrongConfig:
    return StrongWeakStrongConfig(
        require_k1_bull=True,
        require_k2_bear=True,
        require_k3_bull=True,
        strict_engulf=False,
    )


def build_forward_rows(
    strategy_name: str,
    symbol: str,
    name: str,
    signal_dates: list[pd.Timestamp],
    daily_df: pd.DataFrame,
    rows: list[dict[str, object]],
) -> None:
    if not signal_dates:
        return

    ordered_daily = daily_df.sort_index().copy()
    date_to_pos = {timestamp: idx for idx, timestamp in enumerate(ordered_daily.index)}
    open_prices = ordered_daily["open"].tolist()
    close_prices = ordered_daily["close"].tolist()

    for signal_date in signal_dates:
        if signal_date not in date_to_pos:
            continue

        signal_pos = date_to_pos[signal_date]
        entry_pos = signal_pos + 1
        if entry_pos >= len(ordered_daily):
            continue

        entry_date = ordered_daily.index[entry_pos]
        entry_open = float(open_prices[entry_pos])
        signal_close = float(close_prices[signal_pos])

        base_row = {
            "strategy": strategy_name,
            "file_symbol": symbol,
            "name": name,
            "signal_date": signal_date.date().isoformat(),
            "entry_date": entry_date.date().isoformat(),
            "signal_close": signal_close,
            "entry_open": entry_open,
        }

        for horizon in FORWARD_HORIZONS:
            exit_pos = signal_pos + horizon
            if exit_pos >= len(ordered_daily):
                base_row[f"ret_{horizon}d"] = pd.NA
                base_row[f"exit_date_{horizon}d"] = pd.NA
                continue
            exit_date = ordered_daily.index[exit_pos]
            exit_close = float(close_prices[exit_pos])
            base_row[f"ret_{horizon}d"] = exit_close / entry_open - 1.0
            base_row[f"exit_date_{horizon}d"] = exit_date.date().isoformat()

        rows.append(base_row)


def summarize_strategy(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for strategy_name, group in detail_df.groupby("strategy", sort=False):
        strategy_row = {
            "strategy": strategy_name,
            "signal_count_total": int(len(group)),
            "unique_symbols": int(group["file_symbol"].nunique()),
            "signal_start": group["signal_date"].min(),
            "signal_end": group["signal_date"].max(),
        }

        for horizon in FORWARD_HORIZONS:
            col = f"ret_{horizon}d"
            valid = group[col].dropna().astype(float)
            strategy_row[f"count_{horizon}d"] = int(valid.shape[0])
            if valid.empty:
                strategy_row[f"avg_ret_{horizon}d"] = pd.NA
                strategy_row[f"median_ret_{horizon}d"] = pd.NA
                strategy_row[f"win_rate_{horizon}d"] = pd.NA
                strategy_row[f"avg_gain_{horizon}d"] = pd.NA
                strategy_row[f"avg_loss_{horizon}d"] = pd.NA
                continue

            wins = valid[valid > 0]
            losses = valid[valid <= 0]
            strategy_row[f"avg_ret_{horizon}d"] = float(valid.mean())
            strategy_row[f"median_ret_{horizon}d"] = float(valid.median())
            strategy_row[f"win_rate_{horizon}d"] = float((valid > 0).mean())
            strategy_row[f"avg_gain_{horizon}d"] = float(wins.mean()) if not wins.empty else pd.NA
            strategy_row[f"avg_loss_{horizon}d"] = float(losses.mean()) if not losses.empty else pd.NA

        summary_rows.append(strategy_row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(["avg_ret_5d", "avg_ret_10d"], ascending=False, na_position="last").reset_index(drop=True)


def compare_strategies(
    data_root: Path,
    output_dir: Path,
    common_start: pd.Timestamp,
    signal_end: pd.Timestamp,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_root = data_root / "daily"
    weekly_root = data_root / "weekly"
    m30_root = data_root / "30m"

    weekly_config = weekly_bullish_config()
    pullback_config = dow_pullback_config()
    sws_config = strong_weak_strong_config()

    detail_rows: list[dict[str, object]] = []
    symbol_paths = sorted(daily_root.glob("*.parquet"))
    if limit is not None:
        symbol_paths = symbol_paths[:limit]

    for idx, daily_path in enumerate(symbol_paths, start=1):
        symbol = daily_path.stem
        weekly_path = weekly_root / daily_path.name
        m30_path = m30_root / daily_path.name
        if not weekly_path.exists() or not m30_path.exists():
            continue

        daily_df = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
        daily_df = daily_df.loc[str(common_start.date()) :].copy()
        if len(daily_df) < 30:
            continue

        name = str(daily_df["name"].dropna().iloc[-1]) if "name" in daily_df.columns and not daily_df["name"].dropna().empty else symbol

        weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        weekly_df = dedupe_to_last_bar_per_week(weekly_df)
        weekly_df = weekly_df.loc[str(common_start.date()) :].copy()
        if len(weekly_df) >= 2:
            weekly_signal = detect_bullish_up(weekly_df, config=weekly_config).fillna(False)
            weekly_signal_dates = [
                pd.Timestamp(ts).normalize()
                for ts, flag in weekly_signal.items()
                if bool(flag) and pd.Timestamp(ts).normalize() <= signal_end
            ]
            weekly_state = pd.Series(
                weekly_signal.values,
                index=pd.to_datetime(weekly_df.index).normalize(),
                dtype="boolean",
            )
            weekly_active = weekly_state.reindex(pd.to_datetime(daily_df.index).normalize(), method="ffill")
            weekly_active = weekly_active.astype("boolean").fillna(False).astype(bool)
        else:
            weekly_signal_dates = []
            weekly_active = pd.Series(False, index=pd.to_datetime(daily_df.index).normalize())

        sws_signal = detect_strong_weak_strong(daily_df, config=sws_config).fillna(False)
        sws_signal_dates = [
            pd.Timestamp(ts).normalize()
            for ts, flag in sws_signal.items()
            if bool(flag) and pd.Timestamp(ts).normalize() <= signal_end
        ]

        m30_df = pd.read_parquet(m30_path, columns=M30_COLUMNS).sort_index()
        m30_df = m30_df.loc[str(common_start.date()) : str(signal_end.date())].copy()
        if m30_df.empty:
            continue
        m30_analysis = analyze_dow_pullback(m30_df, config=pullback_config)
        m30_close_rows = m30_analysis.loc[m30_analysis.index.strftime("%H:%M:%S") == "15:00:00"].copy()
        m30_signal_dates = [
            pd.Timestamp(ts).normalize()
            for ts, flag in m30_close_rows["dow_pullback"].fillna(False).items()
            if bool(flag) and pd.Timestamp(ts).normalize() <= signal_end
        ]

        m30_signal_date_set = set(m30_signal_dates)
        combined_signal_dates = [
            pd.Timestamp(day).normalize()
            for day, is_active in weekly_active.items()
            if bool(is_active) and pd.Timestamp(day).normalize() in m30_signal_date_set and pd.Timestamp(day).normalize() <= signal_end
        ]

        build_forward_rows("weekly_bullish", symbol, name, weekly_signal_dates, daily_df, detail_rows)
        build_forward_rows("m30_dow_pullback", symbol, name, m30_signal_dates, daily_df, detail_rows)
        build_forward_rows("weekly_bullish_plus_m30_pullback", symbol, name, combined_signal_dates, daily_df, detail_rows)
        build_forward_rows("strong_weak_strong", symbol, name, sws_signal_dates, daily_df, detail_rows)

        if idx % 100 == 0:
            print(f"processed_symbols={idx}")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    detail_df = detail_df.sort_values(["strategy", "signal_date", "file_symbol"]).reset_index(drop=True)
    summary_df = summarize_strategy(detail_df)

    suffix = f"{common_start.date()}_to_{signal_end.date()}"
    detail_path = output_dir / f"strategy_forward_compare_detail_{suffix}.csv"
    summary_path = output_dir / f"strategy_forward_compare_summary_{suffix}.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return detail_df, summary_df


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_start = pd.Timestamp(args.common_start)
    signal_end = pd.Timestamp(args.signal_end)

    detail_df, summary_df = compare_strategies(
        data_root=Path(args.data_root),
        output_dir=output_dir,
        common_start=common_start,
        signal_end=signal_end,
        limit=args.limit,
    )

    print(f"detail_rows={len(detail_df)}")
    print(f"summary_rows={len(summary_df)}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
