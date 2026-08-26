# 作用：演示 Bullish Up（两根K线偏强上行形态）的基础用法。
# 内容：加载单只股票日线数据、筛选形态信号，并扫描股票池里的最新匹配结果。
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, filter_bullish_up_signals, load_stock_bars, scan_period_for_bullish_up
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.测试策略.筛选周线多头_三十分钟道氏回调 import screen_latest_weekly_bullish


OUTPUT_NAME_BY_PERIOD = {
    "daily": "daily_bullish_latest.csv",
    "weekly": "weekly_bullish_latest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bullish-up screening for a single stock and the whole pool.")
    parser.add_argument("--period", default="weekly", choices=["daily", "weekly"], help="Bar period to screen.")
    parser.add_argument("--symbol", default="sh601138", help="Single stock file_symbol to preview.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to screening_results/<period>_bullish_latest.csv.",
    )
    return parser.parse_args()


def build_config() -> BullishUpConfig:
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


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    config = build_config()
    output_path = args.output or (SCREENING_RESULTS_DIR / OUTPUT_NAME_BY_PERIOD[args.period])

    stock_df = load_stock_bars(args.symbol, period=args.period)
    if args.period == "weekly":
        week_key = stock_df.index.to_series().dt.strftime("%G-%V")
        stock_df = stock_df.groupby(week_key, sort=False).tail(1).copy()
    stock_signals = filter_bullish_up_signals(stock_df, config=config)
    print(f"Single stock latest signals ({args.symbol}, period={args.period}):")
    if stock_signals.empty:
        print("No signal found for the single stock.")
    else:
        print(stock_signals[["open", "high", "low", "close"]].tail(10).to_string())

    if args.period == "weekly":
        # Reuse the repo's existing weekly-only screener logic instead of inferring
        # weekly bars from the generic scan function.
        pool_signals, _ = screen_latest_weekly_bullish(
            weekly_config=config,
            output_dir=SCREENING_RESULTS_DIR,
        )
        if output_path != SCREENING_RESULTS_DIR / OUTPUT_NAME_BY_PERIOD["weekly"]:
            pool_signals.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        pool_signals = scan_period_for_bullish_up(
            period=args.period,
            config=config,
            limit=args.limit,
        )
        if not pool_signals.empty:
            latest_time = pool_signals.index.max()
            pool_signals = pool_signals.loc[pool_signals.index == latest_time].copy()
            time_column = pool_signals.index.name or "datetime"
            pool_signals = (
                pool_signals.sort_values(["file_symbol"])
                .reset_index()
                .rename(columns={time_column: f"{args.period}_time"})
            )
            pool_signals.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nPool latest-bar screen ({args.period}):")
    if pool_signals.empty:
        print("No signal found.")
    else:
        if args.period == "weekly":
            print(
                pool_signals[
                    ["file_symbol", "symbol", "name", "weekly_time", "weekly_open", "weekly_high", "weekly_low", "weekly_close"]
                ].to_string(index=False)
            )
        else:
            print(
                pool_signals[[f"{args.period}_time", "file_symbol", "symbol", "open", "high", "low", "close"]].to_string(index=False)
            )
        print(f"\nSaved latest results to: {output_path}")


if __name__ == "__main__":
    main()
