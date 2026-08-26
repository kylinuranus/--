# 作用：演示 Dow Pullback 回调形态的最新K线筛选流程。
# 内容：分别对单只股票和股票池做“当前是否处于上升趋势回调中”的筛选，并对比是否启用价格落在回调区下半区的过滤条件。
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import (
    DowPullbackConfig,
    evaluate_latest_dow_pullback,
    get_latest_confirmed_swings,
    load_stock_bars,
    scan_period_for_latest_dow_pullback,
)


def main() -> None:
    base_config = DowPullbackConfig(
        swing_window=3,
        min_pullback_bars=2,
        min_pullback_retracement=0.10,
        max_pullback_retracement=0.60,
        require_price_in_lower_half=False,
    )

    stock_df = load_stock_bars("sh601138", period="daily")
    latest_signal = evaluate_latest_dow_pullback(stock_df, config=base_config, recent_bars=180)
    latest_swings = get_latest_confirmed_swings(stock_df, config=base_config, recent_bars=180, count=4)
    print("Single stock latest-bar screen (lower-half filter OFF):")
    print(
        latest_signal[
            [
                "symbol",
                "name",
                "close",
                "dow_pullback",
                "low1_price",
                "high1_price",
                "low2_price",
                "high2_price",
                "price_zone_ratio",
                "last_higher_low",
                "last_swing_high",
                "pullback_retracement",
                "bars_since_last_high",
                "screen_window_bars",
            ]
        ].to_string()
    )
    print("\nLatest confirmed swings:")
    if latest_swings.empty:
        print("No confirmed swings.")
    else:
        print(latest_swings[["pivot_time", "swing_type", "swing_price", "confirmed_time"]].to_string(index=False))

    strict_config = DowPullbackConfig(
        swing_window=3,
        min_pullback_bars=2,
        min_pullback_retracement=0.10,
        max_pullback_retracement=0.60,
        require_price_in_lower_half=True,
        lower_half_threshold=0.50,
    )
    strict_latest_signal = evaluate_latest_dow_pullback(stock_df, config=strict_config, recent_bars=180)
    print("\nSingle stock latest-bar screen (lower-half filter ON):")
    print(
        strict_latest_signal[
            [
                "symbol",
                "name",
                "close",
                "dow_pullback",
                "low2_price",
                "high2_price",
                "price_zone_ratio",
            ]
        ].to_string()
    )

    pool_signals = scan_period_for_latest_dow_pullback(
        period="daily",
        config=base_config,
        recent_bars=180,
        limit=100,
    )
    print("\nPool latest-bar screen (lower-half filter OFF):")
    if pool_signals.empty:
        print("No signal found.")
    else:
        print(
            pool_signals[
                [
                    "symbol",
                    "name",
                    "close",
                    "low1_price",
                    "high1_price",
                    "low2_price",
                    "high2_price",
                    "price_zone_ratio",
                    "last_higher_low",
                    "last_swing_high",
                    "pullback_retracement",
                    "bars_since_last_high",
                ]
            ].to_string(index=False)
        )

    strict_pool_signals = scan_period_for_latest_dow_pullback(
        period="daily",
        config=strict_config,
        recent_bars=180,
        limit=100,
    )
    print("\nPool latest-bar screen (lower-half filter ON):")
    if strict_pool_signals.empty:
        print("No signal found.")
    else:
        print(
            strict_pool_signals[
                [
                    "symbol",
                    "name",
                    "close",
                    "low2_price",
                    "high2_price",
                    "price_zone_ratio",
                    "pullback_retracement",
                    "bars_since_last_high",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
