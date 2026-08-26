# 作用：演示 Dow Pullback 回调形态的常规使用方法。
# 内容：查看单只股票历史回调信号、评估最新一根K线状态，并扫描股票池中的相关机会。
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
    filter_dow_pullback_signals,
    load_stock_bars,
    scan_period_for_dow_pullback,
    scan_period_for_latest_dow_pullback,
)


def main() -> None:
    config = DowPullbackConfig(
        swing_window=3,
        min_pullback_bars=2,
        min_pullback_retracement=0.15,
        max_pullback_retracement=0.70,
    )

    stock_df = load_stock_bars("sh601138", period="daily")
    stock_signals = filter_dow_pullback_signals(stock_df, config=config)
    print("Single stock latest pullback signals:")
    if stock_signals.empty:
        print("No signal found.")
    else:
        print(
            stock_signals[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "last_higher_low",
                    "last_swing_high",
                    "pullback_retracement",
                    "bars_since_last_high",
                ]
            ]
            .tail(10)
            .to_string()
        )

    latest_snapshot = evaluate_latest_dow_pullback(stock_df, config=config, recent_bars=180)
    print("\nSingle stock latest-bar screening result:")
    print(
        latest_snapshot[
            [
                "open",
                "high",
                "low",
                "close",
                "dow_pullback",
                "last_higher_low",
                "last_swing_high",
                "pullback_retracement",
                "bars_since_last_high",
                "screen_window_bars",
            ]
        ].to_string()
    )

    pool_signals = scan_period_for_dow_pullback(
        period="daily",
        config=config,
        limit=20,
    )
    print("\nPool scan latest pullback signals:")
    if pool_signals.empty:
        print("No signal found.")
    else:
        print(
            pool_signals[
                [
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "last_higher_low",
                    "last_swing_high",
                    "pullback_retracement",
                    "bars_since_last_high",
                ]
            ]
            .tail(20)
            .to_string()
        )

    latest_pool_signals = scan_period_for_latest_dow_pullback(
        period="daily",
        config=config,
        recent_bars=180,
        limit=20,
    )
    print("\nPool latest-bar screening signals:")
    if latest_pool_signals.empty:
        print("No signal found.")
    else:
        print(
            latest_pool_signals[
                [
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "last_higher_low",
                    "last_swing_high",
                    "pullback_retracement",
                    "bars_since_last_high",
                    "screen_window_bars",
                ]
            ].tail(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()
