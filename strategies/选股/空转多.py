from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import (
    KongTurnBullConfig,
    evaluate_latest_kong_turn_bull,
    load_stock_bars,
    scan_period_for_latest_kong_turn_bull,
)
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


OUTPUT_PATH = SCREENING_RESULTS_DIR / "kong_turn_bull_latest.csv"


def main() -> None:
    ensure_project_dirs()
    config = KongTurnBullConfig(
        require_yesterday_lower_low=True,
        require_today_higher_high=True,
    )

    stock_df = load_stock_bars("sh601138", period="daily")
    latest_signal = evaluate_latest_kong_turn_bull(stock_df, config=config)
    print("Single stock latest-bar screen:")
    print(
        latest_signal[
            [
                "symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "kong_turn_bull",
                "kong_turn_bull_pattern",
                "prev2_time",
                "prev1_time",
                "current_time",
                "prev2_low",
                "prev1_low",
                "prev1_high",
                "current_open",
                "current_close",
                "current_high",
                "yesterday_lower_low",
                "today_higher_high",
                "today_lower_low",
                "today_close_breaks_yesterday_high",
                "today_bullish_close",
            ]
        ].to_string()
    )

    pool_signals = scan_period_for_latest_kong_turn_bull(
        period="daily",
        config=config,
    )
    pool_signals.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nPool latest-bar screen:")
    if pool_signals.empty:
        print("No signal found.")
    else:
        print(
            pool_signals[
                [
                    "file_symbol",
                    "name",
                    "kong_turn_bull_pattern",
                    "prev2_time",
                    "prev1_time",
                    "current_time",
                    "prev2_low",
                    "prev1_low",
                    "prev1_high",
                    "current_open",
                    "current_close",
                    "current_high",
                    "current_low",
                    "close",
                    "today_bullish_close",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved latest results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
