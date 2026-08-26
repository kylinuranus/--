# 作用：提供严格版 Pin Bar K 线的定义、最新一根K线评估和全市场扫描入口。
# 内容：把常见 Pin Bar 形态量化为可复用规则，支持日线/周线/月线最新完成K线筛选。
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import load_stock_bars
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


A_SHARE_MARKET_CLOSE_HOUR = 15
REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class StrictPinBarConfig:
    """Strict Pin Bar configuration tuned for strong rejection candles."""

    max_body_ratio: float = 0.25
    min_long_shadow_to_body_ratio: float = 3.0
    min_long_shadow_ratio: float = 0.60
    max_short_shadow_ratio: float = 0.10
    bullish_min_body_zone_ratio: float = 0.65
    bearish_max_body_zone_ratio: float = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict Pin Bar screening on the latest completed bar.")
    parser.add_argument("--period", default="daily", choices=["daily", "weekly", "monthly"], help="Bar period to screen.")
    parser.add_argument(
        "--direction",
        default="both",
        choices=["both", "bullish", "bearish"],
        help="Only keep bullish or bearish pin bars. Default keeps both.",
    )
    parser.add_argument("--symbol", default="sh601138", help="Single stock file_symbol to preview.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N files.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional reference date in YYYY-MM-DD. Used to decide whether weekly/monthly bars are completed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to screening_results/strict_pinbar_<period>_<direction>_latest.csv.",
    )
    return parser.parse_args()


def build_config() -> StrictPinBarConfig:
    # Strict version:
    # - body <= 25% of full range
    # - dominant wick >= 3x body
    # - dominant wick >= 60% of full range
    # - opposite wick <= 10% of full range
    # - bullish body fully sits in the upper 35% of the range
    # - bearish body fully sits in the lower 35% of the range
    return StrictPinBarConfig()


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _bar_range(df: pd.DataFrame) -> pd.Series:
    raw_range = df["high"] - df["low"]
    return raw_range.where(raw_range > 0, 1e-9)


def _normalize_reference_time(reference_time: str | pd.Timestamp | None = None) -> pd.Timestamp:
    if reference_time is None:
        return pd.Timestamp.now(tz="Asia/Shanghai")

    current_time = pd.Timestamp(reference_time)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("Asia/Shanghai")
    else:
        current_time = current_time.tz_convert("Asia/Shanghai")

    if isinstance(reference_time, str) and len(reference_time.strip()) == 10:
        current_time = current_time.normalize() + pd.Timedelta(hours=A_SHARE_MARKET_CLOSE_HOUR)

    return current_time


def _iso_week_key(timestamp: pd.Timestamp) -> tuple[int, int]:
    iso = timestamp.isocalendar()
    return int(iso.year), int(iso.week)


def _dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def _dedupe_to_last_bar_per_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    month_key = df.index.to_series().dt.strftime("%Y-%m")
    return df.groupby(month_key, sort=False).tail(1).copy()


def _is_a_share_week_closed(reference_time: pd.Timestamp) -> bool:
    if reference_time.weekday() < 4:
        return False
    if reference_time.weekday() > 4:
        return True

    market_close_time = reference_time.normalize() + pd.Timedelta(hours=A_SHARE_MARKET_CLOSE_HOUR)
    return reference_time >= market_close_time


def prepare_latest_screen_frame(
    df: pd.DataFrame,
    period: str,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    prepared = df.sort_index()
    if prepared.empty:
        return prepared

    current_time = _normalize_reference_time(reference_time)

    if period == "weekly":
        prepared = _dedupe_to_last_bar_per_week(prepared)
        if len(prepared) < 1:
            return prepared

        latest_bar_time = pd.Timestamp(prepared.index[-1])
        if _iso_week_key(latest_bar_time) == _iso_week_key(current_time) and not _is_a_share_week_closed(current_time):
            return prepared.iloc[:-1].copy()
        return prepared

    if period == "monthly":
        prepared = _dedupe_to_last_bar_per_month(prepared)
        if len(prepared) < 1:
            return prepared

        latest_bar_time = pd.Timestamp(prepared.index[-1])
        if latest_bar_time.year == current_time.year and latest_bar_time.month == current_time.month:
            return prepared.iloc[:-1].copy()
        return prepared

    return prepared


def analyze_strict_pinbar(
    df: pd.DataFrame,
    config: StrictPinBarConfig | None = None,
) -> pd.DataFrame:
    """Analyze strict bullish/bearish pin bar structure on every row."""

    pattern_config = config or StrictPinBarConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    bar_range = _bar_range(df)
    body = (df["close"] - df["open"]).abs()
    safe_body = body.where(body > 0, 1e-9)
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    body_ratio = body / bar_range
    upper_shadow_ratio = upper_shadow / bar_range
    lower_shadow_ratio = lower_shadow / bar_range
    open_location = (df["open"] - df["low"]) / bar_range
    close_location = (df["close"] - df["low"]) / bar_range
    body_low_location = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_range
    body_high_location = (df[["open", "close"]].max(axis=1) - df["low"]) / bar_range

    body_small = body_ratio <= pattern_config.max_body_ratio
    bullish_long_shadow = (lower_shadow >= safe_body * pattern_config.min_long_shadow_to_body_ratio) & (
        lower_shadow_ratio >= pattern_config.min_long_shadow_ratio
    )
    bearish_long_shadow = (upper_shadow >= safe_body * pattern_config.min_long_shadow_to_body_ratio) & (
        upper_shadow_ratio >= pattern_config.min_long_shadow_ratio
    )
    bullish_short_shadow = upper_shadow_ratio <= pattern_config.max_short_shadow_ratio
    bearish_short_shadow = lower_shadow_ratio <= pattern_config.max_short_shadow_ratio
    bullish_body_high_zone = body_low_location >= pattern_config.bullish_min_body_zone_ratio
    bearish_body_low_zone = body_high_location <= pattern_config.bearish_max_body_zone_ratio

    bullish_pinbar = body_small & bullish_long_shadow & bullish_short_shadow & bullish_body_high_zone
    bearish_pinbar = body_small & bearish_long_shadow & bearish_short_shadow & bearish_body_low_zone
    strict_pinbar = bullish_pinbar | bearish_pinbar

    direction = pd.Series(pd.NA, index=df.index, dtype="object", name="pinbar_direction")
    direction.loc[bullish_pinbar] = "bullish"
    direction.loc[bearish_pinbar] = "bearish"

    analysis = pd.DataFrame(index=df.index)
    analysis["strict_pinbar"] = strict_pinbar
    analysis["pinbar_direction"] = direction
    analysis["bullish_pinbar"] = bullish_pinbar
    analysis["bearish_pinbar"] = bearish_pinbar
    analysis["body"] = body
    analysis["bar_range"] = bar_range
    analysis["upper_shadow"] = upper_shadow
    analysis["lower_shadow"] = lower_shadow
    analysis["body_ratio"] = body_ratio
    analysis["upper_shadow_ratio"] = upper_shadow_ratio
    analysis["lower_shadow_ratio"] = lower_shadow_ratio
    analysis["open_location"] = open_location
    analysis["close_location"] = close_location
    analysis["body_low_location"] = body_low_location
    analysis["body_high_location"] = body_high_location
    analysis.attrs["config"] = asdict(pattern_config)
    return analysis


def evaluate_latest_strict_pinbar(
    df: pd.DataFrame,
    period: str,
    config: StrictPinBarConfig | None = None,
    reference_time: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Evaluate the latest completed bar for strict pin bar structure."""

    if df.empty:
        raise ValueError("DataFrame is empty")

    prepared_df = prepare_latest_screen_frame(df, period=period, reference_time=reference_time)
    if prepared_df.empty:
        raise ValueError("DataFrame has no completed bars for the requested period")

    analysis = analyze_strict_pinbar(prepared_df, config=config)
    latest_snapshot = prepared_df.iloc[-1].copy()
    latest_analysis = analysis.iloc[-1]
    for column in analysis.columns:
        latest_snapshot[column] = latest_analysis[column]
    latest_snapshot.attrs["config"] = analysis.attrs.get("config", {})
    return latest_snapshot


def scan_period_for_latest_strict_pinbar(
    period: str = "daily",
    direction: str = "both",
    data_root: str | Path | None = None,
    config: StrictPinBarConfig | None = None,
    limit: int | None = None,
    reference_time: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Scan the latest completed bar of every stock for strict pin bar signals."""

    root_data = Path(data_root) if data_root is not None else Path(r"D:\a_stock_data")
    root = root_data / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
    matches: list[pd.Series] = []
    pattern_config = config or StrictPinBarConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        if df.empty:
            continue

        try:
            latest_snapshot = evaluate_latest_strict_pinbar(
                df,
                period=period,
                config=pattern_config,
                reference_time=reference_time,
            )
        except ValueError:
            continue

        if not bool(latest_snapshot["strict_pinbar"]):
            continue
        if direction != "both" and latest_snapshot["pinbar_direction"] != direction:
            continue

        latest_snapshot = latest_snapshot.copy()
        if "symbol" not in latest_snapshot.index or pd.isna(latest_snapshot["symbol"]):
            latest_snapshot["symbol"] = parquet_path.stem[-6:]
        latest_snapshot["file_symbol"] = parquet_path.stem
        matches.append(latest_snapshot)

    if not matches:
        return pd.DataFrame(
            columns=[
                "symbol",
                "file_symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "strict_pinbar",
                "pinbar_direction",
                "bullish_pinbar",
                "bearish_pinbar",
                "body",
                "bar_range",
                "upper_shadow",
                "lower_shadow",
                "body_ratio",
                "upper_shadow_ratio",
                "lower_shadow_ratio",
                "open_location",
                "close_location",
                "body_low_location",
                "body_high_location",
            ]
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(pattern_config)
    return combined


def build_default_output_path(period: str, direction: str) -> Path:
    return SCREENING_RESULTS_DIR / f"strict_pinbar_{period}_{direction}_latest.csv"


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    config = build_config()
    reference_time = _normalize_reference_time(args.date) if args.date else None
    output_path = args.output or build_default_output_path(args.period, args.direction)

    stock_df = load_stock_bars(args.symbol, period=args.period)
    latest_signal = evaluate_latest_strict_pinbar(
        stock_df,
        period=args.period,
        config=config,
        reference_time=reference_time,
    )
    print("Single stock latest completed-bar screen:")
    print(
        latest_signal[
            [
                "symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "strict_pinbar",
                "pinbar_direction",
                "bullish_pinbar",
                "bearish_pinbar",
                "body",
                "bar_range",
                "upper_shadow",
                "lower_shadow",
                "body_ratio",
                "upper_shadow_ratio",
                "lower_shadow_ratio",
                "body_low_location",
                "body_high_location",
            ]
        ].to_string()
    )

    pool_signals = scan_period_for_latest_strict_pinbar(
        period=args.period,
        direction=args.direction,
        config=config,
        limit=args.limit,
        reference_time=reference_time,
    )
    pool_signals.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nPool latest completed-bar screen:")
    if pool_signals.empty:
        print("No strict pin bar signal found.")
    else:
        print(
            pool_signals[
                [
                    "file_symbol",
                    "name",
                    "pinbar_direction",
                    "open",
                    "high",
                    "low",
                    "close",
                    "body_ratio",
                    "upper_shadow_ratio",
                    "lower_shadow_ratio",
                    "body_low_location",
                    "body_high_location",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved latest results to: {output_path}")


if __name__ == "__main__":
    main()
