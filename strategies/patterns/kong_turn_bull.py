from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS


@dataclass(frozen=True)
class KongTurnBullConfig:
    """Parameters for multi-day kong-turn-bull pattern variants."""

    require_yesterday_lower_low: bool = True
    require_today_higher_high: bool = True
    require_three_day_bullish_close: bool = True
    enable_three_day_pattern: bool = True
    enable_two_day_close_breakout: bool = True


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def analyze_kong_turn_bull(
    df: pd.DataFrame,
    config: KongTurnBullConfig | None = None,
) -> pd.DataFrame:
    """
    Analyze "kong turn bull" pattern variants.

    Supported variants:
    - Three-day pattern:
      Yesterday's low is lower than the prior day's low, and today's high is
      higher than yesterday's high.
    - Two-day pattern:
      Today's low is lower than yesterday's low, and today's close is higher
      than yesterday's high.

    The signal is attached to today's bar.
    """

    pattern_config = config or KongTurnBullConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    prev2_high = df["high"].shift(2)
    prev2_low = df["low"].shift(2)
    prev1_high = df["high"].shift(1)
    prev1_low = df["low"].shift(1)
    current_open = df["open"]
    current_close = df["close"]
    current_high = df["high"]
    current_low = df["low"]

    yesterday_lower_low = prev1_low < prev2_low
    today_higher_high = current_high > prev1_high
    today_lower_low = current_low < prev1_low
    today_close_breaks_yesterday_high = current_close > prev1_high
    today_bullish_close = current_close > current_open

    three_day_signal = pd.Series(True, index=df.index, dtype=bool, name="three_day_kong_turn_bull")
    if pattern_config.require_yesterday_lower_low:
        three_day_signal &= yesterday_lower_low.fillna(False)
    if pattern_config.require_today_higher_high:
        three_day_signal &= today_higher_high.fillna(False)
    if pattern_config.require_three_day_bullish_close:
        three_day_signal &= today_bullish_close.fillna(False)
    if not pattern_config.enable_three_day_pattern:
        three_day_signal &= False

    two_day_signal = pd.Series(True, index=df.index, dtype=bool, name="two_day_kong_turn_bull")
    if pattern_config.require_yesterday_lower_low:
        two_day_signal &= today_lower_low.fillna(False)
    if pattern_config.enable_two_day_close_breakout:
        two_day_signal &= today_close_breaks_yesterday_high.fillna(False)
    else:
        two_day_signal &= False

    signal = (three_day_signal | two_day_signal).rename("kong_turn_bull")
    pattern_type = pd.Series(pd.NA, index=df.index, dtype="object", name="kong_turn_bull_pattern")
    pattern_type.loc[three_day_signal] = "three_day"
    pattern_type.loc[two_day_signal] = "two_day"
    pattern_type.loc[three_day_signal & two_day_signal] = "both"

    index_series = df.index.to_series()
    analysis = pd.DataFrame(index=df.index)
    analysis["kong_turn_bull"] = signal
    analysis["kong_turn_bull_pattern"] = pattern_type
    analysis["prev2_time"] = index_series.shift(2)
    analysis["prev1_time"] = index_series.shift(1)
    analysis["current_time"] = index_series
    analysis["prev2_high"] = prev2_high
    analysis["prev2_low"] = prev2_low
    analysis["prev1_high"] = prev1_high
    analysis["prev1_low"] = prev1_low
    analysis["current_open"] = current_open
    analysis["current_close"] = current_close
    analysis["current_high"] = current_high
    analysis["current_low"] = current_low
    analysis["yesterday_lower_low"] = yesterday_lower_low
    analysis["today_higher_high"] = today_higher_high
    analysis["today_lower_low"] = today_lower_low
    analysis["today_close_breaks_yesterday_high"] = today_close_breaks_yesterday_high
    analysis["today_bullish_close"] = today_bullish_close
    analysis["three_day_kong_turn_bull"] = three_day_signal
    analysis["two_day_kong_turn_bull"] = two_day_signal
    analysis.attrs["config"] = asdict(pattern_config)
    return analysis


def detect_kong_turn_bull(
    df: pd.DataFrame,
    config: KongTurnBullConfig | None = None,
) -> pd.Series:
    """Return a boolean signal for the three-day kong-turn-bull pattern."""

    return analyze_kong_turn_bull(df, config=config)["kong_turn_bull"].rename("kong_turn_bull")


def filter_kong_turn_bull_signals(
    df: pd.DataFrame,
    config: KongTurnBullConfig | None = None,
) -> pd.DataFrame:
    """Return rows where the kong-turn-bull pattern is triggered."""

    analysis = analyze_kong_turn_bull(df, config=config)
    signal = analysis["kong_turn_bull"].fillna(False)
    result = df.loc[signal].copy()
    result["kong_turn_bull"] = True
    for column in [
        "kong_turn_bull_pattern",
        "prev2_time",
        "prev1_time",
        "current_time",
        "prev2_high",
        "prev2_low",
        "prev1_high",
        "prev1_low",
        "current_open",
        "current_close",
        "current_high",
        "current_low",
        "yesterday_lower_low",
        "today_higher_high",
        "today_lower_low",
        "today_close_breaks_yesterday_high",
        "today_bullish_close",
        "three_day_kong_turn_bull",
        "two_day_kong_turn_bull",
    ]:
        result[column] = analysis.loc[signal, column]
    return result


def evaluate_latest_kong_turn_bull(
    df: pd.DataFrame,
    config: KongTurnBullConfig | None = None,
) -> pd.Series:
    """Evaluate only the latest bar as today's bar of the pattern."""

    if df.empty:
        raise ValueError("DataFrame is empty")

    analysis = analyze_kong_turn_bull(df, config=config)
    latest_snapshot = df.iloc[-1].copy()
    latest_analysis = analysis.iloc[-1]
    for column in analysis.columns:
        latest_snapshot[column] = latest_analysis[column]
    latest_snapshot.attrs["config"] = analysis.attrs.get("config", {})
    return latest_snapshot


def detect_latest_kong_turn_bull(
    df: pd.DataFrame,
    config: KongTurnBullConfig | None = None,
) -> bool:
    """Return whether the newest bar matches the kong-turn-bull pattern."""

    latest_snapshot = evaluate_latest_kong_turn_bull(df, config=config)
    return bool(latest_snapshot["kong_turn_bull"])


def scan_period_for_kong_turn_bull(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: KongTurnBullConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan a period folder and return all matching kong-turn-bull bars."""

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    matches: list[pd.DataFrame] = []
    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path).sort_index()
        if df.empty:
            continue
        signal_rows = filter_kong_turn_bull_signals(df, config=config)
        if signal_rows.empty:
            continue

        signal_rows = signal_rows.copy()
        if "symbol" not in signal_rows.columns:
            signal_rows["symbol"] = parquet_path.stem
        signal_rows["file_symbol"] = parquet_path.stem
        matches.append(signal_rows)

    if not matches:
        return pd.DataFrame(columns=["symbol", "file_symbol", "kong_turn_bull"])

    combined = pd.concat(matches).sort_index()
    combined.attrs["config"] = asdict(config or KongTurnBullConfig())
    return combined


def scan_period_for_latest_kong_turn_bull(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: KongTurnBullConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan the latest bar of every stock for the kong-turn-bull pattern."""

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
    matches: list[pd.Series] = []
    pattern_config = config or KongTurnBullConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        if df.empty:
            continue
        latest_snapshot = evaluate_latest_kong_turn_bull(df, config=pattern_config)
        if not bool(latest_snapshot["kong_turn_bull"]):
            continue

        latest_snapshot = latest_snapshot.copy()
        if "symbol" not in latest_snapshot.index or pd.isna(latest_snapshot["symbol"]):
            latest_snapshot["symbol"] = parquet_path.stem
        latest_snapshot["file_symbol"] = parquet_path.stem
        matches.append(latest_snapshot)

    if not matches:
        return pd.DataFrame(
            columns=[
                "symbol",
                "file_symbol",
                "open",
                "high",
                "low",
                "close",
                "kong_turn_bull",
                "kong_turn_bull_pattern",
                "prev2_time",
                "prev1_time",
                "current_time",
                "prev2_high",
                "prev2_low",
                "prev1_high",
                "prev1_low",
                "current_open",
                "current_close",
                "current_high",
                "current_low",
                "yesterday_lower_low",
                "today_higher_high",
                "today_lower_low",
                "today_close_breaks_yesterday_high",
                "today_bullish_close",
                "three_day_kong_turn_bull",
                "two_day_kong_turn_bull",
            ]
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(pattern_config)
    return combined
