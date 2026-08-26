from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS, load_stock_bars


@dataclass(frozen=True)
class StrongWeakStrongConfig:
    """Parameters for the three-bar strong -> weak -> strong engulfing pattern."""

    require_k1_bull: bool = True
    require_k2_bear: bool = True
    require_k3_bull: bool = True
    strict_engulf: bool = False


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def analyze_strong_weak_strong(
    df: pd.DataFrame,
    config: StrongWeakStrongConfig | None = None,
) -> pd.DataFrame:
    """
    Analyze a three-candle strong -> weak -> strong reversal pattern.

    Pattern summary:
    - k1 is bullish.
    - k2 is bearish and its real body engulfs k1's real body.
    - k3 is bullish and its real body engulfs k2's real body.

    The signal is attached to k3, which means the current row represents the
    third candle of the sequence.
    """

    pattern_config = config or StrongWeakStrongConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    k1_open = df["open"].shift(2)
    k1_close = df["close"].shift(2)
    k2_open = df["open"].shift(1)
    k2_close = df["close"].shift(1)
    k3_open = df["open"]
    k3_close = df["close"]

    k1_bull = k1_close > k1_open
    k2_bear = k2_close < k2_open
    k3_bull = k3_close > k3_open

    if pattern_config.strict_engulf:
        k2_engulfs_k1 = (k2_open > k1_close) & (k2_close < k1_open)
        k3_engulfs_k2 = (k3_open < k2_close) & (k3_close > k2_open)
    else:
        k2_engulfs_k1 = (k2_open >= k1_close) & (k2_close <= k1_open)
        k3_engulfs_k2 = (k3_open <= k2_close) & (k3_close >= k2_open)

    conditions: list[pd.Series] = [k2_engulfs_k1, k3_engulfs_k2]
    if pattern_config.require_k1_bull:
        conditions.append(k1_bull)
    if pattern_config.require_k2_bear:
        conditions.append(k2_bear)
    if pattern_config.require_k3_bull:
        conditions.append(k3_bull)

    signal = pd.Series(True, index=df.index, name="strong_weak_strong")
    for condition in conditions:
        signal &= condition.fillna(False)

    index_series = df.index.to_series()
    analysis = pd.DataFrame(index=df.index)
    analysis["strong_weak_strong"] = signal
    analysis["k1_time"] = index_series.shift(2)
    analysis["k2_time"] = index_series.shift(1)
    analysis["k3_time"] = index_series
    analysis["k1_open"] = k1_open
    analysis["k1_close"] = k1_close
    analysis["k2_open"] = k2_open
    analysis["k2_close"] = k2_close
    analysis["k3_open"] = k3_open
    analysis["k3_close"] = k3_close
    analysis["k1_bull"] = k1_bull
    analysis["k2_bear"] = k2_bear
    analysis["k3_bull"] = k3_bull
    analysis["k2_engulfs_k1"] = k2_engulfs_k1
    analysis["k3_engulfs_k2"] = k3_engulfs_k2
    analysis.attrs["config"] = asdict(pattern_config)
    return analysis


def detect_strong_weak_strong(
    df: pd.DataFrame,
    config: StrongWeakStrongConfig | None = None,
) -> pd.Series:
    """Return a boolean signal for the three-bar strong -> weak -> strong pattern."""

    return analyze_strong_weak_strong(df, config=config)["strong_weak_strong"].rename("strong_weak_strong")


def filter_strong_weak_strong_signals(
    df: pd.DataFrame,
    config: StrongWeakStrongConfig | None = None,
) -> pd.DataFrame:
    """Return rows where the strong -> weak -> strong pattern is triggered."""

    analysis = analyze_strong_weak_strong(df, config=config)
    signal = analysis["strong_weak_strong"].fillna(False)
    result = df.loc[signal].copy()
    result["strong_weak_strong"] = True
    for column in [
        "k1_time",
        "k2_time",
        "k3_time",
        "k1_open",
        "k1_close",
        "k2_open",
        "k2_close",
        "k3_open",
        "k3_close",
        "k2_engulfs_k1",
        "k3_engulfs_k2",
    ]:
        result[column] = analysis.loc[signal, column]
    return result


def evaluate_latest_strong_weak_strong(
    df: pd.DataFrame,
    config: StrongWeakStrongConfig | None = None,
) -> pd.Series:
    """Evaluate only the latest bar as k3 of the pattern."""

    if df.empty:
        raise ValueError("DataFrame is empty")

    analysis = analyze_strong_weak_strong(df, config=config)
    latest_snapshot = df.iloc[-1].copy()
    latest_analysis = analysis.iloc[-1]
    for column in analysis.columns:
        latest_snapshot[column] = latest_analysis[column]
    latest_snapshot.attrs["config"] = analysis.attrs.get("config", {})
    return latest_snapshot


def detect_latest_strong_weak_strong(
    df: pd.DataFrame,
    config: StrongWeakStrongConfig | None = None,
) -> bool:
    """Return whether the newest bar matches the strong -> weak -> strong pattern."""

    latest_snapshot = evaluate_latest_strong_weak_strong(df, config=config)
    return bool(latest_snapshot["strong_weak_strong"])


def scan_period_for_strong_weak_strong(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: StrongWeakStrongConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan a period folder and return all matching strong -> weak -> strong bars."""

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
        signal_rows = filter_strong_weak_strong_signals(df, config=config)
        if signal_rows.empty:
            continue

        signal_rows = signal_rows.copy()
        if "symbol" not in signal_rows.columns:
            signal_rows["symbol"] = parquet_path.stem
        signal_rows["file_symbol"] = parquet_path.stem
        matches.append(signal_rows)

    if not matches:
        return pd.DataFrame(columns=["symbol", "file_symbol", "strong_weak_strong"])

    combined = pd.concat(matches).sort_index()
    combined.attrs["config"] = asdict(config or StrongWeakStrongConfig())
    return combined


def scan_period_for_latest_strong_weak_strong(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: StrongWeakStrongConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan the latest bar of every stock for the strong -> weak -> strong pattern."""

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
    matches: list[pd.Series] = []
    pattern_config = config or StrongWeakStrongConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        if df.empty:
            continue
        latest_snapshot = evaluate_latest_strong_weak_strong(df, config=pattern_config)
        if not bool(latest_snapshot["strong_weak_strong"]):
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
                "strong_weak_strong",
                "k1_time",
                "k2_time",
                "k3_time",
                "k1_open",
                "k1_close",
                "k2_open",
                "k2_close",
                "k3_open",
                "k3_close",
                "k2_engulfs_k1",
                "k3_engulfs_k2",
            ]
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(pattern_config)
    return combined
