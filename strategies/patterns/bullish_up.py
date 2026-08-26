# 作用：封装 Bullish Up 形态的参数、检测、信号过滤和批量扫描逻辑。
# 内容：从本地 parquet 行情数据中识别两根K线的看涨延续形态，并提供单股与全市场扫描接口。
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_ROOT = Path(r"D:\a_stock_data")
REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class BullishUpConfig:
    """Parameters for a configurable two-candle bullish pattern."""

    body_growth_ratio: float = 1.10
    close_near_high_ratio: float = 0.15
    body_to_range_ratio: float = 0.70
    open_near_prev_close_ratio: float = 0.01
    low_not_weaker_ratio: float = 0.98
    breakout_field: str = "close"
    require_prev_bull: bool = True
    require_curr_bull: bool = True
    require_body_growth: bool = True
    require_breakout: bool = True
    require_close_near_high: bool = True
    require_strong_body: bool = True
    require_open_above_prev_low: bool = False
    require_gap_stability: bool = True
    require_low_not_weaker: bool = True
    require_low_below_prev_low: bool = False
    require_volume_confirmation: bool = False
    volume_column: str = "volume"
    volume_ma_window: int = 5
    volume_multiplier: float = 1.20


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _bar_range(df: pd.DataFrame) -> pd.Series:
    raw_range = df["high"] - df["low"]
    return raw_range.where(raw_range > 0, 1e-9)


def detect_bullish_up(
    df: pd.DataFrame,
    config: BullishUpConfig | None = None,
) -> pd.Series:
    """
    Detect a two-candle bullish continuation pattern.

    Pattern summary:
    - Previous bar is bullish.
    - Current bar is bullish and stronger than the previous bar.
    - Current close is near its high and breaks the previous high or close.
    - Optional gap stability, low stability, and volume confirmation filters.
    """

    pattern_config = config or BullishUpConfig()
    required_columns = list(REQUIRED_COLUMNS)
    if pattern_config.require_volume_confirmation:
        required_columns.append(pattern_config.volume_column)
    _validate_columns(df, required_columns)

    prev_open = df["open"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    curr_body = (df["close"] - df["open"]).abs()
    prev_body = (prev_close - prev_open).abs()
    curr_range = _bar_range(df)

    prev_bull = prev_close > prev_open
    curr_bull = df["close"] > df["open"]
    curr_stronger = curr_body > prev_body * pattern_config.body_growth_ratio

    if pattern_config.breakout_field == "high":
        breakout_reference = prev_high
    elif pattern_config.breakout_field == "close":
        breakout_reference = prev_close
    else:
        raise ValueError("breakout_field must be either 'high' or 'close'")

    breakout = df["close"] > breakout_reference
    close_near_high = (df["high"] - df["close"]) / curr_range < pattern_config.close_near_high_ratio
    strong_body = curr_body / curr_range > pattern_config.body_to_range_ratio

    conditions: list[pd.Series] = []

    if pattern_config.require_prev_bull:
        conditions.append(prev_bull)
    if pattern_config.require_curr_bull:
        conditions.append(curr_bull)
    if pattern_config.require_body_growth:
        conditions.append(curr_stronger)
    if pattern_config.require_breakout:
        conditions.append(breakout)
    if pattern_config.require_close_near_high:
        conditions.append(close_near_high)
    if pattern_config.require_strong_body:
        conditions.append(strong_body)
    if pattern_config.require_open_above_prev_low:
        conditions.append(df["open"] > prev_low)

    if pattern_config.require_gap_stability:
        open_near_prev_close = (df["open"] - prev_close).abs() / prev_close.abs().where(prev_close.abs() > 0, 1e-9)
        conditions.append(open_near_prev_close < pattern_config.open_near_prev_close_ratio)

    if pattern_config.require_low_not_weaker:
        conditions.append(df["low"] >= prev_low * pattern_config.low_not_weaker_ratio)
    if pattern_config.require_low_below_prev_low:
        conditions.append(df["low"] < prev_low)

    if pattern_config.require_volume_confirmation:
        volume_series = df[pattern_config.volume_column]
        volume_ma = volume_series.rolling(pattern_config.volume_ma_window).mean()
        conditions.append(volume_series > volume_ma * pattern_config.volume_multiplier)

    signal = pd.Series(True, index=df.index, name="bullish_up")
    for condition in conditions:
        signal &= condition.fillna(False)

    return signal


def filter_bullish_up_signals(
    df: pd.DataFrame,
    config: BullishUpConfig | None = None,
) -> pd.DataFrame:
    """Return rows where the bullish-up pattern is triggered."""

    signal = detect_bullish_up(df, config=config)
    result = df.loc[signal].copy()
    result["bullish_up"] = True
    return result


def load_stock_bars(
    symbol: str,
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    """Load a stock parquet file from the local data lake."""

    data_path = Path(data_root) / period / f"{symbol}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_parquet(data_path).sort_index()


def scan_period_for_bullish_up(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: BullishUpConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Scan all parquet files for a period and return matching rows.

    The result can be used directly in stock screening or fed into a backtest.
    """

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    matches: list[pd.DataFrame] = []
    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path).sort_index()
        signal_rows = filter_bullish_up_signals(df, config=config)
        if signal_rows.empty:
            continue

        signal_rows = signal_rows.copy()
        if "symbol" not in signal_rows.columns:
            signal_rows["symbol"] = parquet_path.stem
        signal_rows["file_symbol"] = parquet_path.stem
        matches.append(signal_rows)

    if not matches:
        return pd.DataFrame(columns=["symbol", "file_symbol", "bullish_up"])

    combined = pd.concat(matches).sort_index()
    combined.attrs["config"] = asdict(config or BullishUpConfig())
    return combined
