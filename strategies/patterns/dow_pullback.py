# 作用：封装 Dow Pullback（道氏上升趋势中的回调）形态识别与筛选逻辑。
# 内容：识别低点抬高、高点抬高后的回调结构，支持历史信号分析、最新一根K线筛选以及股票池批量扫描。
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS, load_stock_bars


@dataclass(frozen=True)
class DowPullbackConfig:
    """Parameters for an intact Dow-style uptrend that is currently pulling back."""

    swing_window: int = 3
    max_lookback_bars: int = 120
    min_higher_high_ratio: float = 0.02
    min_higher_low_ratio: float = 0.01
    min_pullback_bars: int = 2
    min_pullback_retracement: float = 0.15
    max_pullback_retracement: float = 0.70
    structure_break_tolerance: float = 0.01
    require_down_close_bias: bool = True
    down_close_window: int = 3
    min_down_close_count: int = 2
    require_close_below_last_high: bool = True
    screen_latest_by_last_four_swings: bool = True
    # Toggle for the latest-bar screener:
    # when enabled, the current close must stay in the lower portion of the
    # latest low2 -> high2 swing range.
    require_price_in_lower_half: bool = False
    # 0.50 means "lower half of the latest low2 -> high2 range".
    lower_half_threshold: float = 0.50
    screening_window_bars: int | None = None


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _detect_pivot_flags(df: pd.DataFrame, swing_window: int) -> tuple[pd.Series, pd.Series]:
    highs = df["high"].reset_index(drop=True)
    lows = df["low"].reset_index(drop=True)
    pivot_high = pd.Series(False, index=df.index, dtype=bool)
    pivot_low = pd.Series(False, index=df.index, dtype=bool)

    if swing_window < 1 or len(df) < swing_window * 2 + 1:
        return pivot_high, pivot_low

    for bar_index in range(swing_window, len(df) - swing_window):
        high_slice = highs.iloc[bar_index - swing_window : bar_index + swing_window + 1]
        low_slice = lows.iloc[bar_index - swing_window : bar_index + swing_window + 1]
        current_high = highs.iat[bar_index]
        current_low = lows.iat[bar_index]

        if current_high == high_slice.max() and int((high_slice == current_high).sum()) == 1:
            pivot_high.iat[bar_index] = True
        if current_low == low_slice.min() and int((low_slice == current_low).sum()) == 1:
            pivot_low.iat[bar_index] = True

    return pivot_high, pivot_low


def _build_confirmed_swings(df: pd.DataFrame, config: DowPullbackConfig) -> pd.DataFrame:
    pivot_high, pivot_low = _detect_pivot_flags(df, config.swing_window)
    swings: list[dict[str, object]] = []

    for bar_index, index_label in enumerate(df.index):
        high_flag = bool(pivot_high.iat[bar_index])
        low_flag = bool(pivot_low.iat[bar_index])

        if high_flag and low_flag:
            continue

        if high_flag:
            swings.append(
                {
                    "pivot_index": bar_index,
                    "pivot_time": index_label,
                    "confirmed_index": bar_index + config.swing_window,
                    "confirmed_time": df.index[bar_index + config.swing_window],
                    "swing_type": "H",
                    "swing_price": float(df["high"].iat[bar_index]),
                }
            )
        elif low_flag:
            swings.append(
                {
                    "pivot_index": bar_index,
                    "pivot_time": index_label,
                    "confirmed_index": bar_index + config.swing_window,
                    "confirmed_time": df.index[bar_index + config.swing_window],
                    "swing_type": "L",
                    "swing_price": float(df["low"].iat[bar_index]),
                }
            )

    if not swings:
        return pd.DataFrame(
            columns=[
                "pivot_index",
                "pivot_time",
                "confirmed_index",
                "confirmed_time",
                "swing_type",
                "swing_price",
            ]
        )

    compressed: list[dict[str, object]] = []
    for swing in swings:
        if not compressed:
            compressed.append(swing)
            continue

        previous_swing = compressed[-1]
        if swing["swing_type"] == previous_swing["swing_type"]:
            if swing["swing_type"] == "H" and swing["swing_price"] >= previous_swing["swing_price"]:
                compressed[-1] = swing
            elif swing["swing_type"] == "L" and swing["swing_price"] <= previous_swing["swing_price"]:
                compressed[-1] = swing
        else:
            compressed.append(swing)

    return pd.DataFrame(compressed)


def _minimum_screening_window(config: DowPullbackConfig) -> int:
    if config.screen_latest_by_last_four_swings:
        base_window = config.swing_window * 12 + config.down_close_window + 40
    else:
        base_window = config.max_lookback_bars + config.swing_window * 6 + config.down_close_window + 10
    return max(base_window, config.swing_window * 2 + 1)


def _trim_for_latest_screening(
    df: pd.DataFrame,
    config: DowPullbackConfig,
    recent_bars: int | None = None,
) -> pd.DataFrame:
    requested_bars = recent_bars or config.screening_window_bars
    effective_window = max(requested_bars or 0, _minimum_screening_window(config))
    return df.sort_index().tail(effective_window)


def _evaluate_dow_pullback_at_bar(
    active_swings: list[dict[str, object]],
    close_series: pd.Series,
    low_series: pd.Series,
    bar_index: int,
    config: DowPullbackConfig,
    use_last_four_only: bool = False,
) -> dict[str, float | int | bool] | None:
    if len(active_swings) < 4:
        return None

    if use_last_four_only:
        recent_swings = active_swings[-4:]
    else:
        recent_swings = [
            swing
            for swing in active_swings
            if bar_index - int(swing["pivot_index"]) <= config.max_lookback_bars
        ]
    if len(recent_swings) < 4:
        return None

    last_four = recent_swings[-4:]
    swing_types = [str(swing["swing_type"]) for swing in last_four]
    if swing_types != ["L", "H", "L", "H"]:
        return None

    low1, high1, higher_low, higher_high = last_four
    low1_price = float(low1["swing_price"])
    high1_price = float(high1["swing_price"])
    higher_low_price = float(higher_low["swing_price"])
    higher_high_price = float(higher_high["swing_price"])
    higher_high_bar = int(higher_high["pivot_index"])

    if higher_high_price <= high1_price * (1 + config.min_higher_high_ratio):
        return None
    if higher_low_price <= low1_price * (1 + config.min_higher_low_ratio):
        return None
    if bar_index <= higher_high_bar:
        return None

    bars_since_last_high = bar_index - higher_high_bar
    if bars_since_last_high < config.min_pullback_bars:
        return None

    swing_advance = higher_high_price - higher_low_price
    if swing_advance <= 0:
        return None

    current_close = float(close_series.iat[bar_index])
    current_low = float(low_series.iat[bar_index])
    retracement = (higher_high_price - current_close) / swing_advance
    price_zone_ratio = (current_close - higher_low_price) / swing_advance
    lowest_since_last_high = float(low_series.iloc[higher_high_bar : bar_index + 1].min())

    if retracement < config.min_pullback_retracement:
        return None
    if retracement > config.max_pullback_retracement:
        return None
    if lowest_since_last_high < higher_low_price * (1 - config.structure_break_tolerance):
        return None
    if current_low < higher_low_price * (1 - config.structure_break_tolerance):
        return None

    if config.require_close_below_last_high and current_close >= higher_high_price:
        return None

    # Dynamic zone filter switch for stock screening.
    # Example: threshold 0.50 keeps the close in the lower half of low2 -> high2.
    if config.require_price_in_lower_half and price_zone_ratio > config.lower_half_threshold:
        return None

    if config.require_down_close_bias:
        window = min(config.down_close_window, bar_index + 1)
        recent_closes = close_series.iloc[bar_index - window + 1 : bar_index + 1]
        down_close_count = int((recent_closes.diff() < 0).sum())
        if down_close_count < config.min_down_close_count:
            return None

    return {
        "dow_pullback": True,
        "low1_price": low1_price,
        "high1_price": high1_price,
        "low2_price": higher_low_price,
        "high2_price": higher_high_price,
        "price_zone_ratio": price_zone_ratio,
        "last_higher_low": higher_low_price,
        "last_swing_high": higher_high_price,
        "pullback_retracement": retracement,
        "bars_since_last_high": bars_since_last_high,
    }


def analyze_dow_pullback(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
) -> pd.DataFrame:
    """
    Analyze whether each bar sits inside an intact Dow-style uptrend pullback.

    The signal looks for the confirmed swing sequence:
    low1 -> high1 -> higher low -> higher high
    and then checks whether the current bar is pulling back without breaking
    the latest higher low.
    """

    structure_config = config or DowPullbackConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    analysis = pd.DataFrame(index=df.index)
    analysis["dow_pullback"] = False
    analysis["last_higher_low"] = pd.NA
    analysis["last_swing_high"] = pd.NA
    analysis["pullback_retracement"] = pd.NA
    analysis["price_zone_ratio"] = pd.NA
    analysis["bars_since_last_high"] = pd.NA

    swings = _build_confirmed_swings(df, structure_config)
    if swings.empty:
        analysis.attrs["config"] = asdict(structure_config)
        analysis.attrs["swings"] = swings
        return analysis

    active_swings: list[dict[str, object]] = []
    swing_pointer = 0
    close_series = df["close"].reset_index(drop=True)
    low_series = df["low"].reset_index(drop=True)

    for bar_index in range(len(df)):
        while swing_pointer < len(swings) and int(swings.iloc[swing_pointer]["confirmed_index"]) <= bar_index:
            active_swings.append(swings.iloc[swing_pointer].to_dict())
            swing_pointer += 1

        signal_details = _evaluate_dow_pullback_at_bar(
            active_swings=active_swings,
            close_series=close_series,
            low_series=low_series,
            bar_index=bar_index,
            config=structure_config,
            use_last_four_only=False,
        )
        if signal_details is None:
            continue

        analysis.iat[bar_index, analysis.columns.get_loc("dow_pullback")] = True
        analysis.iat[bar_index, analysis.columns.get_loc("last_higher_low")] = signal_details["last_higher_low"]
        analysis.iat[bar_index, analysis.columns.get_loc("last_swing_high")] = signal_details["last_swing_high"]
        analysis.iat[bar_index, analysis.columns.get_loc("pullback_retracement")] = signal_details["pullback_retracement"]
        analysis.iat[bar_index, analysis.columns.get_loc("price_zone_ratio")] = signal_details["price_zone_ratio"]
        analysis.iat[bar_index, analysis.columns.get_loc("bars_since_last_high")] = signal_details["bars_since_last_high"]

    analysis.attrs["config"] = asdict(structure_config)
    analysis.attrs["swings"] = swings
    return analysis


def detect_dow_pullback(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
) -> pd.Series:
    """Return a boolean signal for intact-uptrend pullback bars."""

    return analyze_dow_pullback(df, config=config)["dow_pullback"].rename("dow_pullback")


def filter_dow_pullback_signals(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
) -> pd.DataFrame:
    """Return rows where the Dow-style pullback structure is active."""

    analysis = analyze_dow_pullback(df, config=config)
    signal = analysis["dow_pullback"].fillna(False)
    result = df.loc[signal].copy()
    result["dow_pullback"] = True
    result["last_higher_low"] = analysis.loc[signal, "last_higher_low"]
    result["last_swing_high"] = analysis.loc[signal, "last_swing_high"]
    result["pullback_retracement"] = analysis.loc[signal, "pullback_retracement"]
    result["price_zone_ratio"] = analysis.loc[signal, "price_zone_ratio"]
    result["bars_since_last_high"] = analysis.loc[signal, "bars_since_last_high"]
    return result


def evaluate_latest_dow_pullback(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
    recent_bars: int | None = None,
) -> pd.Series:
    """
    Evaluate only the latest bar.

    This is intended for daily stock screening where we only care whether the
    newest bar is currently sitting inside an intact Dow-style pullback.
    """

    structure_config = config or DowPullbackConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    screened_df = _trim_for_latest_screening(df, structure_config, recent_bars=recent_bars)
    if screened_df.empty:
        raise ValueError("DataFrame is empty")

    swings = _build_confirmed_swings(screened_df, structure_config)
    close_series = screened_df["close"].reset_index(drop=True)
    low_series = screened_df["low"].reset_index(drop=True)
    latest_bar_index = len(screened_df) - 1

    if swings.empty:
        signal_details = None
    else:
        active_swings = swings.loc[swings["confirmed_index"] <= latest_bar_index].to_dict("records")
        signal_details = _evaluate_dow_pullback_at_bar(
            active_swings=active_swings,
            close_series=close_series,
            low_series=low_series,
            bar_index=latest_bar_index,
            config=structure_config,
            use_last_four_only=structure_config.screen_latest_by_last_four_swings,
        )

    latest_snapshot = screened_df.iloc[-1].copy()
    latest_snapshot["dow_pullback"] = signal_details is not None
    latest_snapshot["low1_price"] = pd.NA if signal_details is None else signal_details["low1_price"]
    latest_snapshot["high1_price"] = pd.NA if signal_details is None else signal_details["high1_price"]
    latest_snapshot["low2_price"] = pd.NA if signal_details is None else signal_details["low2_price"]
    latest_snapshot["high2_price"] = pd.NA if signal_details is None else signal_details["high2_price"]
    latest_snapshot["price_zone_ratio"] = pd.NA if signal_details is None else signal_details["price_zone_ratio"]
    latest_snapshot["last_higher_low"] = pd.NA if signal_details is None else signal_details["last_higher_low"]
    latest_snapshot["last_swing_high"] = pd.NA if signal_details is None else signal_details["last_swing_high"]
    latest_snapshot["pullback_retracement"] = pd.NA if signal_details is None else signal_details["pullback_retracement"]
    latest_snapshot["bars_since_last_high"] = pd.NA if signal_details is None else signal_details["bars_since_last_high"]
    latest_snapshot["screen_window_bars"] = len(screened_df)
    latest_snapshot.attrs["config"] = asdict(structure_config)
    latest_snapshot.attrs["swings"] = swings
    return latest_snapshot


def detect_latest_dow_pullback(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
    recent_bars: int | None = None,
) -> bool:
    """Return whether the newest bar matches the Dow-style pullback structure."""

    latest_snapshot = evaluate_latest_dow_pullback(df, config=config, recent_bars=recent_bars)
    return bool(latest_snapshot["dow_pullback"])


def get_latest_confirmed_swings(
    df: pd.DataFrame,
    config: DowPullbackConfig | None = None,
    recent_bars: int | None = None,
    count: int = 4,
) -> pd.DataFrame:
    """
    Return the latest confirmed swings used by the screener.

    This helps inspect the exact four pivots that the latest-bar screen is
    comparing against the current price.
    """

    structure_config = config or DowPullbackConfig()
    _validate_columns(df, REQUIRED_COLUMNS)

    screened_df = _trim_for_latest_screening(df, structure_config, recent_bars=recent_bars)
    swings = _build_confirmed_swings(screened_df, structure_config)
    if swings.empty:
        return swings

    latest_bar_index = len(screened_df) - 1
    confirmed_swings = swings.loc[swings["confirmed_index"] <= latest_bar_index].copy()
    if confirmed_swings.empty:
        return confirmed_swings

    return confirmed_swings.tail(count).reset_index(drop=True)


def scan_period_for_dow_pullback(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: DowPullbackConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Scan an entire period folder and return matching pullback structures."""

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    matches: list[pd.DataFrame] = []
    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path).sort_index()
        signal_rows = filter_dow_pullback_signals(df, config=config)
        if signal_rows.empty:
            continue

        signal_rows = signal_rows.copy()
        if "symbol" not in signal_rows.columns:
            signal_rows["symbol"] = parquet_path.stem
        signal_rows["file_symbol"] = parquet_path.stem
        matches.append(signal_rows)

    if not matches:
        return pd.DataFrame(
            columns=[
                "symbol",
                "file_symbol",
                "dow_pullback",
                "last_higher_low",
                "last_swing_high",
                "pullback_retracement",
                "bars_since_last_high",
            ]
        )

    combined = pd.concat(matches).sort_index()
    combined.attrs["config"] = asdict(config or DowPullbackConfig())
    return combined


def scan_period_for_latest_dow_pullback(
    period: str = "daily",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: DowPullbackConfig | None = None,
    recent_bars: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Scan the latest bar of every stock in a period folder.

    Compared with the backtest-oriented scan, this only evaluates the newest
    bar for each symbol and is meant for end-of-day stock screening.
    """

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close", "volume"]
    matches: list[pd.Series] = []
    structure_config = config or DowPullbackConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        latest_snapshot = evaluate_latest_dow_pullback(df, config=structure_config, recent_bars=recent_bars)
        if not bool(latest_snapshot["dow_pullback"]):
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
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(structure_config)
    return combined
