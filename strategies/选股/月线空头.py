# 作用：封装月线空头形态的参数、检测、分析和最新一根月K评估逻辑。
# 内容：从本地 parquet 月线数据中识别两根K线的看跌延续形态，并提供最新完成月K的判定接口。
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from strategies.patterns.bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS
except ModuleNotFoundError:
    from patterns.bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS

from strategies.选股.pinbar import StrictPinBarConfig, analyze_strict_pinbar


@dataclass(frozen=True)
class MonthlyBearishConfig:
    """Parameters for a configurable two-candle monthly bearish pattern."""

    body_growth_ratio: float = 1.10
    close_near_low_ratio: float = 0.15
    body_to_range_ratio: float = 0.70
    open_near_prev_close_ratio: float = 0.01
    high_not_stronger_ratio: float = 1.02
    breakdown_field: str = "close"
    require_prev_bear: bool = True
    require_curr_bear: bool = True
    require_body_growth: bool = True
    require_breakdown: bool = True
    require_close_near_low: bool = True
    require_strong_body: bool = True
    require_open_below_prev_high: bool = False
    require_gap_stability: bool = True
    require_high_not_stronger: bool = True
    require_high_above_prev_high: bool = False
    require_volume_confirmation: bool = False
    volume_column: str = "volume"
    volume_ma_window: int = 5
    volume_multiplier: float = 1.20
    exclude_if_current_strict_pinbar: bool = True
    current_pinbar_config: StrictPinBarConfig = field(default_factory=StrictPinBarConfig)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _bar_range(df: pd.DataFrame) -> pd.Series:
    raw_range = df["high"] - df["low"]
    return raw_range.where(raw_range > 0, 1e-9)


def _normalize_reference_time(reference_time: pd.Timestamp | None = None) -> pd.Timestamp:
    if reference_time is None:
        return pd.Timestamp.now(tz="Asia/Shanghai")

    current_time = pd.Timestamp(reference_time)
    if current_time.tzinfo is None:
        return current_time.tz_localize("Asia/Shanghai")
    return current_time.tz_convert("Asia/Shanghai")


def _dedupe_to_last_bar_per_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    month_key = df.index.to_series().dt.strftime("%Y-%m")
    return df.groupby(month_key, sort=False).tail(1).copy()


def prepare_latest_monthly_screen_frame(
    df: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only completed monthly bars for the latest-pattern evaluation."""

    current_time = _normalize_reference_time(reference_time)
    current_day = current_time.tz_localize(None).normalize() if current_time.tzinfo else current_time.normalize()

    monthly_df = df.sort_index().loc[df.sort_index().index.normalize() <= current_day].copy()
    monthly_df = _dedupe_to_last_bar_per_month(monthly_df)
    if len(monthly_df) < 2:
        return monthly_df

    latest_bar_time = pd.Timestamp(monthly_df.index[-1])

    # The current month bar is still rolling inside the month, so only treat it
    # as complete after we have moved into the next calendar month.
    if latest_bar_time.year == current_time.year and latest_bar_time.month == current_time.month:
        return monthly_df.iloc[:-1].copy()

    return monthly_df


def analyze_monthly_bearish(
    df: pd.DataFrame,
    config: MonthlyBearishConfig | None = None,
) -> pd.DataFrame:
    """
    Analyze a two-candle bearish continuation pattern on monthly bars.

    Pattern summary:
    - Previous bar is bearish.
    - Current bar is bearish and stronger than the previous bar.
    - Current close is near its low and breaks the previous low or close.
    - Optional gap stability, high stability, and volume confirmation filters.
    """

    pattern_config = config or MonthlyBearishConfig()
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

    prev_bear = prev_close < prev_open
    curr_bear = df["close"] < df["open"]
    stronger_body = curr_body > prev_body * pattern_config.body_growth_ratio

    if pattern_config.breakdown_field == "low":
        breakdown_reference = prev_low
    elif pattern_config.breakdown_field == "close":
        breakdown_reference = prev_close
    else:
        raise ValueError("breakdown_field must be either 'low' or 'close'")

    breakdown = df["close"] < breakdown_reference
    close_near_low = (df["close"] - df["low"]) / curr_range < pattern_config.close_near_low_ratio
    strong_body = curr_body / curr_range > pattern_config.body_to_range_ratio
    open_below_prev_high = df["open"] < prev_high
    gap_stability = (
        (df["open"] - prev_close).abs()
        / prev_close.abs().where(prev_close.abs() > 0, 1e-9)
        < pattern_config.open_near_prev_close_ratio
    )
    high_not_stronger = df["high"] <= prev_high * pattern_config.high_not_stronger_ratio
    high_above_prev_high = df["high"] > prev_high

    if pattern_config.require_volume_confirmation:
        volume_series = df[pattern_config.volume_column]
        volume_ma = volume_series.rolling(pattern_config.volume_ma_window).mean()
        volume_confirmation = volume_series > volume_ma * pattern_config.volume_multiplier
    else:
        volume_confirmation = pd.Series(True, index=df.index, dtype=bool)

    current_pinbar_analysis = analyze_strict_pinbar(df, config=pattern_config.current_pinbar_config)
    current_strict_pinbar = current_pinbar_analysis["strict_pinbar"]
    current_pinbar_direction = current_pinbar_analysis["pinbar_direction"]

    signal = pd.Series(True, index=df.index, dtype=bool, name="monthly_bearish")
    if pattern_config.require_prev_bear:
        signal &= prev_bear.fillna(False)
    if pattern_config.require_curr_bear:
        signal &= curr_bear.fillna(False)
    if pattern_config.require_body_growth:
        signal &= stronger_body.fillna(False)
    if pattern_config.require_breakdown:
        signal &= breakdown.fillna(False)
    if pattern_config.require_close_near_low:
        signal &= close_near_low.fillna(False)
    if pattern_config.require_strong_body:
        signal &= strong_body.fillna(False)
    if pattern_config.require_open_below_prev_high:
        signal &= open_below_prev_high.fillna(False)
    if pattern_config.require_gap_stability:
        signal &= gap_stability.fillna(False)
    if pattern_config.require_high_not_stronger:
        signal &= high_not_stronger.fillna(False)
    if pattern_config.require_high_above_prev_high:
        signal &= high_above_prev_high.fillna(False)
    if pattern_config.require_volume_confirmation:
        signal &= volume_confirmation.fillna(False)
    if pattern_config.exclude_if_current_strict_pinbar:
        signal &= ~current_strict_pinbar

    index_series = df.index.to_series()
    analysis = pd.DataFrame(index=df.index)
    analysis["monthly_bearish"] = signal
    analysis["prev_time"] = index_series.shift(1)
    analysis["current_time"] = index_series
    analysis["prev_open"] = prev_open
    analysis["prev_high"] = prev_high
    analysis["prev_low"] = prev_low
    analysis["prev_close"] = prev_close
    analysis["current_open"] = df["open"]
    analysis["current_high"] = df["high"]
    analysis["current_low"] = df["low"]
    analysis["current_close"] = df["close"]
    analysis["prev_bear"] = prev_bear
    analysis["curr_bear"] = curr_bear
    analysis["stronger_body"] = stronger_body
    analysis["breakdown"] = breakdown
    analysis["close_near_low"] = close_near_low
    analysis["strong_body"] = strong_body
    analysis["open_below_prev_high"] = open_below_prev_high
    analysis["gap_stability"] = gap_stability
    analysis["high_not_stronger"] = high_not_stronger
    analysis["high_above_prev_high"] = high_above_prev_high
    analysis["volume_confirmation"] = volume_confirmation
    analysis["current_strict_pinbar"] = current_strict_pinbar
    analysis["current_pinbar_direction"] = current_pinbar_direction
    analysis.attrs["config"] = asdict(pattern_config)
    return analysis


def detect_monthly_bearish(
    df: pd.DataFrame,
    config: MonthlyBearishConfig | None = None,
) -> pd.Series:
    """Return a boolean signal for the monthly bearish pattern."""

    return analyze_monthly_bearish(df, config=config)["monthly_bearish"].rename("monthly_bearish")


def filter_monthly_bearish_signals(
    df: pd.DataFrame,
    config: MonthlyBearishConfig | None = None,
) -> pd.DataFrame:
    """Return rows where the monthly bearish pattern is triggered."""

    analysis = analyze_monthly_bearish(df, config=config)
    signal = analysis["monthly_bearish"].fillna(False)
    result = df.loc[signal].copy()
    result["monthly_bearish"] = True
    for column in [
        "prev_time",
        "current_time",
        "prev_open",
        "prev_high",
        "prev_low",
        "prev_close",
        "current_open",
        "current_high",
        "current_low",
        "current_close",
        "prev_bear",
        "curr_bear",
        "stronger_body",
        "breakdown",
        "close_near_low",
        "strong_body",
        "open_below_prev_high",
        "gap_stability",
        "high_not_stronger",
        "high_above_prev_high",
        "volume_confirmation",
        "current_strict_pinbar",
        "current_pinbar_direction",
    ]:
        result[column] = analysis.loc[signal, column]
    return result


def evaluate_latest_monthly_bearish(
    df: pd.DataFrame,
    config: MonthlyBearishConfig | None = None,
    reference_time: pd.Timestamp | None = None,
) -> pd.Series:
    """Evaluate the latest completed monthly bar for the bearish pattern."""

    if df.empty:
        raise ValueError("DataFrame is empty")

    prepared_df = prepare_latest_monthly_screen_frame(df, reference_time=reference_time)
    if len(prepared_df) < 2:
        raise ValueError("DataFrame must contain at least two completed monthly bars")

    analysis = analyze_monthly_bearish(prepared_df, config=config)
    latest_snapshot = prepared_df.iloc[-1].copy()
    latest_analysis = analysis.iloc[-1]
    for column in analysis.columns:
        latest_snapshot[column] = latest_analysis[column]
    latest_snapshot.attrs["config"] = analysis.attrs.get("config", {})
    return latest_snapshot


def detect_latest_monthly_bearish(
    df: pd.DataFrame,
    config: MonthlyBearishConfig | None = None,
    reference_time: pd.Timestamp | None = None,
) -> bool:
    """Return whether the latest completed monthly bar matches the pattern."""

    latest_snapshot = evaluate_latest_monthly_bearish(df, config=config, reference_time=reference_time)
    return bool(latest_snapshot["monthly_bearish"])


def scan_period_for_latest_monthly_bearish(
    period: str = "monthly",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: MonthlyBearishConfig | None = None,
    limit: int | None = None,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Scan the latest completed monthly bar of every stock for the bearish pattern."""

    root = Path(data_root) / period
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close", "volume"]
    matches: list[pd.Series] = []
    pattern_config = config or MonthlyBearishConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        if df.empty:
            continue

        try:
            latest_snapshot = evaluate_latest_monthly_bearish(
                df,
                config=pattern_config,
                reference_time=reference_time,
            )
        except ValueError:
            continue

        if not bool(latest_snapshot["monthly_bearish"]):
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
                "monthly_bearish",
                "prev_time",
                "current_time",
                "prev_open",
                "prev_high",
                "prev_low",
                "prev_close",
                "current_open",
                "current_high",
                "current_low",
                "current_close",
                "prev_bear",
                "curr_bear",
                "stronger_body",
                "breakdown",
                "close_near_low",
                "strong_body",
                "open_below_prev_high",
                "gap_stability",
                "high_not_stronger",
                "high_above_prev_high",
                "volume_confirmation",
                "current_strict_pinbar",
                "current_pinbar_direction",
            ]
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(pattern_config)
    return combined
