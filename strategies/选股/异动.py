from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

try:
    from strategies.patterns.bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS
except ModuleNotFoundError:
    from patterns.bullish_up import DEFAULT_DATA_ROOT, REQUIRED_COLUMNS

from patterns import load_stock_bars
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


A_SHARE_MARKET_CLOSE_HOUR = 15


@dataclass(frozen=True)
class AbnormalMoveConfig:
    """Recent abnormal-move filter based on daily close-to-close gains."""

    lookback_bars: int = 20
    single_day_gain_threshold: float = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen daily abnormal-move stocks on the latest completed day.")
    parser.add_argument("--symbol", default="sz300058", help="Single stock file_symbol to preview.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N files.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional reference date in YYYY-MM-DD. Used to decide whether today's daily bar is completed.",
    )
    parser.add_argument("--lookback", type=int, default=20, help="How many completed daily bars to look back.")
    parser.add_argument("--threshold", type=float, default=0.08, help="Single-day gain threshold. Default is 0.08.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to screening_results/abnormal_move_latest.csv.",
    )
    return parser.parse_args()


def build_config(lookback: int = 20, threshold: float = 0.08) -> AbnormalMoveConfig:
    return AbnormalMoveConfig(
        lookback_bars=lookback,
        single_day_gain_threshold=threshold,
    )


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


def prepare_latest_daily_screen_frame(
    df: pd.DataFrame,
    reference_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    prepared = df.sort_index()
    if prepared.empty:
        return prepared

    current_time = _normalize_reference_time(reference_time)
    current_day = current_time.tz_localize(None).normalize() if current_time.tzinfo else current_time.normalize()
    prepared = prepared.loc[prepared.index.normalize() <= current_day].copy()
    if prepared.empty:
        return prepared

    latest_bar_time = pd.Timestamp(prepared.index[-1]).normalize()
    market_close_time = current_time.normalize() + pd.Timedelta(hours=A_SHARE_MARKET_CLOSE_HOUR)
    if latest_bar_time == current_day and current_time < market_close_time:
        return prepared.iloc[:-1].copy()

    return prepared


def evaluate_latest_abnormal_move(
    df: pd.DataFrame,
    config: AbnormalMoveConfig | None = None,
    reference_time: str | pd.Timestamp | None = None,
) -> pd.Series:
    if df.empty:
        raise ValueError("DataFrame is empty")

    pattern_config = config or AbnormalMoveConfig()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    prepared_df = prepare_latest_daily_screen_frame(df, reference_time=reference_time)
    if len(prepared_df) < 2:
        raise ValueError("DataFrame must contain at least two completed daily bars")

    window_df = prepared_df.tail(pattern_config.lookback_bars).copy()
    prev_close = prepared_df["close"].shift(1)
    window_prev_close = prev_close.loc[window_df.index]
    daily_gain_pct = window_df["close"] / window_prev_close - 1
    hit_mask = (daily_gain_pct > pattern_config.single_day_gain_threshold).fillna(False)
    max_gain_pct = daily_gain_pct.max(skipna=True)

    latest_snapshot = prepared_df.iloc[-1].copy()
    latest_snapshot["abnormal_move"] = bool(hit_mask.any())
    latest_snapshot["window_start_time"] = window_df.index[0]
    latest_snapshot["window_end_time"] = window_df.index[-1]
    latest_snapshot["lookback_bars"] = pattern_config.lookback_bars
    latest_snapshot["single_day_gain_threshold"] = pattern_config.single_day_gain_threshold
    latest_snapshot["hit_count"] = int(hit_mask.sum())
    latest_snapshot["max_gain_pct"] = max_gain_pct

    if bool(hit_mask.any()):
        event_time = daily_gain_pct.loc[hit_mask].idxmax()
        event_row = prepared_df.loc[event_time]
        latest_snapshot["event_time"] = event_time
        latest_snapshot["event_prev_close"] = prev_close.loc[event_time]
        latest_snapshot["event_open"] = event_row["open"]
        latest_snapshot["event_high"] = event_row["high"]
        latest_snapshot["event_low"] = event_row["low"]
        latest_snapshot["event_close"] = event_row["close"]
        latest_snapshot["event_gain_pct"] = daily_gain_pct.loc[event_time]
    else:
        latest_snapshot["event_time"] = pd.NaT
        latest_snapshot["event_prev_close"] = pd.NA
        latest_snapshot["event_open"] = pd.NA
        latest_snapshot["event_high"] = pd.NA
        latest_snapshot["event_low"] = pd.NA
        latest_snapshot["event_close"] = pd.NA
        latest_snapshot["event_gain_pct"] = pd.NA

    latest_snapshot.attrs["config"] = asdict(pattern_config)
    return latest_snapshot


def detect_latest_abnormal_move(
    df: pd.DataFrame,
    config: AbnormalMoveConfig | None = None,
    reference_time: str | pd.Timestamp | None = None,
) -> bool:
    latest_snapshot = evaluate_latest_abnormal_move(df, config=config, reference_time=reference_time)
    return bool(latest_snapshot["abnormal_move"])


def scan_period_for_latest_abnormal_move(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    config: AbnormalMoveConfig | None = None,
    limit: int | None = None,
    reference_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    root = Path(data_root) / "daily"
    if not root.exists():
        raise FileNotFoundError(f"Period folder not found: {root}")

    screen_columns = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
    matches: list[pd.Series] = []
    pattern_config = config or AbnormalMoveConfig()

    for index, parquet_path in enumerate(sorted(root.glob("*.parquet"))):
        if limit is not None and index >= limit:
            break

        df = pd.read_parquet(parquet_path, columns=screen_columns).sort_index()
        if df.empty:
            continue

        try:
            latest_snapshot = evaluate_latest_abnormal_move(
                df,
                config=pattern_config,
                reference_time=reference_time,
            )
        except ValueError:
            continue

        if not bool(latest_snapshot["abnormal_move"]):
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
                "abnormal_move",
                "window_start_time",
                "window_end_time",
                "lookback_bars",
                "single_day_gain_threshold",
                "hit_count",
                "max_gain_pct",
                "event_time",
                "event_prev_close",
                "event_open",
                "event_high",
                "event_low",
                "event_close",
                "event_gain_pct",
            ]
        )

    combined = pd.DataFrame(matches)
    if "symbol" in combined.columns:
        combined = combined.sort_values(["symbol"])
    combined.attrs["config"] = asdict(pattern_config)
    return combined


def build_default_output_path() -> Path:
    return SCREENING_RESULTS_DIR / "abnormal_move_latest.csv"


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    config = build_config(lookback=args.lookback, threshold=args.threshold)
    reference_time = _normalize_reference_time(args.date) if args.date else None
    output_path = args.output or build_default_output_path()

    stock_df = load_stock_bars(args.symbol, period="daily")
    latest_signal = evaluate_latest_abnormal_move(
        stock_df,
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
                "abnormal_move",
                "window_start_time",
                "window_end_time",
                "hit_count",
                "max_gain_pct",
                "event_time",
                "event_prev_close",
                "event_open",
                "event_high",
                "event_low",
                "event_close",
                "event_gain_pct",
            ]
        ].to_string()
    )

    pool_signals = scan_period_for_latest_abnormal_move(
        config=config,
        limit=args.limit,
        reference_time=reference_time,
    )
    pool_signals.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nPool latest completed-bar screen:")
    if pool_signals.empty:
        print("No abnormal-move signal found.")
    else:
        print(
            pool_signals[
                [
                    "file_symbol",
                    "name",
                    "hit_count",
                    "max_gain_pct",
                    "event_time",
                    "event_gain_pct",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved latest results to: {output_path}")


if __name__ == "__main__":
    main()
