# 作用：组合筛选“周线 Bullish Up + 30分钟 Dow Pullback”两类条件。
# 内容：先找出周线最新一根K线满足强势上行形态的标的，再筛选30分钟级别处于道氏回调结构中的标的，并输出交集结果与CSV文件。
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, DowPullbackConfig, detect_bullish_up, evaluate_latest_dow_pullback
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
OUTPUT_DIR = SCREENING_RESULTS_DIR
A_SHARE_MARKET_CLOSE_HOUR = 15
WEEKLY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
M30_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close", "volume"]


def _dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def _normalize_reference_time(reference_time: pd.Timestamp | None = None) -> pd.Timestamp:
    current_time = pd.Timestamp.now(tz="Asia/Shanghai") if reference_time is None else pd.Timestamp(reference_time)
    if current_time.tzinfo is None:
        return current_time.tz_localize("Asia/Shanghai")
    return current_time.tz_convert("Asia/Shanghai")


def _iso_week_key(timestamp: pd.Timestamp) -> tuple[int, int]:
    iso = timestamp.isocalendar()
    return int(iso.year), int(iso.week)


def _is_a_share_week_closed(reference_time: pd.Timestamp | None = None) -> bool:
    current_time = _normalize_reference_time(reference_time)
    if current_time.weekday() < 4:
        return False
    if current_time.weekday() > 4:
        return True

    market_close_time = current_time.normalize() + pd.Timedelta(hours=A_SHARE_MARKET_CLOSE_HOUR)
    return current_time >= market_close_time


def _prepare_weekly_screen_frame(
    df: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    weekly_df = _dedupe_to_last_bar_per_week(df)
    if len(weekly_df) < 2:
        return weekly_df

    current_time = _normalize_reference_time(reference_time)
    latest_bar_time = pd.Timestamp(weekly_df.index[-1])

    if _iso_week_key(latest_bar_time) != _iso_week_key(current_time):
        return weekly_df
    if _is_a_share_week_closed(current_time):
        return weekly_df

    # Current week is still trading, so only compare the last two completed weeks.
    return weekly_df.iloc[:-1].copy()


def screen_latest_weekly_bullish(
    data_root: str | Path = DATA_ROOT,
    weekly_config: BullishUpConfig | None = None,
    output_dir: str | Path | None = OUTPUT_DIR,
    reference_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(data_root)
    weekly_root = root / "weekly"
    if not weekly_root.exists():
        raise FileNotFoundError(f"Weekly folder not found: {weekly_root}")

    bullish_config = weekly_config or BullishUpConfig(
        breakout_field="high",
        require_prev_bull=False,
        require_curr_bull=False,
        require_body_growth=False,
        require_breakout=True,
        require_close_near_high=False,
        require_strong_body=False,
        require_open_above_prev_low=True,
        require_gap_stability=False,
        require_low_not_weaker=False,
        require_low_below_prev_low=False,
        require_volume_confirmation=False,
    )

    weekly_candidates: list[dict[str, object]] = []
    empty_files = 0
    latest_time = None

    for weekly_path in sorted(weekly_root.glob("*.parquet")):
        raw_weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        if raw_weekly_df.empty:
            empty_files += 1
            continue

        weekly_df = _prepare_weekly_screen_frame(raw_weekly_df, reference_time=reference_time)
        if weekly_df.empty:
            empty_files += 1
            continue

        latest_bar_time = weekly_df.index[-1]
        if latest_time is None or latest_bar_time > latest_time:
            latest_time = latest_bar_time
        weekly_signal = detect_bullish_up(weekly_df, config=bullish_config)
        if not bool(weekly_signal.iloc[-1]):
            continue

        weekly_row = weekly_df.iloc[-1]
        weekly_candidates.append(
            {
                "file_symbol": weekly_path.stem,
                "symbol": weekly_row.get("symbol", weekly_path.stem),
                "name": weekly_row.get("name"),
                "weekly_time": weekly_df.index[-1],
                "weekly_open": weekly_row["open"],
                "weekly_high": weekly_row["high"],
                "weekly_low": weekly_row["low"],
                "weekly_close": weekly_row["close"],
            }
        )

    weekly_candidates_df = pd.DataFrame(weekly_candidates)
    if not weekly_candidates_df.empty:
        weekly_candidates_df = weekly_candidates_df.sort_values(["file_symbol"]).reset_index(drop=True)

    stats = {
        "period": "weekly",
        "total_files": len(list(weekly_root.glob("*.parquet"))),
        "empty_files": empty_files,
        "hit_count": len(weekly_candidates_df),
        "latest_time": latest_time,
        "output_path": None,
    }

    if output_dir is not None:
        output_path = Path(output_dir) / "weekly_bullish_latest.csv"
        weekly_candidates_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        stats["output_path"] = output_path

    return weekly_candidates_df, stats


def screen_latest_30m_dow_pullback(
    data_root: str | Path = DATA_ROOT,
    m30_config: DowPullbackConfig | None = None,
    recent_bars: int = 180,
    output_dir: str | Path | None = OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(data_root)
    m30_root = root / "30m"
    if not m30_root.exists():
        raise FileNotFoundError(f"30m folder not found: {m30_root}")

    pullback_config = m30_config or DowPullbackConfig(
        swing_window=3,
        min_pullback_bars=2,
        min_pullback_retracement=0.10,
        max_pullback_retracement=0.60,
        require_price_in_lower_half=False,
    )

    matches: list[dict[str, object]] = []
    empty_files = 0
    latest_time = None

    for m30_path in sorted(m30_root.glob("*.parquet")):
        m30_df = pd.read_parquet(m30_path, columns=M30_COLUMNS).sort_index()
        if m30_df.empty:
            empty_files += 1
            continue

        latest_bar_time = m30_df.index[-1]
        if latest_time is None or latest_bar_time > latest_time:
            latest_time = latest_bar_time
        latest_snapshot = evaluate_latest_dow_pullback(m30_df, config=pullback_config, recent_bars=recent_bars)
        if not bool(latest_snapshot["dow_pullback"]):
            continue

        latest_symbol = latest_snapshot.get("symbol", m30_path.stem)
        if pd.isna(latest_symbol):
            latest_symbol = m30_path.stem

        matches.append(
            {
                "file_symbol": m30_path.stem,
                "symbol": latest_symbol,
                "name": latest_snapshot.get("name"),
                "m30_time": m30_df.index[-1],
                "m30_open": latest_snapshot["open"],
                "m30_high": latest_snapshot["high"],
                "m30_low": latest_snapshot["low"],
                "m30_close": latest_snapshot["close"],
                "low1_price": latest_snapshot["low1_price"],
                "high1_price": latest_snapshot["high1_price"],
                "low2_price": latest_snapshot["low2_price"],
                "high2_price": latest_snapshot["high2_price"],
                "price_zone_ratio": latest_snapshot["price_zone_ratio"],
                "pullback_retracement": latest_snapshot["pullback_retracement"],
                "bars_since_last_high": latest_snapshot["bars_since_last_high"],
                "screen_window_bars": latest_snapshot["screen_window_bars"],
            }
        )

    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(["file_symbol"]).reset_index(drop=True)

    stats = {
        "period": "30m",
        "total_files": len(list(m30_root.glob("*.parquet"))),
        "empty_files": empty_files,
        "hit_count": len(matches_df),
        "latest_time": latest_time,
        "output_path": None,
    }

    if output_dir is not None:
        output_path = Path(output_dir) / "m30_dow_pullback_latest.csv"
        matches_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        stats["output_path"] = output_path

    return matches_df, stats


def screen_weekly_bullish_and_30m_dow_pullback(
    data_root: str | Path = DATA_ROOT,
    weekly_config: BullishUpConfig | None = None,
    m30_config: DowPullbackConfig | None = None,
    recent_bars: int = 180,
    output_dir: str | Path | None = OUTPUT_DIR,
    reference_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    weekly_candidates_df, weekly_stats = screen_latest_weekly_bullish(
        data_root=data_root,
        weekly_config=weekly_config,
        output_dir=None,
        reference_time=reference_time,
    )
    m30_matches_df, m30_stats = screen_latest_30m_dow_pullback(
        data_root=data_root,
        m30_config=m30_config,
        recent_bars=recent_bars,
        output_dir=None,
    )

    if weekly_candidates_df.empty or m30_matches_df.empty:
        combined_df = pd.DataFrame()
    else:
        combined_df = weekly_candidates_df.merge(
            m30_matches_df,
            on=["file_symbol", "symbol", "name"],
            how="inner",
        ).sort_values(["file_symbol"]).reset_index(drop=True)

    combined_stats = {
        "period": "weekly+30m",
        "total_files": min(int(weekly_stats["total_files"]), int(m30_stats["total_files"])),
        "empty_files": None,
        "hit_count": len(combined_df),
        "latest_time": m30_stats["latest_time"],
        "output_path": None,
    }

    if output_dir is not None:
        output_root = Path(output_dir)
        weekly_path = output_root / "weekly_bullish_latest.csv"
        m30_path = output_root / "m30_dow_pullback_latest.csv"
        combined_path = output_root / "weekly_bullish_and_m30_dow_pullback_latest.csv"

        weekly_candidates_df.to_csv(weekly_path, index=False, encoding="utf-8-sig")
        m30_matches_df.to_csv(m30_path, index=False, encoding="utf-8-sig")
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")

        weekly_stats["output_path"] = str(weekly_path)
        m30_stats["output_path"] = str(m30_path)
        combined_stats["output_path"] = str(combined_path)

    summary_df = pd.DataFrame(
        [
            weekly_stats,
            m30_stats,
            combined_stats,
        ]
    )

    if output_dir is not None:
        summary_path = Path(output_dir) / "screening_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return weekly_candidates_df, m30_matches_df, combined_df, weekly_stats, m30_stats, combined_stats


def main() -> None:
    ensure_project_dirs()
    (
        weekly_candidates,
        m30_matches,
        final_matches,
        weekly_stats,
        m30_stats,
        combined_stats,
    ) = screen_weekly_bullish_and_30m_dow_pullback()

    summary_df = pd.DataFrame(
        [
            weekly_stats,
            m30_stats,
            combined_stats,
        ]
    )

    print("Screening summary:")
    print(summary_df.to_string(index=False))

    print("\nWeekly bullish candidates on the latest bar:")
    if weekly_candidates.empty:
        print("No weekly bullish candidate found.")
    else:
        print(
            weekly_candidates[
                [
                    "file_symbol",
                    "name",
                    "weekly_time",
                    "weekly_open",
                    "weekly_high",
                    "weekly_low",
                    "weekly_close",
                ]
            ].to_string(index=False)
        )

    print("\n30m Dow pullback candidates on the latest bar:")
    if m30_matches.empty:
        print("No 30m Dow pullback candidate found.")
    else:
        print(
            m30_matches[
                [
                    "file_symbol",
                    "name",
                    "m30_time",
                    "m30_open",
                    "m30_high",
                    "m30_low",
                    "m30_close",
                    "low2_price",
                    "high2_price",
                    "price_zone_ratio",
                    "pullback_retracement",
                    "bars_since_last_high",
                ]
            ].to_string(index=False)
        )

    print("\nWeekly bullish + 30m Dow pullback matches:")
    if final_matches.empty:
        print("No combined match found.")
    else:
        print(
            final_matches[
                [
                    "file_symbol",
                    "name",
                    "weekly_time",
                    "weekly_close",
                    "m30_time",
                    "m30_close",
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
