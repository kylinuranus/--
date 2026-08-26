from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import (
    BullishUpConfig,
    KongTurnBullConfig,
    filter_kong_turn_bull_signals,
    load_stock_bars,
)
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.选股.异动 import AbnormalMoveConfig, evaluate_latest_abnormal_move
from strategies.选股.月线空头 import MonthlyBearishConfig, evaluate_latest_monthly_bearish
from strategies.测试策略.筛选周线多头_三十分钟道氏回调 import (
    A_SHARE_MARKET_CLOSE_HOUR,
    DATA_ROOT,
    screen_latest_weekly_bullish,
)


WEEKLY_POOL_NAME = "weekly_bullish_latest.csv"
DAILY_SIGNAL_NAME = "kong_turn_bull_latest.csv"
DEFAULT_RESULT_PREFIX = "weekly_bullish_kong_turn_bull"
MONTHLY_BEARISH_EXCLUDED_PREFIX = "weekly_bullish_kong_turn_bull_monthly_bearish_excluded"
ABNORMAL_MOVE_INCLUDED_PREFIX = "weekly_bullish_kong_turn_bull_abnormal_move"
DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "默认选股流程：先取最近两根已完成周K满足周线多头的股票池，"
            "再只在这个母池里筛选指定日期的空转多信号，"
            "最后剔除满足月线空头的股票，并输出结果。"
        )
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="可选参考日期，格式 YYYY-MM-DD。用于决定周K是否已完成；默认使用当前 Asia/Shanghai 时间。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "可选输出文件路径。默认输出到 "
            "screening_results/weekly_bullish_kong_turn_bull_<日期>.csv"
        ),
    )
    return parser.parse_args()


def build_weekly_config() -> BullishUpConfig:
    return BullishUpConfig(
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


def build_kong_turn_bull_config() -> KongTurnBullConfig:
    return KongTurnBullConfig(
        require_yesterday_lower_low=True,
        require_today_higher_high=True,
    )


def build_monthly_bearish_config() -> MonthlyBearishConfig:
    return MonthlyBearishConfig(
        breakdown_field="low",
        require_prev_bear=False,
        require_curr_bear=False,
        require_body_growth=False,
        require_breakdown=True,
        require_close_near_low=False,
        require_strong_body=False,
        require_open_below_prev_high=True,
        require_gap_stability=False,
        require_high_not_stronger=False,
        require_high_above_prev_high=False,
        require_volume_confirmation=False,
    )


def build_abnormal_move_config() -> AbnormalMoveConfig:
    return AbnormalMoveConfig(
        lookback_bars=20,
        single_day_gain_threshold=0.08,
    )


def normalize_reference_time(date_text: str | None) -> pd.Timestamp:
    if not date_text:
        return pd.Timestamp.now(tz="Asia/Shanghai")

    normalized_text = date_text.strip()
    parsed = pd.Timestamp(normalized_text)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    else:
        parsed = parsed.tz_convert("Asia/Shanghai")

    # A plain calendar date should use the market-close timestamp for that day.
    if DATE_ONLY_PATTERN.fullmatch(normalized_text):
        parsed = parsed.normalize() + pd.Timedelta(hours=A_SHARE_MARKET_CLOSE_HOUR)

    return parsed


def build_default_output_path(reference_time: pd.Timestamp) -> Path:
    return SCREENING_RESULTS_DIR / f"{DEFAULT_RESULT_PREFIX}_{reference_time.date().isoformat()}.csv"


def build_monthly_bearish_output_path(reference_time: pd.Timestamp) -> Path:
    return SCREENING_RESULTS_DIR / f"{MONTHLY_BEARISH_EXCLUDED_PREFIX}_{reference_time.date().isoformat()}.csv"


def build_abnormal_move_output_path(reference_time: pd.Timestamp) -> Path:
    return SCREENING_RESULTS_DIR / f"{ABNORMAL_MOVE_INCLUDED_PREFIX}_{reference_time.date().isoformat()}.csv"


def normalize_signal_date(reference_time: pd.Timestamp) -> pd.Timestamp:
    return reference_time.tz_localize(None).normalize() if reference_time.tzinfo else reference_time.normalize()


def scan_kong_turn_bull_within_weekly_pool(
    weekly_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: KongTurnBullConfig,
) -> pd.DataFrame:
    if weekly_df.empty:
        return pd.DataFrame()

    matches: list[pd.DataFrame] = []
    weekly_symbols = weekly_df["file_symbol"].astype(str).tolist()

    for file_symbol in weekly_symbols:
        stock_df = load_stock_bars(file_symbol, period="daily", data_root=DATA_ROOT)
        signal_rows = filter_kong_turn_bull_signals(stock_df, config=config)
        if signal_rows.empty:
            continue

        same_day_rows = signal_rows.loc[signal_rows.index.normalize() == signal_date].copy()
        if same_day_rows.empty:
            continue

        same_day_rows["file_symbol"] = file_symbol
        if "symbol" not in same_day_rows.columns:
            same_day_rows["symbol"] = file_symbol
        matches.append(same_day_rows)

    if not matches:
        return pd.DataFrame()

    combined = pd.concat(matches).sort_index()
    time_column = combined.index.name or "datetime"
    combined = combined.sort_values(["file_symbol"]).reset_index().rename(columns={time_column: "signal_time"})
    combined["file_symbol"] = combined["file_symbol"].astype(str)
    return combined


def standardize_daily_output_columns(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df

    standardized = daily_df.copy()
    if "symbol" not in standardized.columns and "file_symbol" in standardized.columns:
        standardized["symbol"] = standardized["file_symbol"].astype(str).str[-6:]

    name_column = None
    for candidate in ["name", "name_daily", "name_weekly"]:
        if candidate in standardized.columns:
            name_column = candidate
            break
    if name_column and name_column != "name":
        standardized["name"] = standardized[name_column]

    return standardized


def scan_monthly_bearish_within_candidates(
    candidate_df: pd.DataFrame,
    reference_time: pd.Timestamp,
    config: MonthlyBearishConfig,
) -> pd.DataFrame:
    expected_columns = [
        "file_symbol",
        "monthly_symbol",
        "monthly_market",
        "monthly_name",
        "monthly_period",
        "monthly_adjustment",
        "monthly_open",
        "monthly_high",
        "monthly_low",
        "monthly_close",
        "monthly_volume_lots",
        "monthly_volume",
        "monthly_amount",
        "monthly_turnover",
        "monthly_source",
        "monthly_bearish",
        "monthly_prev_time",
        "monthly_current_time",
        "monthly_prev_open",
        "monthly_prev_high",
        "monthly_prev_low",
        "monthly_prev_close",
        "monthly_current_open",
        "monthly_current_high",
        "monthly_current_low",
        "monthly_current_close",
        "monthly_prev_bear",
        "monthly_curr_bear",
        "monthly_stronger_body",
        "monthly_breakdown",
        "monthly_close_near_low",
        "monthly_strong_body",
        "monthly_open_below_prev_high",
        "monthly_gap_stability",
        "monthly_high_not_stronger",
        "monthly_high_above_prev_high",
        "monthly_volume_confirmation",
        "monthly_current_strict_pinbar",
        "monthly_current_pinbar_direction",
    ]
    if candidate_df.empty:
        return pd.DataFrame(columns=expected_columns)

    matches: list[pd.Series] = []
    candidate_symbols = candidate_df["file_symbol"].astype(str).drop_duplicates().tolist()

    for file_symbol in candidate_symbols:
        try:
            stock_df = load_stock_bars(file_symbol, period="monthly", data_root=DATA_ROOT)
            latest_snapshot = evaluate_latest_monthly_bearish(
                stock_df,
                config=config,
                reference_time=reference_time,
            )
        except (FileNotFoundError, ValueError):
            continue

        if not bool(latest_snapshot["monthly_bearish"]):
            continue

        latest_snapshot = latest_snapshot.copy()
        if "symbol" not in latest_snapshot.index or pd.isna(latest_snapshot["symbol"]):
            latest_snapshot["symbol"] = file_symbol[-6:]
        latest_snapshot["file_symbol"] = file_symbol
        matches.append(latest_snapshot)

    if not matches:
        return pd.DataFrame(columns=expected_columns)

    combined = pd.DataFrame(matches).sort_values(["file_symbol"]).reset_index(drop=True)
    rename_map = {column: f"monthly_{column}" for column in combined.columns if column != "file_symbol"}
    combined = combined.rename(columns=rename_map)
    combined = combined.rename(columns={"monthly_monthly_bearish": "monthly_bearish"})

    for column in expected_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[expected_columns]


def scan_abnormal_move_within_candidates(
    candidate_df: pd.DataFrame,
    reference_time: pd.Timestamp,
    config: AbnormalMoveConfig,
) -> pd.DataFrame:
    expected_columns = [
        "file_symbol",
        "abnormal_move_symbol",
        "abnormal_move_market",
        "abnormal_move_name",
        "abnormal_move_period",
        "abnormal_move_adjustment",
        "abnormal_move_open",
        "abnormal_move_high",
        "abnormal_move_low",
        "abnormal_move_close",
        "abnormal_move_volume_lots",
        "abnormal_move_volume",
        "abnormal_move_amount",
        "abnormal_move_turnover",
        "abnormal_move_source",
        "abnormal_move",
        "abnormal_move_window_start_time",
        "abnormal_move_window_end_time",
        "abnormal_move_lookback_bars",
        "abnormal_move_single_day_gain_threshold",
        "abnormal_move_hit_count",
        "abnormal_move_max_gain_pct",
        "abnormal_move_event_time",
        "abnormal_move_event_prev_close",
        "abnormal_move_event_open",
        "abnormal_move_event_high",
        "abnormal_move_event_low",
        "abnormal_move_event_close",
        "abnormal_move_event_gain_pct",
    ]
    if candidate_df.empty:
        return pd.DataFrame(columns=expected_columns)

    matches: list[pd.Series] = []
    candidate_symbols = candidate_df["file_symbol"].astype(str).drop_duplicates().tolist()

    for file_symbol in candidate_symbols:
        try:
            stock_df = load_stock_bars(file_symbol, period="daily", data_root=DATA_ROOT)
            latest_snapshot = evaluate_latest_abnormal_move(
                stock_df,
                config=config,
                reference_time=reference_time,
            )
        except (FileNotFoundError, ValueError):
            continue

        if not bool(latest_snapshot["abnormal_move"]):
            continue

        latest_snapshot = latest_snapshot.copy()
        if "symbol" not in latest_snapshot.index or pd.isna(latest_snapshot["symbol"]):
            latest_snapshot["symbol"] = file_symbol[-6:]
        latest_snapshot["file_symbol"] = file_symbol
        matches.append(latest_snapshot)

    if not matches:
        return pd.DataFrame(columns=expected_columns)

    combined = pd.DataFrame(matches).sort_values(["file_symbol"]).reset_index(drop=True)
    rename_map = {column: f"abnormal_move_{column}" for column in combined.columns if column != "file_symbol"}
    combined = combined.rename(columns=rename_map)
    combined = combined.rename(columns={"abnormal_move_abnormal_move": "abnormal_move"})

    for column in expected_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[expected_columns]


def run_default_selection(reference_time: pd.Timestamp, output_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    weekly_df, weekly_stats = screen_latest_weekly_bullish(
        data_root=DATA_ROOT,
        weekly_config=build_weekly_config(),
        output_dir=SCREENING_RESULTS_DIR,
        reference_time=reference_time,
    )
    if not weekly_df.empty:
        weekly_df["file_symbol"] = weekly_df["file_symbol"].astype(str)

    signal_date = normalize_signal_date(reference_time)
    daily_df = scan_kong_turn_bull_within_weekly_pool(
        weekly_df=weekly_df,
        signal_date=signal_date,
        config=build_kong_turn_bull_config(),
    )
    daily_df = standardize_daily_output_columns(daily_df)

    daily_signal_path = SCREENING_RESULTS_DIR / DAILY_SIGNAL_NAME
    daily_df.to_csv(daily_signal_path, index=False, encoding="utf-8-sig")

    if weekly_df.empty or daily_df.empty:
        combined_df = pd.DataFrame()
    else:
        combined_df = weekly_df.merge(
            daily_df,
            on=["file_symbol"],
            how="inner",
            suffixes=("_weekly", "_daily"),
        ).sort_values(["file_symbol"]).reset_index(drop=True)

    if not combined_df.empty:
        if "symbol" not in combined_df.columns:
            if "symbol_daily" in combined_df.columns:
                combined_df["symbol"] = combined_df["symbol_daily"]
            elif "symbol_weekly" in combined_df.columns:
                combined_df["symbol"] = combined_df["symbol_weekly"]
        if "name" not in combined_df.columns:
            if "name_daily" in combined_df.columns:
                combined_df["name"] = combined_df["name_daily"]
            elif "name_weekly" in combined_df.columns:
                combined_df["name"] = combined_df["name_weekly"]

    monthly_bearish_df = scan_monthly_bearish_within_candidates(
        candidate_df=combined_df,
        reference_time=reference_time,
        config=build_monthly_bearish_config(),
    )
    monthly_bearish_output_path = build_monthly_bearish_output_path(reference_time)
    monthly_bearish_df.to_csv(monthly_bearish_output_path, index=False, encoding="utf-8-sig")

    if not combined_df.empty and not monthly_bearish_df.empty:
        excluded_symbols = set(monthly_bearish_df["file_symbol"].astype(str))
        combined_df = combined_df.loc[~combined_df["file_symbol"].isin(excluded_symbols)].copy().reset_index(drop=True)

    post_monthly_count = len(combined_df)
    abnormal_move_df = scan_abnormal_move_within_candidates(
        candidate_df=combined_df,
        reference_time=reference_time,
        config=build_abnormal_move_config(),
    )
    abnormal_move_output_path = build_abnormal_move_output_path(reference_time)
    abnormal_move_df.to_csv(abnormal_move_output_path, index=False, encoding="utf-8-sig")

    if not combined_df.empty:
        if abnormal_move_df.empty:
            combined_df = combined_df.iloc[0:0].copy()
        else:
            kept_symbols = set(abnormal_move_df["file_symbol"].astype(str))
            combined_df = combined_df.loc[combined_df["file_symbol"].isin(kept_symbols)].copy().reset_index(drop=True)

    final_output = output_path or build_default_output_path(reference_time)
    combined_df.to_csv(final_output, index=False, encoding="utf-8-sig")

    summary = {
        "reference_date": reference_time.date().isoformat(),
        "weekly_pool_count": len(weekly_df),
        "kong_turn_bull_count": len(daily_df),
        "monthly_bearish_excluded_count": len(monthly_bearish_df),
        "post_monthly_count": post_monthly_count,
        "abnormal_move_included_count": len(abnormal_move_df),
        "final_count": len(combined_df),
        "weekly_pool_path": str(SCREENING_RESULTS_DIR / WEEKLY_POOL_NAME),
        "daily_signal_path": str(daily_signal_path),
        "monthly_bearish_output_path": str(monthly_bearish_output_path),
        "abnormal_move_output_path": str(abnormal_move_output_path),
        "output_path": str(final_output),
        "weekly_latest_time": weekly_stats.get("latest_time"),
    }
    return combined_df, summary


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    reference_time = normalize_reference_time(args.date)
    result_df, summary = run_default_selection(reference_time=reference_time, output_path=args.output)

    print("默认选股流程：先周线多头母池，再在母池内筛当日空转多，剔除月线空头，最后只保留近20天内出现过单日涨幅超过8%的异动股")
    print(f"参考日期: {summary['reference_date']}")
    print(f"周线多头母池数量: {summary['weekly_pool_count']}")
    print(f"母池内当日空转多数量: {summary['kong_turn_bull_count']}")
    print(f"月线空头剔除数量: {summary['monthly_bearish_excluded_count']}")
    print(f"月线空头剔除后数量: {summary['post_monthly_count']}")
    print(f"异动保留数量: {summary['abnormal_move_included_count']}")
    print(f"最终结果数量: {summary['final_count']}")
    print(f"周线母池文件: {summary['weekly_pool_path']}")
    print(f"空转多文件: {summary['daily_signal_path']}")
    print(f"月线空头剔除文件: {summary['monthly_bearish_output_path']}")
    print(f"异动明细文件: {summary['abnormal_move_output_path']}")
    print(f"最终结果文件: {summary['output_path']}")
    print(f"周线母池使用的最新完成周K时间: {summary['weekly_latest_time']}")

    if result_df.empty:
        print("\n本次没有结果。")
        return

    preview_columns = [
        "file_symbol",
        "name_weekly",
        "kong_turn_bull_pattern",
        "weekly_time",
        "weekly_open",
        "weekly_high",
        "weekly_low",
        "weekly_close",
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
        "today_bullish_close",
    ]
    available_columns = [column for column in preview_columns if column in result_df.columns]

    print("\n结果预览:")
    print(result_df[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()
