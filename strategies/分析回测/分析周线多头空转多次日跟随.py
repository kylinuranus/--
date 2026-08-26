from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, KongTurnBullConfig, detect_bullish_up, evaluate_latest_kong_turn_bull
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
WEEKLY_ROOT = DATA_ROOT / "weekly"
DAILY_ROOT = DATA_ROOT / "daily"

SIGNAL_DATES = [
    pd.Timestamp("2026-06-15"),
    pd.Timestamp("2026-06-16"),
    pd.Timestamp("2026-06-17"),
    pd.Timestamp("2026-06-18"),
]

WEEKLY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
DAILY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]

OUTPUT_DETAIL = SCREENING_RESULTS_DIR / "weekly_bullish_kong_turn_bull_detail_2026-06-15_to_2026-06-18.csv"
OUTPUT_SUMMARY = SCREENING_RESULTS_DIR / "weekly_bullish_kong_turn_bull_summary_2026-06-15_to_2026-06-18.csv"
OUTPUT_POOL = SCREENING_RESULTS_DIR / "weekly_bullish_pool_asof_2026-06-15.csv"
OUTPUT_REPORT = SCREENING_RESULTS_DIR / "weekly_bullish_kong_turn_bull_report_2026-06-15_to_2026-06-18.md"


def weekly_bullish_config() -> BullishUpConfig:
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


def kong_turn_bull_config() -> KongTurnBullConfig:
    return KongTurnBullConfig(
        require_yesterday_lower_low=True,
        require_today_higher_high=True,
    )


def dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def iso_week_key(timestamp: pd.Timestamp) -> tuple[int, int]:
    iso = timestamp.isocalendar()
    return int(iso.year), int(iso.week)


def is_a_share_week_closed(reference_time: pd.Timestamp) -> bool:
    current_time = pd.Timestamp(reference_time)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("Asia/Shanghai")
    else:
        current_time = current_time.tz_convert("Asia/Shanghai")

    if current_time.weekday() < 4:
        return False
    if current_time.weekday() > 4:
        return True

    market_close_time = current_time.normalize() + pd.Timedelta(hours=15)
    return current_time >= market_close_time


def prepare_weekly_screen_frame(df: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df

    truncated = df.loc[: str(signal_date.date())].copy()
    if truncated.empty:
        return truncated

    weekly_df = dedupe_to_last_bar_per_week(truncated)
    if len(weekly_df) < 2:
        return weekly_df

    reference_time = pd.Timestamp(signal_date.date()).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    latest_bar_time = pd.Timestamp(weekly_df.index[-1])

    if iso_week_key(latest_bar_time) != iso_week_key(reference_time):
        return weekly_df
    if is_a_share_week_closed(reference_time):
        return weekly_df

    return weekly_df.iloc[:-1].copy()


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def compute_return(current: float | None, base: float | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base) or abs(float(base)) <= 1e-9:
        return None
    return float(current) / float(base) - 1.0


def safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def safe_median(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def build_weekly_pool_asof(signal_date: pd.Timestamp, config: BullishUpConfig) -> pd.DataFrame:
    matches: list[dict[str, object]] = []

    for weekly_path in sorted(WEEKLY_ROOT.glob("*.parquet")):
        raw_weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        if raw_weekly_df.empty:
            continue

        weekly_df = prepare_weekly_screen_frame(raw_weekly_df, signal_date=signal_date)
        if len(weekly_df) < 2:
            continue

        weekly_signal = detect_bullish_up(weekly_df, config=config).fillna(False)
        if not bool(weekly_signal.iloc[-1]):
            continue

        latest_row = weekly_df.iloc[-1]
        matches.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "file_symbol": weekly_path.stem,
                "symbol": latest_row.get("symbol", weekly_path.stem),
                "name": latest_row.get("name"),
                "weekly_time": pd.Timestamp(weekly_df.index[-1]).date().isoformat(),
                "weekly_open": float(latest_row["open"]),
                "weekly_high": float(latest_row["high"]),
                "weekly_low": float(latest_row["low"]),
                "weekly_close": float(latest_row["close"]),
            }
        )

    pool_df = pd.DataFrame(matches)
    if not pool_df.empty:
        pool_df = pool_df.sort_values(["file_symbol"]).reset_index(drop=True)
    return pool_df


def analyze_signal_day(
    signal_date: pd.Timestamp,
    weekly_pool_df: pd.DataFrame,
    daily_cache: dict[str, pd.DataFrame],
    config: KongTurnBullConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    daily_eligible_count = 0

    for weekly_row in weekly_pool_df.itertuples(index=False):
        file_symbol = str(weekly_row.file_symbol)
        if file_symbol not in daily_cache:
            daily_path = DAILY_ROOT / f"{file_symbol}.parquet"
            if not daily_path.exists():
                continue
            daily_cache[file_symbol] = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()

        daily_df = daily_cache[file_symbol]
        signal_slice = daily_df.loc[: str(signal_date.date())].copy()
        if signal_slice.empty:
            continue

        latest_bar_time = pd.Timestamp(signal_slice.index[-1]).normalize()
        if latest_bar_time != signal_date.normalize():
            continue

        daily_eligible_count += 1
        latest_signal = evaluate_latest_kong_turn_bull(signal_slice, config=config)
        if not bool(latest_signal["kong_turn_bull"]):
            continue

        next_day_df = daily_df.loc[daily_df.index > signal_date]
        if next_day_df.empty:
            continue

        signal_bar = signal_slice.iloc[-1]
        next_bar = next_day_df.iloc[0]
        next_trade_date = pd.Timestamp(next_day_df.index[0]).normalize()
        next2_day_df = daily_df.loc[daily_df.index > next_trade_date]

        signal_close = float(signal_bar["close"])
        signal_open = float(signal_bar["open"])
        next_open = float(next_bar["open"])
        next_high = float(next_bar["high"])
        next_low = float(next_bar["low"])
        next_close = float(next_bar["close"])
        prev1_low = float(latest_signal["prev1_low"])

        next_close_ret = compute_return(next_close, signal_close)
        next_intraday_ret = compute_return(next_close, next_open)
        next_day_holds_prev1_new_low = bool(next_low >= prev1_low)
        next_day_low_vs_prev1_low_ret = compute_return(next_low, prev1_low)

        if next2_day_df.empty:
            next2_trade_date = pd.NA
            next2_open = pd.NA
            next2_high = pd.NA
            next2_low = pd.NA
            next2_close = pd.NA
            next2_close_ret_from_next_close = pd.NA
            next2_close_ret_from_signal_close = pd.NA
            next2_up_close_to_close = pd.NA
            next2_day_green = pd.NA
        else:
            next2_bar = next2_day_df.iloc[0]
            next2_trade_date = pd.Timestamp(next2_day_df.index[0]).normalize().date().isoformat()
            next2_open = float(next2_bar["open"])
            next2_high = float(next2_bar["high"])
            next2_low = float(next2_bar["low"])
            next2_close = float(next2_bar["close"])
            next2_close_ret_from_next_close = compute_return(next2_close, next_close)
            next2_close_ret_from_signal_close = compute_return(next2_close, signal_close)
            next2_up_close_to_close = bool(
                next2_close_ret_from_next_close is not None
                and not pd.isna(next2_close_ret_from_next_close)
                and next2_close_ret_from_next_close > 0
            )
            next2_day_green = bool(next2_close > next2_open)

        detail_rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "next_trade_date": next_trade_date.date().isoformat(),
                "file_symbol": file_symbol,
                "symbol": latest_signal.get("symbol", weekly_row.symbol),
                "name": latest_signal.get("name", weekly_row.name),
                "weekly_time": weekly_row.weekly_time,
                "weekly_open": weekly_row.weekly_open,
                "weekly_high": weekly_row.weekly_high,
                "weekly_low": weekly_row.weekly_low,
                "weekly_close": weekly_row.weekly_close,
                "signal_open": signal_open,
                "signal_high": float(signal_bar["high"]),
                "signal_low": float(signal_bar["low"]),
                "signal_close": signal_close,
                "kong_turn_bull_pattern": latest_signal.get("kong_turn_bull_pattern"),
                "prev2_time": pd.Timestamp(latest_signal["prev2_time"]).date().isoformat(),
                "prev1_time": pd.Timestamp(latest_signal["prev1_time"]).date().isoformat(),
                "current_time": pd.Timestamp(latest_signal["current_time"]).date().isoformat(),
                "prev2_low": float(latest_signal["prev2_low"]),
                "prev1_low": prev1_low,
                "prev1_high": float(latest_signal["prev1_high"]),
                "current_open": float(latest_signal["current_open"]),
                "current_close": float(latest_signal["current_close"]),
                "current_high": float(latest_signal["current_high"]),
                "current_low": float(latest_signal["current_low"]),
                "yesterday_lower_low": bool(latest_signal["yesterday_lower_low"]),
                "today_higher_high": bool(latest_signal["today_higher_high"]),
                "today_lower_low": bool(latest_signal["today_lower_low"]),
                "today_close_breaks_yesterday_high": bool(latest_signal["today_close_breaks_yesterday_high"]),
                "today_bullish_close": bool(latest_signal["today_bullish_close"]),
                "three_day_kong_turn_bull": bool(latest_signal["three_day_kong_turn_bull"]),
                "two_day_kong_turn_bull": bool(latest_signal["two_day_kong_turn_bull"]),
                "next_open": next_open,
                "next_high": next_high,
                "next_low": next_low,
                "next_close": next_close,
                "next_close_ret": next_close_ret,
                "next_intraday_ret": next_intraday_ret,
                "next_up_close_to_close": bool(next_close_ret > 0) if next_close_ret is not None and not pd.isna(next_close_ret) else False,
                "next_day_green": bool(next_close > next_open),
                "next_day_holds_prev1_new_low": next_day_holds_prev1_new_low,
                "next_day_low_vs_prev1_low_ret": next_day_low_vs_prev1_low_ret,
                "next2_trade_date": next2_trade_date,
                "next2_open": next2_open,
                "next2_high": next2_high,
                "next2_low": next2_low,
                "next2_close": next2_close,
                "next2_close_ret_from_next_close": next2_close_ret_from_next_close,
                "next2_close_ret_from_signal_close": next2_close_ret_from_signal_close,
                "next2_up_close_to_close": next2_up_close_to_close,
                "next2_day_green": next2_day_green,
            }
        )

    signal_detail_df = pd.DataFrame(detail_rows)
    next_trade_dates = sorted(signal_detail_df["next_trade_date"].dropna().unique().tolist()) if not signal_detail_df.empty else []
    positive_next = signal_detail_df.loc[pd.to_numeric(signal_detail_df.get("next_close_ret"), errors="coerce") > 0, "next_close_ret"] if not signal_detail_df.empty else pd.Series(dtype=float)
    non_positive_next = signal_detail_df.loc[pd.to_numeric(signal_detail_df.get("next_close_ret"), errors="coerce") <= 0, "next_close_ret"] if not signal_detail_df.empty else pd.Series(dtype=float)
    holds_low_df = signal_detail_df[signal_detail_df["next_day_holds_prev1_new_low"].fillna(False)].copy() if not signal_detail_df.empty else pd.DataFrame()
    summary_row = {
        "signal_date": signal_date.date().isoformat(),
        "weekly_pool_count": int(len(weekly_pool_df)),
        "daily_eligible_count": int(daily_eligible_count),
        "trigger_count": int(len(signal_detail_df)),
        "next_trade_dates": ",".join(next_trade_dates),
        "next_up_count_close_to_close": int(signal_detail_df["next_up_close_to_close"].sum()) if not signal_detail_df.empty else 0,
        "win_rate_close_to_close": (
            float(signal_detail_df["next_up_close_to_close"].mean()) if not signal_detail_df.empty else pd.NA
        ),
        "next_green_count": int(signal_detail_df["next_day_green"].sum()) if not signal_detail_df.empty else 0,
        "green_rate": float(signal_detail_df["next_day_green"].mean()) if not signal_detail_df.empty else pd.NA,
        "avg_next_close_ret": safe_mean(signal_detail_df["next_close_ret"]) if not signal_detail_df.empty else pd.NA,
        "median_next_close_ret": safe_median(signal_detail_df["next_close_ret"]) if not signal_detail_df.empty else pd.NA,
        "avg_next_gain_ret": safe_mean(positive_next) if not positive_next.empty else pd.NA,
        "avg_next_loss_ret": safe_mean(non_positive_next) if not non_positive_next.empty else pd.NA,
        "hold_prev1_low_count": int(len(holds_low_df)),
        "hold_prev1_low_rate": float(len(holds_low_df) / len(signal_detail_df)) if not signal_detail_df.empty else pd.NA,
        "hold_prev1_low_next2_sample_count": int(holds_low_df["next2_up_close_to_close"].notna().sum()) if not holds_low_df.empty else 0,
        "hold_prev1_low_next2_up_count": int(holds_low_df["next2_up_close_to_close"].fillna(False).sum()) if not holds_low_df.empty else 0,
        "hold_prev1_low_next2_win_rate": (
            float(holds_low_df["next2_up_close_to_close"].dropna().astype(bool).mean())
            if not holds_low_df.empty and not holds_low_df["next2_up_close_to_close"].dropna().empty
            else pd.NA
        ),
        "hold_prev1_low_next2_avg_ret_from_next_close": (
            safe_mean(holds_low_df["next2_close_ret_from_next_close"]) if not holds_low_df.empty else pd.NA
        ),
        "hold_prev1_low_next2_median_ret_from_next_close": (
            safe_median(holds_low_df["next2_close_ret_from_next_close"]) if not holds_low_df.empty else pd.NA
        ),
        "hold_prev1_low_next2_avg_ret_from_signal_close": (
            safe_mean(holds_low_df["next2_close_ret_from_signal_close"]) if not holds_low_df.empty else pd.NA
        ),
        "hold_prev1_low_next2_median_ret_from_signal_close": (
            safe_median(holds_low_df["next2_close_ret_from_signal_close"]) if not holds_low_df.empty else pd.NA
        ),
    }
    return detail_rows, summary_row


def build_report(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    weekly_pool_by_day: dict[str, pd.DataFrame],
) -> str:
    lines: list[str] = []
    lines.append("# 周线多头 + 空转多 调研报告")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 时间范围：2026-06-15、2026-06-16、2026-06-17、2026-06-18 这 4 个交易日。")
    lines.append("- 初步股票池：先按 `strategies/选股/多头.py` 的周线多头定义筛出母池。")
    lines.append("- 二次筛选：在母池里继续用当前策略 `空转多`，条件是“昨日最低价低于前日最低价，今日最高价高于昨日最高价”。")
    lines.append("- 次日上涨口径：主口径按“下一交易日收盘价 > 信号日收盘价”统计。")
    lines.append("- 补充口径：同时统计“下一交易日是否收阳”，也就是 `next_close > next_open`。")
    lines.append("- “次日不破前日新低”口径：这里的“前日新低”取信号结构里昨日那根创新低K线的最低价，也就是 `prev1_low`；若次日 `low >= prev1_low`，记为“不破”。")
    lines.append("- 后一天胜率口径：在“不破前日新低”的样本里，再看后一个交易日相对前一个交易日收盘是否上涨，同时也统计相对加入日收盘的涨跌幅。")
    lines.append("")

    pool_sets = {signal_date: set(df["file_symbol"]) for signal_date, df in weekly_pool_by_day.items()}
    unique_pool_count = len(next(iter(pool_sets.values()))) if pool_sets else 0
    same_pool = len({frozenset(pool) for pool in pool_sets.values()}) == 1 if pool_sets else False

    total_signals = int(len(detail_df))
    total_wins = int(detail_df["next_up_close_to_close"].sum()) if not detail_df.empty else 0
    total_green = int(detail_df["next_day_green"].sum()) if not detail_df.empty else 0
    total_win_rate = float(detail_df["next_up_close_to_close"].mean()) if not detail_df.empty else pd.NA
    total_green_rate = float(detail_df["next_day_green"].mean()) if not detail_df.empty else pd.NA
    unique_symbols = int(detail_df["file_symbol"].nunique()) if not detail_df.empty else 0
    total_positive_next = detail_df.loc[pd.to_numeric(detail_df.get("next_close_ret"), errors="coerce") > 0, "next_close_ret"] if not detail_df.empty else pd.Series(dtype=float)
    total_non_positive_next = detail_df.loc[pd.to_numeric(detail_df.get("next_close_ret"), errors="coerce") <= 0, "next_close_ret"] if not detail_df.empty else pd.Series(dtype=float)
    total_holds_low_df = detail_df[detail_df["next_day_holds_prev1_new_low"].fillna(False)].copy() if not detail_df.empty else pd.DataFrame()

    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 周线多头母池只数：{unique_pool_count}")
    lines.append(f"- 4 个信号日的周线母池是否一致：{'是' if same_pool else '否'}")
    lines.append(f"- 4 天累计触发信号数：{total_signals}")
    lines.append(f"- 触发过信号的去重股票数：{unique_symbols}")
    lines.append(f"- 次日上涨只数：{total_wins}")
    lines.append(f"- 次日上涨胜率：{format_pct(total_win_rate)}")
    lines.append(f"- 次日收阳只数：{total_green}")
    lines.append(f"- 次日收阳胜率：{format_pct(total_green_rate)}")
    if not detail_df.empty:
        lines.append(f"- 次日收盘相对信号日收盘的平均涨跌幅：{format_pct(safe_mean(detail_df['next_close_ret']))}")
        lines.append(f"- 次日收盘相对信号日收盘的中位数涨跌幅：{format_pct(safe_median(detail_df['next_close_ret']))}")
        lines.append(f"- 次日平均上涨幅度：{format_pct(safe_mean(total_positive_next))}")
        lines.append(f"- 次日平均下跌幅度：{format_pct(safe_mean(total_non_positive_next))}")
    if not total_holds_low_df.empty:
        next2_win_series = total_holds_low_df["next2_up_close_to_close"].dropna().astype(bool)
        lines.append(f"- 次日不破前日新低的样本数：{len(total_holds_low_df)}")
        lines.append(f"- 这些样本里，后一天上涨胜率：{format_pct(float(next2_win_series.mean())) if not next2_win_series.empty else '-'}")
        lines.append(f"- 这些样本里，后一天相对加入日的平均涨跌幅：{format_pct(safe_mean(total_holds_low_df['next2_close_ret_from_signal_close']))}")
        lines.append(f"- 这些样本里，后一天相对加入日的中位数涨跌幅：{format_pct(safe_median(total_holds_low_df['next2_close_ret_from_signal_close']))}")
    lines.append("")

    lines.append("## 逐日结果")
    lines.append("")
    for row in summary_df.itertuples(index=False):
        lines.append(f"### {row.signal_date}")
        lines.append("")
        lines.append(f"- 周线多头母池：{row.weekly_pool_count} 只")
        lines.append(f"- 当天在母池里有日线可评估的股票：{row.daily_eligible_count} 只")
        lines.append(f"- 当天空转多触发：{row.trigger_count} 只")
        lines.append(f"- 对应下一交易日：{row.next_trade_dates}")
        lines.append(f"- 次日上涨只数：{row.next_up_count_close_to_close} 只")
        lines.append(f"- 次日上涨胜率：{format_pct(row.win_rate_close_to_close)}")
        lines.append(f"- 次日收阳只数：{row.next_green_count} 只")
        lines.append(f"- 次日收阳胜率：{format_pct(row.green_rate)}")
        lines.append(f"- 次日平均涨跌幅：{format_pct(row.avg_next_close_ret)}")
        lines.append(f"- 次日中位数涨跌幅：{format_pct(row.median_next_close_ret)}")
        lines.append(f"- 次日平均上涨幅度：{format_pct(row.avg_next_gain_ret)}")
        lines.append(f"- 次日平均下跌幅度：{format_pct(row.avg_next_loss_ret)}")
        lines.append(f"- 次日不破前日新低样本：{row.hold_prev1_low_count} 只")
        lines.append(f"- 这些样本里，后一天上涨胜率：{format_pct(row.hold_prev1_low_next2_win_rate)}")
        lines.append(f"- 这些样本里，后一天相对次日收盘的平均涨跌幅：{format_pct(row.hold_prev1_low_next2_avg_ret_from_next_close)}")
        lines.append(f"- 这些样本里，后一天相对加入日收盘的平均涨跌幅：{format_pct(row.hold_prev1_low_next2_avg_ret_from_signal_close)}")
        lines.append("")

    if not detail_df.empty:
        best_days = summary_df.sort_values(["win_rate_close_to_close", "avg_next_close_ret"], ascending=[False, False]).head(1)
        worst_days = summary_df.sort_values(["win_rate_close_to_close", "avg_next_close_ret"], ascending=[True, True]).head(1)
        lines.append("## 观察")
        lines.append("")
        if same_pool:
            lines.append("- 这 4 个交易日的周线母池没有变化，说明在 2026-06-15 到 2026-06-18 之间，周线多头判断一直沿用截至 2026-06-12 的最后一个完整周线。")
        if not best_days.empty:
            best_row = best_days.iloc[0]
            lines.append(
                f"- 胜率最高的信号日是 {best_row['signal_date']}，当日触发 {int(best_row['trigger_count'])} 只，次日上涨胜率 {format_pct(float(best_row['win_rate_close_to_close']))}。"
            )
        if not worst_days.empty:
            worst_row = worst_days.iloc[0]
            lines.append(
                f"- 胜率最低的信号日是 {worst_row['signal_date']}，当日触发 {int(worst_row['trigger_count'])} 只，次日上涨胜率 {format_pct(float(worst_row['win_rate_close_to_close']))}。"
            )

        repeat_df = (
            detail_df.groupby(["file_symbol", "name"], as_index=False)
            .agg(signal_count=("signal_date", "count"), avg_next_close_ret=("next_close_ret", "mean"))
            .sort_values(["signal_count", "avg_next_close_ret"], ascending=[False, False])
        )
        repeat_df = repeat_df[repeat_df["signal_count"] > 1].head(10)
        if not repeat_df.empty:
            lines.append("- 有些股票会在连续几个交易日反复触发，说明这个形态本身有一定“连触发”特征，不能简单把多次触发完全当成独立样本。")
        if not total_holds_low_df.empty:
            lines.append("- 从补充统计看，“次日不破前日新低”更像一个二次确认条件，适合和后一天表现一起看，而不是只盯着次日本身。")
        lines.append("")

        top_winners = detail_df.sort_values(["next_close_ret"], ascending=False).head(10)
        top_losers = detail_df.sort_values(["next_close_ret"], ascending=True).head(10)

        lines.append("## 强弱样本")
        lines.append("")
        lines.append("### 次日表现最强")
        lines.append("")
        for row in top_winners.itertuples(index=False):
            lines.append(
                f"- {row.signal_date} | {row.file_symbol} {row.name} | 次日 {row.next_trade_date} 涨跌幅 {format_pct(float(row.next_close_ret))}"
            )
        lines.append("")

        lines.append("### 次日表现最弱")
        lines.append("")
        for row in top_losers.itertuples(index=False):
            lines.append(
                f"- {row.signal_date} | {row.file_symbol} {row.name} | 次日 {row.next_trade_date} 涨跌幅 {format_pct(float(row.next_close_ret))}"
            )
        lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 明细：`{OUTPUT_DETAIL.name}`")
    lines.append(f"- 汇总：`{OUTPUT_SUMMARY.name}`")
    lines.append(f"- 母池：`{OUTPUT_POOL.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_project_dirs()

    weekly_config = weekly_bullish_config()
    kong_config = kong_turn_bull_config()
    daily_cache: dict[str, pd.DataFrame] = {}
    weekly_pool_by_day: dict[str, pd.DataFrame] = {}
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for signal_date in SIGNAL_DATES:
        weekly_pool_df = build_weekly_pool_asof(signal_date=signal_date, config=weekly_config)
        weekly_pool_by_day[signal_date.date().isoformat()] = weekly_pool_df

        day_detail_rows, day_summary_row = analyze_signal_day(
            signal_date=signal_date,
            weekly_pool_df=weekly_pool_df,
            daily_cache=daily_cache,
            config=kong_config,
        )
        detail_rows.extend(day_detail_rows)
        summary_rows.append(day_summary_row)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["signal_date", "next_close_ret", "file_symbol"], ascending=[True, False, True]).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["signal_date"]).reset_index(drop=True)

    first_day_pool_df = weekly_pool_by_day[SIGNAL_DATES[0].date().isoformat()].copy()
    if not first_day_pool_df.empty:
        first_day_pool_df.to_csv(OUTPUT_POOL, index=False, encoding="utf-8-sig")

    detail_df.to_csv(OUTPUT_DETAIL, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    OUTPUT_REPORT.write_text(build_report(summary_df, detail_df, weekly_pool_by_day), encoding="utf-8")

    print(f"detail_rows={len(detail_df)}")
    print(f"summary_rows={len(summary_df)}")
    print(f"detail_path={OUTPUT_DETAIL}")
    print(f"summary_path={OUTPUT_SUMMARY}")
    print(f"pool_path={OUTPUT_POOL}")
    print(f"report_path={OUTPUT_REPORT}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
