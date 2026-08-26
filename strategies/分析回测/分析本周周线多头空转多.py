from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, KongTurnBullConfig, evaluate_latest_kong_turn_bull
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.测试策略.筛选周线多头_三十分钟道氏回调 import DATA_ROOT, screen_latest_weekly_bullish


DAILY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build this week's weekly-bullish pool, screen yesterday/today kong-turn-bull signals, and analyze yesterday picks versus today's move."
    )
    parser.add_argument("--today", default="2026-06-23", help="Today in YYYY-MM-DD.")
    parser.add_argument("--yesterday", default="2026-06-22", help="Previous trading date in YYYY-MM-DD.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--output-dir", type=Path, default=SCREENING_RESULTS_DIR, help="Directory for CSV/report outputs.")
    return parser.parse_args()


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
        enable_three_day_pattern=True,
        enable_two_day_close_breakout=True,
    )


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


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


def compute_return(current: float | None, base: float | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base) or abs(float(base)) <= 1e-9:
        return None
    return float(current) / float(base) - 1.0


def load_daily_bars(file_symbol: str, data_root: Path, cache: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if file_symbol in cache:
        return cache[file_symbol]

    daily_path = data_root / "daily" / f"{file_symbol}.parquet"
    if not daily_path.exists():
        return None

    cache[file_symbol] = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
    return cache[file_symbol]


def screen_kong_turn_bull_in_pool(
    signal_date: pd.Timestamp,
    weekly_pool_df: pd.DataFrame,
    data_root: Path,
    config: KongTurnBullConfig,
    daily_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    matches: list[dict[str, object]] = []

    for row in weekly_pool_df.itertuples(index=False):
        file_symbol = str(row.file_symbol)
        daily_df = load_daily_bars(file_symbol=file_symbol, data_root=data_root, cache=daily_cache)
        if daily_df is None:
            continue

        signal_slice = daily_df.loc[: str(signal_date.date())].copy()
        if signal_slice.empty:
            continue

        latest_bar_time = pd.Timestamp(signal_slice.index[-1]).normalize()
        if latest_bar_time != signal_date.normalize():
            continue

        latest_signal = evaluate_latest_kong_turn_bull(signal_slice, config=config)
        if not bool(latest_signal["kong_turn_bull"]):
            continue

        signal_bar = signal_slice.iloc[-1]
        matches.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "file_symbol": file_symbol,
                "symbol": latest_signal.get("symbol", row.symbol),
                "name": latest_signal.get("name", row.name),
                "weekly_time": row.weekly_time,
                "weekly_open": row.weekly_open,
                "weekly_high": row.weekly_high,
                "weekly_low": row.weekly_low,
                "weekly_close": row.weekly_close,
                "signal_open": float(signal_bar["open"]),
                "signal_high": float(signal_bar["high"]),
                "signal_low": float(signal_bar["low"]),
                "signal_close": float(signal_bar["close"]),
                "kong_turn_bull": bool(latest_signal["kong_turn_bull"]),
                "kong_turn_bull_pattern": latest_signal.get("kong_turn_bull_pattern"),
                "prev2_time": pd.Timestamp(latest_signal["prev2_time"]).date().isoformat() if pd.notna(latest_signal["prev2_time"]) else pd.NA,
                "prev1_time": pd.Timestamp(latest_signal["prev1_time"]).date().isoformat() if pd.notna(latest_signal["prev1_time"]) else pd.NA,
                "current_time": pd.Timestamp(latest_signal["current_time"]).date().isoformat() if pd.notna(latest_signal["current_time"]) else pd.NA,
                "prev2_high": float(latest_signal["prev2_high"]) if pd.notna(latest_signal["prev2_high"]) else pd.NA,
                "prev2_low": float(latest_signal["prev2_low"]) if pd.notna(latest_signal["prev2_low"]) else pd.NA,
                "prev1_high": float(latest_signal["prev1_high"]) if pd.notna(latest_signal["prev1_high"]) else pd.NA,
                "prev1_low": float(latest_signal["prev1_low"]) if pd.notna(latest_signal["prev1_low"]) else pd.NA,
                "current_open": float(latest_signal["current_open"]) if pd.notna(latest_signal["current_open"]) else pd.NA,
                "current_close": float(latest_signal["current_close"]) if pd.notna(latest_signal["current_close"]) else pd.NA,
                "current_high": float(latest_signal["current_high"]) if pd.notna(latest_signal["current_high"]) else pd.NA,
                "current_low": float(latest_signal["current_low"]) if pd.notna(latest_signal["current_low"]) else pd.NA,
                "yesterday_lower_low": bool(latest_signal["yesterday_lower_low"]),
                "today_higher_high": bool(latest_signal["today_higher_high"]),
                "today_lower_low": bool(latest_signal["today_lower_low"]),
                "today_close_breaks_yesterday_high": bool(latest_signal["today_close_breaks_yesterday_high"]),
                "today_bullish_close": bool(latest_signal["today_bullish_close"]),
                "three_day_kong_turn_bull": bool(latest_signal["three_day_kong_turn_bull"]),
                "two_day_kong_turn_bull": bool(latest_signal["two_day_kong_turn_bull"]),
            }
        )

    result_df = pd.DataFrame(matches)
    if not result_df.empty:
        result_df = result_df.sort_values(["file_symbol"]).reset_index(drop=True)
    return result_df


def analyze_yesterday_signals_vs_today(
    yesterday_df: pd.DataFrame,
    today_df: pd.DataFrame,
    today_date: pd.Timestamp,
    data_root: Path,
    daily_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    today_lookup = today_df.set_index("file_symbol").to_dict("index") if not today_df.empty else {}
    analysis_rows: list[dict[str, object]] = []

    for row in yesterday_df.itertuples(index=False):
        file_symbol = str(row.file_symbol)
        daily_df = load_daily_bars(file_symbol=file_symbol, data_root=data_root, cache=daily_cache)
        if daily_df is None or today_date not in pd.to_datetime(daily_df.index).normalize():
            continue

        today_bar = daily_df.loc[str(today_date.date())].iloc[-1] if isinstance(daily_df.loc[str(today_date.date())], pd.DataFrame) else daily_df.loc[str(today_date.date())]
        today_open = float(today_bar["open"])
        today_high = float(today_bar["high"])
        today_low = float(today_bar["low"])
        today_close = float(today_bar["close"])
        today_ret_vs_signal_close = compute_return(today_close, float(row.signal_close))
        today_intraday_ret = compute_return(today_close, today_open)
        today_is_signal = file_symbol in today_lookup
        today_pattern = today_lookup[file_symbol]["kong_turn_bull_pattern"] if today_is_signal else pd.NA

        analysis_rows.append(
            {
                "signal_date": row.signal_date,
                "today_date": today_date.date().isoformat(),
                "file_symbol": file_symbol,
                "symbol": row.symbol,
                "name": row.name,
                "yesterday_pattern": row.kong_turn_bull_pattern,
                "yesterday_signal_close": float(row.signal_close),
                "today_open": today_open,
                "today_high": today_high,
                "today_low": today_low,
                "today_close": today_close,
                "today_ret_vs_yesterday_signal_close": today_ret_vs_signal_close,
                "today_intraday_ret": today_intraday_ret,
                "today_green": bool(today_close > today_open),
                "today_is_kong_turn_bull_again": today_is_signal,
                "today_pattern": today_pattern,
            }
        )

    analysis_df = pd.DataFrame(analysis_rows)
    positive = analysis_df.loc[pd.to_numeric(analysis_df.get("today_ret_vs_yesterday_signal_close"), errors="coerce") > 0, "today_ret_vs_yesterday_signal_close"] if not analysis_df.empty else pd.Series(dtype=float)
    non_positive = analysis_df.loc[pd.to_numeric(analysis_df.get("today_ret_vs_yesterday_signal_close"), errors="coerce") <= 0, "today_ret_vs_yesterday_signal_close"] if not analysis_df.empty else pd.Series(dtype=float)

    summary = {
        "yesterday_signal_count": int(len(yesterday_df)),
        "yesterday_with_today_bar_count": int(len(analysis_df)),
        "today_up_count": int(analysis_df["today_ret_vs_yesterday_signal_close"].gt(0).sum()) if not analysis_df.empty else 0,
        "today_up_win_rate": float(analysis_df["today_ret_vs_yesterday_signal_close"].gt(0).mean()) if not analysis_df.empty else pd.NA,
        "today_green_count": int(analysis_df["today_green"].sum()) if not analysis_df.empty else 0,
        "today_green_rate": float(analysis_df["today_green"].mean()) if not analysis_df.empty else pd.NA,
        "avg_today_ret_vs_signal_close": safe_mean(analysis_df["today_ret_vs_yesterday_signal_close"]) if not analysis_df.empty else pd.NA,
        "median_today_ret_vs_signal_close": safe_median(analysis_df["today_ret_vs_yesterday_signal_close"]) if not analysis_df.empty else pd.NA,
        "avg_today_gain": safe_mean(positive) if not positive.empty else pd.NA,
        "avg_today_loss": safe_mean(non_positive) if not non_positive.empty else pd.NA,
        "today_repeat_signal_count": int(analysis_df["today_is_kong_turn_bull_again"].sum()) if not analysis_df.empty else 0,
    }
    return analysis_df, summary


def build_report(
    today_date: pd.Timestamp,
    yesterday_date: pd.Timestamp,
    weekly_pool_df: pd.DataFrame,
    yesterday_df: pd.DataFrame,
    today_df: pd.DataFrame,
    yesterday_today_df: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# 本周周多头 + 空转多 选股报告")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append(f"- 运行日期：{today_date.date().isoformat()} Asia/Shanghai。")
    lines.append(f"- 昨日信号日：{yesterday_date.date().isoformat()}。")
    lines.append(f"- 今日信号日：{today_date.date().isoformat()}。")
    lines.append("- 周多头母池：按 `strategies/选股/多头.py` 的周线多头定义生成。")
    lines.append("- 空转多：同时包含 `three_day`、`two_day`、`both` 三种触发类型。")
    lines.append("- 今日走势分析：用昨日信号股的今日收盘，相对昨日信号日收盘做对比。")
    lines.append("")

    weekly_times = sorted(weekly_pool_df["weekly_time"].dropna().astype(str).unique().tolist()) if not weekly_pool_df.empty else []
    lines.append("## 结果总览")
    lines.append("")
    lines.append(f"- 本周周多头母池只数：{len(weekly_pool_df)}")
    lines.append(f"- 当前母池使用的最新完成周K时间：{', '.join(weekly_times) if weekly_times else '-'}")
    lines.append(f"- {yesterday_date.date().isoformat()} 空转多只数：{len(yesterday_df)}")
    lines.append(f"- {today_date.date().isoformat()} 空转多只数：{len(today_df)}")
    lines.append(f"- 昨日信号股里，今天有日线可分析的只数：{summary['yesterday_with_today_bar_count']}")
    lines.append(f"- 昨日信号股今天上涨只数：{summary['today_up_count']}")
    lines.append(f"- 昨日信号股今天上涨胜率：{format_pct(summary['today_up_win_rate'])}")
    lines.append(f"- 昨日信号股今天收阳只数：{summary['today_green_count']}")
    lines.append(f"- 昨日信号股今天收阳率：{format_pct(summary['today_green_rate'])}")
    lines.append(f"- 昨日信号股今天平均涨跌幅：{format_pct(summary['avg_today_ret_vs_signal_close'])}")
    lines.append(f"- 昨日信号股今天中位数涨跌幅：{format_pct(summary['median_today_ret_vs_signal_close'])}")
    lines.append(f"- 昨日信号股今天平均上涨幅度：{format_pct(summary['avg_today_gain'])}")
    lines.append(f"- 昨日信号股今天平均下跌幅度：{format_pct(summary['avg_today_loss'])}")
    lines.append(f"- 昨日信号股今天再次触发空转多的只数：{summary['today_repeat_signal_count']}")
    lines.append("")

    if not yesterday_df.empty:
        lines.append(f"## {yesterday_date.date().isoformat()} 信号分布")
        lines.append("")
        for pattern, count in yesterday_df["kong_turn_bull_pattern"].fillna("unknown").value_counts().items():
            lines.append(f"- {pattern}：{int(count)} 只")
        lines.append("")

    if not today_df.empty:
        lines.append(f"## {today_date.date().isoformat()} 信号分布")
        lines.append("")
        for pattern, count in today_df["kong_turn_bull_pattern"].fillna("unknown").value_counts().items():
            lines.append(f"- {pattern}：{int(count)} 只")
        lines.append("")

    if not yesterday_today_df.empty:
        overlap_df = yesterday_today_df[yesterday_today_df["today_is_kong_turn_bull_again"].fillna(False)].copy()
        top_winners = yesterday_today_df.sort_values(["today_ret_vs_yesterday_signal_close"], ascending=False).head(10)
        top_losers = yesterday_today_df.sort_values(["today_ret_vs_yesterday_signal_close"], ascending=True).head(10)

        lines.append("## 昨日样本看今天")
        lines.append("")
        if not overlap_df.empty:
            lines.append(f"- 昨日信号里今天再次触发的有 {len(overlap_df)} 只。")
            lines.append("- 重复触发名单：")
            for row in overlap_df.itertuples(index=False):
                lines.append(
                    f"- {row.file_symbol} {row.name} | 昨日 {row.yesterday_pattern} | 今日 {row.today_pattern} | 今日涨跌幅 {format_pct(row.today_ret_vs_yesterday_signal_close)}"
                )
            lines.append("")

        lines.append("### 今日表现最强")
        lines.append("")
        for row in top_winners.itertuples(index=False):
            lines.append(
                f"- {row.file_symbol} {row.name} | 今日涨跌幅 {format_pct(row.today_ret_vs_yesterday_signal_close)} | 今日是否再触发 {'是' if row.today_is_kong_turn_bull_again else '否'}"
            )
        lines.append("")

        lines.append("### 今日表现最弱")
        lines.append("")
        for row in top_losers.itertuples(index=False):
            lines.append(
                f"- {row.file_symbol} {row.name} | 今日涨跌幅 {format_pct(row.today_ret_vs_yesterday_signal_close)} | 今日是否再触发 {'是' if row.today_is_kong_turn_bull_again else '否'}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    today_date = pd.Timestamp(args.today).normalize()
    yesterday_date = pd.Timestamp(args.yesterday).normalize()
    reference_time = today_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)

    weekly_pool_df, weekly_stats = screen_latest_weekly_bullish(
        data_root=args.data_root,
        weekly_config=weekly_bullish_config(),
        output_dir=None,
        reference_time=reference_time,
    )

    daily_cache: dict[str, pd.DataFrame] = {}
    kong_config = kong_turn_bull_config()
    yesterday_df = screen_kong_turn_bull_in_pool(
        signal_date=yesterday_date,
        weekly_pool_df=weekly_pool_df,
        data_root=Path(args.data_root),
        config=kong_config,
        daily_cache=daily_cache,
    )
    today_df = screen_kong_turn_bull_in_pool(
        signal_date=today_date,
        weekly_pool_df=weekly_pool_df,
        data_root=Path(args.data_root),
        config=kong_config,
        daily_cache=daily_cache,
    )
    yesterday_today_df, summary = analyze_yesterday_signals_vs_today(
        yesterday_df=yesterday_df,
        today_df=today_df,
        today_date=today_date,
        data_root=Path(args.data_root),
        daily_cache=daily_cache,
    )

    weekly_pool_path = output_dir / f"weekly_bullish_pool_asof_{today_date.date().isoformat()}.csv"
    yesterday_path = output_dir / f"weekly_bullish_kong_turn_bull_signal_{yesterday_date.date().isoformat()}.csv"
    today_path = output_dir / f"weekly_bullish_kong_turn_bull_signal_{today_date.date().isoformat()}.csv"
    analyze_detail_path = output_dir / f"weekly_bullish_kong_turn_bull_yesterday_to_today_detail_{yesterday_date.date().isoformat()}_{today_date.date().isoformat()}.csv"
    analyze_summary_path = output_dir / f"weekly_bullish_kong_turn_bull_yesterday_to_today_summary_{yesterday_date.date().isoformat()}_{today_date.date().isoformat()}.csv"
    report_path = output_dir / f"weekly_bullish_kong_turn_bull_yesterday_to_today_report_{yesterday_date.date().isoformat()}_{today_date.date().isoformat()}.md"

    weekly_pool_df.to_csv(weekly_pool_path, index=False, encoding="utf-8-sig")
    yesterday_df.to_csv(yesterday_path, index=False, encoding="utf-8-sig")
    today_df.to_csv(today_path, index=False, encoding="utf-8-sig")
    yesterday_today_df.to_csv(analyze_detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{**summary, "weekly_pool_count": len(weekly_pool_df), "weekly_latest_time": weekly_stats.get("latest_time")}]).to_csv(
        analyze_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    report_path.write_text(
        build_report(
            today_date=today_date,
            yesterday_date=yesterday_date,
            weekly_pool_df=weekly_pool_df,
            yesterday_df=yesterday_df,
            today_df=today_df,
            yesterday_today_df=yesterday_today_df,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print(f"weekly_pool_count={len(weekly_pool_df)}")
    print(f"yesterday_signal_count={len(yesterday_df)}")
    print(f"today_signal_count={len(today_df)}")
    print(f"yesterday_today_analyze_count={len(yesterday_today_df)}")
    print(f"weekly_pool_path={weekly_pool_path}")
    print(f"yesterday_path={yesterday_path}")
    print(f"today_path={today_path}")
    print(f"analyze_detail_path={analyze_detail_path}")
    print(f"analyze_summary_path={analyze_summary_path}")
    print(f"report_path={report_path}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
