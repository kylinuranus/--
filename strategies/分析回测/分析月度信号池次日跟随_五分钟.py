from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs
from strategies.分析回测.分析信号池次日跟随_五分钟 import (
    analyze_pool,
    format_pct,
    selected_5m_green_mask,
)
from strategies.选股.默认选股 import run_default_selection


DATA_ROOT = Path(r"D:\a_stock_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the default daily stock selection across one month and build a next-trading-day "
            "follow-through report, including a 5-minute green-bar subset."
        )
    )
    parser.add_argument(
        "--month",
        type=str,
        required=True,
        help="Month in YYYY-MM format, for example 2026-06.",
    )
    parser.add_argument(
        "--m5-bar",
        choices=["first", "last"],
        default="first",
        help="Which 5-minute bar to inspect on the follow-up date. Default uses the first bar.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Local parquet root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCREENING_RESULTS_DIR,
        help="Directory for CSV/report outputs.",
    )
    return parser.parse_args()


def parse_month(month_text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start = pd.Timestamp(f"{month_text}-01").normalize()
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
    return month_start, month_end


def build_reference_time(signal_date: pd.Timestamp) -> pd.Timestamp:
    return signal_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)


def bar_mode_label(bar_mode: str) -> str:
    return "首根5分钟K线" if bar_mode == "first" else "最后一根5分钟K线"


def bar_mode_suffix(bar_mode: str) -> str:
    return "first5m" if bar_mode == "first" else "last5m"


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


def summarize_detail_subset(detail_df: pd.DataFrame) -> dict[str, object]:
    if detail_df.empty:
        return {
            "sample_count": 0,
            "win_count": 0,
            "win_rate": pd.NA,
            "avg_ret_vs_signal_close": pd.NA,
            "median_ret_vs_signal_close": pd.NA,
            "avg_intraday_ret": pd.NA,
            "median_intraday_ret": pd.NA,
        }

    return {
        "sample_count": int(len(detail_df)),
        "win_count": int(detail_df["follow_ret_vs_signal_close"].gt(0).sum()),
        "win_rate": float(detail_df["follow_ret_vs_signal_close"].gt(0).mean()),
        "avg_ret_vs_signal_close": safe_mean(detail_df["follow_ret_vs_signal_close"]),
        "median_ret_vs_signal_close": safe_median(detail_df["follow_ret_vs_signal_close"]),
        "avg_intraday_ret": safe_mean(detail_df["follow_intraday_ret"]),
        "median_intraday_ret": safe_median(detail_df["follow_intraday_ret"]),
    }


def safe_rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    numeric = pd.to_numeric(mask, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.astype(bool).mean())


def discover_trading_dates(
    data_root: Path,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    daily_root = data_root / "daily"
    if not daily_root.exists():
        raise FileNotFoundError(f"Daily folder not found: {daily_root}")

    dates: set[pd.Timestamp] = set()
    for parquet_path in sorted(daily_root.glob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["close"])
        if df.empty:
            continue
        for trade_date in pd.to_datetime(df.index).normalize().unique():
            timestamp = pd.Timestamp(trade_date)
            if range_start <= timestamp <= range_end:
                dates.add(timestamp)

    return sorted(dates)


def build_signal_follow_pairs(
    trading_dates: list[pd.Timestamp],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[pd.Timestamp]]:
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    covered_signal_dates: set[pd.Timestamp] = set()

    for index in range(len(trading_dates) - 1):
        signal_date = trading_dates[index]
        follow_date = trading_dates[index + 1]
        if month_start <= signal_date <= month_end:
            pairs.append((signal_date, follow_date))
            covered_signal_dates.add(signal_date)

    uncovered = [date for date in trading_dates if month_start <= date <= month_end and date not in covered_signal_dates]
    return pairs, uncovered


def build_report(
    month_text: str,
    signal_daily_df: pd.DataFrame,
    monthly_detail_df: pd.DataFrame,
    monthly_summary: dict[str, object],
    excluded_signal_dates: list[pd.Timestamp],
    report_paths: dict[str, Path],
) -> str:
    green_mask = selected_5m_green_mask(monthly_detail_df)
    green_df = monthly_detail_df.loc[green_mask].copy() if not monthly_detail_df.empty else pd.DataFrame()
    non_green_df = monthly_detail_df.loc[~green_mask].copy() if not monthly_detail_df.empty else pd.DataFrame()
    top_winners = monthly_detail_df.sort_values("follow_ret_vs_signal_close", ascending=False).head(15) if not monthly_detail_df.empty else pd.DataFrame()
    top_losers = monthly_detail_df.sort_values("follow_ret_vs_signal_close", ascending=True).head(15) if not monthly_detail_df.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append(f"# {month_text} 选股与次日表现月报")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 选股方法：先取最近两根已完成周K定义的周线多头母池，再只在母池里筛选当日空转多。")
    lines.append("- 次日表现口径：下一交易日收盘价 / 信号日收盘价 - 1。")
    lines.append(f"- 5分钟收阳口径：下一交易日{monthly_summary['selected_5m_bar_label']}收阳，即该 5 分钟K线 `close > open`。")
    lines.append("- 时间解释：全部按本地日线数据中的交易日顺序计算“次日”，不是按自然日。")
    lines.append("")

    lines.append("## 覆盖范围")
    lines.append("")
    lines.append(f"- 请求月份：{month_text}")
    lines.append(f"- 本地日线最新日期：{monthly_summary['latest_local_daily_date']}")
    lines.append(f"- 月内交易日数量：{monthly_summary['month_trading_day_count']}")
    lines.append(f"- 可做次日分析的信号日数量：{monthly_summary['analyzable_signal_day_count']}")
    lines.append(f"- 实际有选股结果的信号日数量：{monthly_summary['signal_days_with_picks']}")
    if excluded_signal_dates:
        lines.append(
            "- 因缺少下一交易日数据而未纳入次日分析的信号日："
            + "、".join(date.date().isoformat() for date in excluded_signal_dates)
        )
    lines.append("")

    lines.append("## 月度汇总")
    lines.append("")
    lines.append(f"- 月内累计选股样本数：{monthly_summary['total_signal_count']}")
    lines.append(f"- 成功完成次日分析的样本数：{monthly_summary['total_analyzed_count']}")
    lines.append(f"- 其中 {monthly_summary['selected_5m_bar_label']}收阳样本数：{monthly_summary['green_sample_count']}")
    lines.append(f"- 整体平均涨跌幅：{format_pct(monthly_summary['all_avg_ret_vs_signal_close'])}")
    lines.append(f"- 整体中位数涨跌幅：{format_pct(monthly_summary['all_median_ret_vs_signal_close'])}")
    lines.append(f"- 整体上涨胜率：{format_pct(monthly_summary['all_win_rate'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}收阳子集平均涨跌幅：{format_pct(monthly_summary['green_avg_ret_vs_signal_close'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}收阳子集中位数涨跌幅：{format_pct(monthly_summary['green_median_ret_vs_signal_close'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}收阳子集上涨胜率：{format_pct(monthly_summary['green_win_rate'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}非收阳子集平均涨跌幅：{format_pct(monthly_summary['non_green_avg_ret_vs_signal_close'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}非收阳子集中位数涨跌幅：{format_pct(monthly_summary['non_green_median_ret_vs_signal_close'])}")
    lines.append(f"- {monthly_summary['selected_5m_bar_label']}非收阳子集上涨胜率：{format_pct(monthly_summary['non_green_win_rate'])}")
    lines.append("")

    if not signal_daily_df.empty:
        analyzable_daily = signal_daily_df.loc[signal_daily_df["analyzed_count"] > 0].copy()
        if not analyzable_daily.empty:
            best_day = analyzable_daily.sort_values("avg_ret_vs_signal_close", ascending=False).iloc[0]
            worst_day = analyzable_daily.sort_values("avg_ret_vs_signal_close", ascending=True).iloc[0]
            lines.append("## 最强与最弱信号日")
            lines.append("")
            lines.append(
                f"- 最强信号日：{best_day['signal_date']} -> {best_day['follow_date']} | "
                f"样本 {int(best_day['signal_count'])} | 平均涨跌幅 {format_pct(best_day['avg_ret_vs_signal_close'])} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳 {int(best_day['green_count'])} 只，平均 {format_pct(best_day['green_avg_ret_vs_signal_close'])}"
            )
            lines.append(
                f"- 最弱信号日：{worst_day['signal_date']} -> {worst_day['follow_date']} | "
                f"样本 {int(worst_day['signal_count'])} | 平均涨跌幅 {format_pct(worst_day['avg_ret_vs_signal_close'])} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳 {int(worst_day['green_count'])} 只，平均 {format_pct(worst_day['green_avg_ret_vs_signal_close'])}"
            )
            lines.append("")

        lines.append("## 每日拆解")
        lines.append("")
        for row in signal_daily_df.itertuples(index=False):
            lines.append(
                f"- {row.signal_date} -> {row.follow_date} | 选股 {int(row.signal_count)} | "
                f"次日分析样本 {int(row.analyzed_count)} | 平均涨跌幅 {format_pct(row.avg_ret_vs_signal_close)} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳 {int(row.green_count)} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳占比 {format_pct(row.green_rate)} | "
                f"当日收阳概率 {format_pct(row.follow_green_rate)} | "
                f"当日收阴概率 {format_pct(row.follow_red_rate)} | "
                f"收阳子集平均 {format_pct(row.green_avg_ret_vs_signal_close)}"
            )
        lines.append("")

    if not top_winners.empty:
        lines.append("## 月内涨幅前十五")
        lines.append("")
        for row in top_winners.itertuples(index=False):
            green_text = "是" if bool(getattr(row, "selected_5m_green", False)) else "否"
            lines.append(
                f"- {row.signal_date} -> {row.follow_date} | {row.file_symbol} {row.name} | "
                f"次日涨跌幅 {format_pct(row.follow_ret_vs_signal_close)} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳 {green_text}"
            )
        lines.append("")

    if not top_losers.empty:
        lines.append("## 月内跌幅前十五")
        lines.append("")
        for row in top_losers.itertuples(index=False):
            green_text = "是" if bool(getattr(row, "selected_5m_green", False)) else "否"
            lines.append(
                f"- {row.signal_date} -> {row.follow_date} | {row.file_symbol} {row.name} | "
                f"次日涨跌幅 {format_pct(row.follow_ret_vs_signal_close)} | "
                f"{monthly_summary['selected_5m_bar_label']}收阳 {green_text}"
            )
        lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- 逐日汇总：`{report_paths['daily_summary'].name}`")
    lines.append(f"- 月度明细：`{report_paths['detail'].name}`")
    lines.append(f"- 月度汇总：`{report_paths['summary'].name}`")
    lines.append(f"- 月报：`{report_paths['report'].name}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    month_start, month_end = parse_month(args.month)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trading_dates = discover_trading_dates(
        data_root=Path(args.data_root),
        range_start=month_start,
        range_end=month_end + pd.Timedelta(days=10),
    )
    if not trading_dates:
        raise ValueError(f"No local trading dates found for {args.month}.")

    month_trading_dates = [date for date in trading_dates if month_start <= date <= month_end]
    signal_follow_pairs, excluded_signal_dates = build_signal_follow_pairs(
        trading_dates=trading_dates,
        month_start=month_start,
        month_end=month_end,
    )

    monthly_detail_frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, object]] = []
    monthly_signal_dir = output_dir / f"weekly_bullish_kong_turn_bull_signal_pool_{args.month}"
    monthly_signal_dir.mkdir(parents=True, exist_ok=True)

    for signal_date, follow_date in signal_follow_pairs:
        print(f"processing {signal_date.date().isoformat()} -> {follow_date.date().isoformat()}")
        output_path = monthly_signal_dir / f"weekly_bullish_kong_turn_bull_{signal_date.date().isoformat()}.csv"
        signal_df, selection_summary = run_default_selection(
            reference_time=build_reference_time(signal_date),
            output_path=output_path,
        )
        detail_df, follow_summary = analyze_pool(
            input_df=signal_df,
            signal_date=signal_date,
            follow_date=follow_date,
            bar_mode=args.m5_bar,
            data_root=Path(args.data_root),
        )
        green_mask = selected_5m_green_mask(detail_df)
        follow_green_rate = safe_rate(detail_df["follow_green"]) if not detail_df.empty and "follow_green" in detail_df.columns else None
        follow_red_rate = safe_rate(~detail_df["follow_green"].astype(bool)) if not detail_df.empty and "follow_green" in detail_df.columns else None
        if not detail_df.empty:
            monthly_detail_frames.append(detail_df)

        daily_rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "follow_date": follow_date.date().isoformat(),
                "weekly_pool_count": selection_summary["weekly_pool_count"],
                "signal_count": int(len(signal_df)),
                "analyzed_count": int(follow_summary["analyzed_count"]),
                "green_count": int(follow_summary["green_sample_count"]),
                "green_rate": float(green_mask.mean()) if len(green_mask) else pd.NA,
                "follow_green_rate": follow_green_rate if follow_green_rate is not None else pd.NA,
                "follow_red_rate": follow_red_rate if follow_red_rate is not None else pd.NA,
                "avg_ret_vs_signal_close": follow_summary["all_avg_ret_vs_signal_close"],
                "median_ret_vs_signal_close": follow_summary["all_median_ret_vs_signal_close"],
                "win_rate": follow_summary["all_win_rate"],
                "green_avg_ret_vs_signal_close": follow_summary["green_avg_ret_vs_signal_close"],
                "green_median_ret_vs_signal_close": follow_summary["green_median_ret_vs_signal_close"],
                "green_win_rate": follow_summary["green_win_rate"],
                "non_green_avg_ret_vs_signal_close": follow_summary["non_green_avg_ret_vs_signal_close"],
                "non_green_win_rate": follow_summary["non_green_win_rate"],
            }
        )

    signal_daily_df = pd.DataFrame(daily_rows)
    if not signal_daily_df.empty:
        signal_daily_df = signal_daily_df.sort_values("signal_date").reset_index(drop=True)

    if monthly_detail_frames:
        monthly_detail_df = pd.concat(monthly_detail_frames, ignore_index=True)
        monthly_detail_df = monthly_detail_df.sort_values(["signal_date", "follow_ret_vs_signal_close"], ascending=[True, False]).reset_index(drop=True)
    else:
        monthly_detail_df = pd.DataFrame()

    green_mask = selected_5m_green_mask(monthly_detail_df)
    green_df = monthly_detail_df.loc[green_mask].copy() if not monthly_detail_df.empty else pd.DataFrame()
    non_green_df = monthly_detail_df.loc[~green_mask].copy() if not monthly_detail_df.empty else pd.DataFrame()

    monthly_summary = {
        "month": args.month,
        "selected_5m_bar_mode": args.m5_bar,
        "selected_5m_bar_label": bar_mode_label(args.m5_bar),
        "latest_local_daily_date": trading_dates[-1].date().isoformat(),
        "month_trading_day_count": len(month_trading_dates),
        "analyzable_signal_day_count": len(signal_follow_pairs),
        "signal_days_with_picks": int(signal_daily_df["signal_count"].gt(0).sum()) if not signal_daily_df.empty else 0,
        "total_signal_count": int(signal_daily_df["signal_count"].sum()) if not signal_daily_df.empty else 0,
        "total_analyzed_count": int(len(monthly_detail_df)),
        **{f"all_{key}": value for key, value in summarize_detail_subset(monthly_detail_df).items()},
        **{f"green_{key}": value for key, value in summarize_detail_subset(green_df).items()},
        **{f"non_green_{key}": value for key, value in summarize_detail_subset(non_green_df).items()},
    }

    suffix = f"{args.month}_{bar_mode_suffix(args.m5_bar)}"
    report_paths = {
        "daily_summary": output_dir / f"weekly_bullish_kong_turn_bull_monthly_followthrough_{suffix}_daily_summary.csv",
        "detail": output_dir / f"weekly_bullish_kong_turn_bull_monthly_followthrough_{suffix}_detail.csv",
        "summary": output_dir / f"weekly_bullish_kong_turn_bull_monthly_followthrough_{suffix}_summary.csv",
        "report": output_dir / f"weekly_bullish_kong_turn_bull_monthly_followthrough_{suffix}_report.md",
    }

    signal_daily_df.to_csv(report_paths["daily_summary"], index=False, encoding="utf-8-sig")
    monthly_detail_df.to_csv(report_paths["detail"], index=False, encoding="utf-8-sig")
    pd.DataFrame([monthly_summary]).to_csv(report_paths["summary"], index=False, encoding="utf-8-sig")
    report_paths["report"].write_text(
        build_report(
            month_text=args.month,
            signal_daily_df=signal_daily_df,
            monthly_detail_df=monthly_detail_df,
            monthly_summary=monthly_summary,
            excluded_signal_dates=excluded_signal_dates,
            report_paths=report_paths,
        ),
        encoding="utf-8",
    )

    print(f"month={args.month}")
    print(f"month_trading_day_count={monthly_summary['month_trading_day_count']}")
    print(f"analyzable_signal_day_count={monthly_summary['analyzable_signal_day_count']}")
    print(f"signal_days_with_picks={monthly_summary['signal_days_with_picks']}")
    print(f"total_signal_count={monthly_summary['total_signal_count']}")
    print(f"total_analyzed_count={monthly_summary['total_analyzed_count']}")
    print(f"report_path={report_paths['report']}")
    print(pd.DataFrame([monthly_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
