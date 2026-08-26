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

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
DAILY_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]
M5_COLUMNS = ["symbol", "market", "name", "period", "adjustment", "open", "high", "low", "close"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one signal-pool CSV on the next trading day, including overall follow-through "
            "and a 5-minute green-bar subset report."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Signal-pool CSV path. Defaults to screening_results/weekly_bullish_kong_turn_bull_<signal_date>.csv.",
    )
    parser.add_argument(
        "--signal-date",
        type=str,
        required=True,
        help="Signal date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--follow-date",
        type=str,
        required=True,
        help="Follow-up date in YYYY-MM-DD.",
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
        help="Directory for detail/summary/report outputs.",
    )
    return parser.parse_args()


def normalize_date(date_text: str) -> pd.Timestamp:
    return pd.Timestamp(date_text).normalize()


def resolve_input_path(input_path: Path | None, signal_date: pd.Timestamp) -> Path:
    if input_path is not None:
        resolved = input_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input CSV not found: {resolved}")
        return resolved

    default_path = SCREENING_RESULTS_DIR / f"weekly_bullish_kong_turn_bull_{signal_date.date().isoformat()}.csv"
    if not default_path.exists():
        raise FileNotFoundError(f"Default signal CSV not found: {default_path}")
    return default_path


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


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def compute_return(current: float | int | None, base: float | int | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base):
        return None
    base_float = float(base)
    if abs(base_float) <= 1e-9:
        return None
    return float(current) / base_float - 1.0


def pick_name(row: pd.Series) -> str:
    for column in ["name", "name_daily", "name_weekly"]:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return str(row.get("file_symbol", ""))


def pick_symbol(row: pd.Series) -> str:
    for column in ["symbol", "symbol_daily", "symbol_weekly"]:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    file_symbol = str(row.get("file_symbol", "")).strip().lower()
    match = re.search(r"(\d{6})", file_symbol)
    return match.group(1) if match else file_symbol


def pick_signal_close(row: pd.Series) -> float:
    for column in ["close", "signal_close", "current_close"]:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    raise ValueError("Signal close column not found in input row.")


def locate_m5_bar_for_day(df: pd.DataFrame, trade_date: pd.Timestamp, bar_mode: str) -> pd.Series | None:
    same_day = df.loc[(df.index >= trade_date) & (df.index < trade_date + pd.Timedelta(days=1))]
    if same_day.empty:
        return None
    if bar_mode == "last":
        return same_day.iloc[-1]
    return same_day.iloc[0]


def bar_mode_label(bar_mode: str) -> str:
    return "首根5分钟K线" if bar_mode == "first" else "最后一根5分钟K线"


def bar_mode_suffix(bar_mode: str) -> str:
    return "first5m" if bar_mode == "first" else "last5m"


def selected_5m_green_mask(detail_df: pd.DataFrame) -> pd.Series:
    if detail_df.empty or "selected_5m_green" not in detail_df.columns:
        return pd.Series(False, index=detail_df.index, dtype=bool)
    return pd.to_numeric(detail_df["selected_5m_green"], errors="coerce").fillna(0).astype(bool)


def summarize_subset(detail_df: pd.DataFrame) -> dict[str, object]:
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


def analyze_pool(
    input_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    follow_date: pd.Timestamp,
    bar_mode: str,
    data_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing_daily = 0
    missing_follow_day = 0
    missing_m5 = 0
    missing_m5_bar = 0

    for _, row in input_df.iterrows():
        file_symbol = str(row["file_symbol"]).strip()
        signal_close = pick_signal_close(row)
        daily_path = data_root / "daily" / f"{file_symbol}.parquet"
        m5_path = data_root / "5m" / f"{file_symbol}.parquet"

        if not daily_path.exists():
            missing_daily += 1
            continue

        daily_df = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
        same_day_mask = pd.to_datetime(daily_df.index).normalize() == follow_date
        if not same_day_mask.any():
            missing_follow_day += 1
            continue

        follow_bar = daily_df.loc[same_day_mask].iloc[-1]
        detail_row: dict[str, object] = {
            "signal_date": signal_date.date().isoformat(),
            "follow_date": follow_date.date().isoformat(),
            "file_symbol": file_symbol,
            "symbol": pick_symbol(row),
            "name": pick_name(row),
            "signal_close": signal_close,
            "follow_open": float(follow_bar["open"]),
            "follow_high": float(follow_bar["high"]),
            "follow_low": float(follow_bar["low"]),
            "follow_close": float(follow_bar["close"]),
            "follow_ret_vs_signal_close": compute_return(float(follow_bar["close"]), signal_close),
            "follow_gap_vs_signal_close": compute_return(float(follow_bar["open"]), signal_close),
            "follow_intraday_ret": compute_return(float(follow_bar["close"]), float(follow_bar["open"])),
            "follow_green": bool(float(follow_bar["close"]) > float(follow_bar["open"])),
        }

        if m5_path.exists():
            m5_df = pd.read_parquet(m5_path, columns=M5_COLUMNS).sort_index()
            selected_bar = locate_m5_bar_for_day(m5_df, follow_date, bar_mode=bar_mode)
            if selected_bar is None:
                missing_m5_bar += 1
            else:
                same_day_m5 = m5_df.loc[(m5_df.index >= follow_date) & (m5_df.index < follow_date + pd.Timedelta(days=1))]
                selected_time = same_day_m5.index[0] if bar_mode == "first" else same_day_m5.index[-1]
                detail_row["selected_5m_time"] = pd.Timestamp(selected_time).isoformat(sep=" ")
                detail_row["selected_5m_open"] = float(selected_bar["open"])
                detail_row["selected_5m_high"] = float(selected_bar["high"])
                detail_row["selected_5m_low"] = float(selected_bar["low"])
                detail_row["selected_5m_close"] = float(selected_bar["close"])
                detail_row["selected_5m_green"] = bool(float(selected_bar["close"]) > float(selected_bar["open"]))
                detail_row["selected_5m_ret_vs_signal_close"] = compute_return(float(selected_bar["close"]), signal_close)
        else:
            missing_m5 += 1

        rows.append(detail_row)

    detail_df = pd.DataFrame(rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["follow_ret_vs_signal_close", "file_symbol"], ascending=[False, True]).reset_index(drop=True)

    green_mask = selected_5m_green_mask(detail_df)
    green_subset = detail_df.loc[green_mask].copy() if not detail_df.empty else pd.DataFrame()
    non_green_subset = detail_df.loc[~green_mask].copy() if not detail_df.empty else pd.DataFrame()

    summary = {
        "signal_date": signal_date.date().isoformat(),
        "follow_date": follow_date.date().isoformat(),
        "input_count": int(len(input_df)),
        "analyzed_count": int(len(detail_df)),
        "missing_daily_count": missing_daily,
        "missing_follow_day_count": missing_follow_day,
        "missing_m5_file_count": missing_m5,
        "missing_selected_5m_bar_count": missing_m5_bar,
        "selected_5m_bar_mode": bar_mode,
        "selected_5m_bar_label": bar_mode_label(bar_mode),
        **{f"all_{key}": value for key, value in summarize_subset(detail_df).items()},
        **{f"green_{key}": value for key, value in summarize_subset(green_subset).items()},
        **{f"non_green_{key}": value for key, value in summarize_subset(non_green_subset).items()},
    }
    return detail_df, summary


def build_report(
    input_path: Path,
    detail_df: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    green_mask = selected_5m_green_mask(detail_df)
    green_subset = detail_df.loc[green_mask].copy() if not detail_df.empty else pd.DataFrame()
    top_winners = detail_df.head(10) if not detail_df.empty else pd.DataFrame()
    top_losers = detail_df.tail(10).sort_values("follow_ret_vs_signal_close", ascending=True) if not detail_df.empty else pd.DataFrame()
    green_winners = green_subset.sort_values("follow_ret_vs_signal_close", ascending=False).head(10) if not green_subset.empty else pd.DataFrame()

    lines: list[str] = []
    lines.append("# 周K多头 + 空转多 次日表现分析报告")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append(f"- 样本来源文件：`{input_path.name}`")
    lines.append(f"- 信号日：{summary['signal_date']} Asia/Shanghai")
    lines.append(f"- 分析日：{summary['follow_date']} Asia/Shanghai")
    lines.append("- 样本池定义：使用信号日的“周K多头 + 空转多”结果文件。")
    lines.append(f"- 主涨跌幅口径：分析日收盘价 / 信号日收盘价 - 1。")
    lines.append(f"- 5分钟收阳口径：分析日{summary['selected_5m_bar_label']}收阳，也就是该 5 分钟K线 `close > open`。")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 输入样本数：{summary['input_count']}")
    lines.append(f"- 成功分析样本数：{summary['analyzed_count']}")
    lines.append(f"- 缺少日线文件数：{summary['missing_daily_count']}")
    lines.append(f"- 缺少分析日日线数：{summary['missing_follow_day_count']}")
    lines.append(f"- 缺少 5 分钟文件数：{summary['missing_m5_file_count']}")
    lines.append(f"- 缺少选定 5 分钟K线数：{summary['missing_selected_5m_bar_count']}")
    lines.append("")

    lines.append("## 整体表现")
    lines.append("")
    lines.append(f"- 平均涨跌幅：{format_pct(summary['all_avg_ret_vs_signal_close'])}")
    lines.append(f"- 中位数涨跌幅：{format_pct(summary['all_median_ret_vs_signal_close'])}")
    lines.append(f"- 上涨只数：{summary['all_win_count']}/{summary['all_sample_count']}")
    lines.append(f"- 上涨胜率：{format_pct(summary['all_win_rate'])}")
    lines.append(f"- 分析日日内平均涨跌幅：{format_pct(summary['all_avg_intraday_ret'])}")
    lines.append(f"- 分析日日内中位数涨跌幅：{format_pct(summary['all_median_intraday_ret'])}")
    lines.append("")

    lines.append(f"## {summary['selected_5m_bar_label']}收阳子集")
    lines.append("")
    lines.append(f"- 子集样本数：{summary['green_sample_count']}")
    lines.append(f"- 子集占全部样本比例：{format_pct(summary['green_sample_count'] / summary['all_sample_count'] if summary['all_sample_count'] else None)}")
    lines.append(f"- 平均涨跌幅：{format_pct(summary['green_avg_ret_vs_signal_close'])}")
    lines.append(f"- 中位数涨跌幅：{format_pct(summary['green_median_ret_vs_signal_close'])}")
    lines.append(f"- 上涨只数：{summary['green_win_count']}/{summary['green_sample_count']}")
    lines.append(f"- 上涨胜率：{format_pct(summary['green_win_rate'])}")
    lines.append(f"- 分析日日内平均涨跌幅：{format_pct(summary['green_avg_intraday_ret'])}")
    lines.append(f"- 分析日日内中位数涨跌幅：{format_pct(summary['green_median_intraday_ret'])}")
    lines.append("")

    lines.append(f"## {summary['selected_5m_bar_label']}非收阳子集")
    lines.append("")
    lines.append(f"- 子集样本数：{summary['non_green_sample_count']}")
    lines.append(f"- 平均涨跌幅：{format_pct(summary['non_green_avg_ret_vs_signal_close'])}")
    lines.append(f"- 中位数涨跌幅：{format_pct(summary['non_green_median_ret_vs_signal_close'])}")
    lines.append(f"- 上涨只数：{summary['non_green_win_count']}/{summary['non_green_sample_count']}")
    lines.append(f"- 上涨胜率：{format_pct(summary['non_green_win_rate'])}")
    lines.append(f"- 分析日日内平均涨跌幅：{format_pct(summary['non_green_avg_intraday_ret'])}")
    lines.append(f"- 分析日日内中位数涨跌幅：{format_pct(summary['non_green_median_intraday_ret'])}")
    lines.append("")

    if not top_winners.empty:
        lines.append("## 整体涨幅前十")
        lines.append("")
        for row in top_winners.itertuples(index=False):
            green_text = "是" if bool(getattr(row, "selected_5m_green", False)) else "否"
            lines.append(
                f"- {row.file_symbol} {row.name} | 次日涨跌幅 {format_pct(row.follow_ret_vs_signal_close)} | 日内涨跌幅 {format_pct(row.follow_intraday_ret)} | {summary['selected_5m_bar_label']}收阳 {green_text}"
            )
        lines.append("")

    if not top_losers.empty:
        lines.append("## 整体跌幅前十")
        lines.append("")
        for row in top_losers.itertuples(index=False):
            green_text = "是" if bool(getattr(row, "selected_5m_green", False)) else "否"
            lines.append(
                f"- {row.file_symbol} {row.name} | 次日涨跌幅 {format_pct(row.follow_ret_vs_signal_close)} | 日内涨跌幅 {format_pct(row.follow_intraday_ret)} | {summary['selected_5m_bar_label']}收阳 {green_text}"
            )
        lines.append("")

    if not green_winners.empty:
        lines.append(f"## {summary['selected_5m_bar_label']}收阳样本中涨幅前十")
        lines.append("")
        for row in green_winners.itertuples(index=False):
            lines.append(
                f"- {row.file_symbol} {row.name} | 次日涨跌幅 {format_pct(row.follow_ret_vs_signal_close)} | 日内涨跌幅 {format_pct(row.follow_intraday_ret)} | 5分钟涨跌幅 {format_pct(row.selected_5m_ret_vs_signal_close)}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    signal_date = normalize_date(args.signal_date)
    follow_date = normalize_date(args.follow_date)
    input_path = resolve_input_path(args.input, signal_date=signal_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)
    detail_df, summary = analyze_pool(
        input_df=input_df,
        signal_date=signal_date,
        follow_date=follow_date,
        bar_mode=args.m5_bar,
        data_root=Path(args.data_root),
    )

    suffix = f"{signal_date.date().isoformat()}_{follow_date.date().isoformat()}_{bar_mode_suffix(args.m5_bar)}"
    detail_path = output_dir / f"weekly_bullish_kong_turn_bull_followthrough_{suffix}_detail.csv"
    summary_path = output_dir / f"weekly_bullish_kong_turn_bull_followthrough_{suffix}_summary.csv"
    report_path = output_dir / f"weekly_bullish_kong_turn_bull_followthrough_{suffix}_report.md"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            input_path=input_path,
            detail_df=detail_df,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print(f"input_count={summary['input_count']}")
    print(f"analyzed_count={summary['analyzed_count']}")
    print(f"green_count={summary['green_sample_count']}")
    print(f"detail_path={detail_path}")
    print(f"summary_path={summary_path}")
    print(f"report_path={report_path}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
