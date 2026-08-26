from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from patterns import BullishUpConfig, KongTurnBullConfig, analyze_kong_turn_bull, detect_bullish_up
from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
WEEKLY_ROOT = DATA_ROOT / "weekly"
DAILY_ROOT = DATA_ROOT / "daily"
HOLD_DAYS = (1, 2, 3, 4, 5)
WEEKLY_COLUMNS = ["symbol", "name", "open", "high", "low", "close"]
DAILY_COLUMNS = ["symbol", "name", "open", "high", "low", "close"]
PRIMARY_GROUPS = ("three_day_all", "two_day_all")
ALL_GROUPS = ("three_day_all", "two_day_all", "both", "three_day_only", "two_day_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare forward close-to-close performance of daily two-day and three-day "
            "kong-turn-bull signals within the repo's weekly-bullish pool."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument("--signal-start", default="2026-04-01", help="First signal date to include.")
    parser.add_argument("--signal-end", default="2026-06-30", help="Last signal date to include.")
    parser.add_argument("--output-dir", type=Path, default=SCREENING_RESULTS_DIR, help="Directory for outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for quick testing.")
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
        require_three_day_bullish_close=True,
        enable_three_day_pattern=True,
        enable_two_day_close_breakout=True,
    )


def dedupe_to_last_bar_per_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    week_key = df.index.to_series().dt.strftime("%G-%V")
    return df.groupby(week_key, sort=False).tail(1).copy()


def iso_week_label(timestamp: pd.Timestamp) -> str:
    iso = pd.Timestamp(timestamp).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def compute_return(current: float | int | None, base: float | int | None) -> float | None:
    if current is None or base is None or pd.isna(current) or pd.isna(base):
        return None
    base_float = float(base)
    if abs(base_float) <= 1e-9:
        return None
    return float(current) / base_float - 1.0


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def build_weekly_pool_week_map(
    weekly_df: pd.DataFrame,
    weekly_signal: pd.Series,
) -> dict[str, dict[str, object]]:
    if weekly_df.empty:
        return {}

    dates = pd.to_datetime(weekly_df.index).normalize()
    week_map: dict[str, dict[str, object]] = {}
    for effective_idx in range(1, len(weekly_df)):
        signal_idx = effective_idx - 1
        if not bool(weekly_signal.iloc[signal_idx]):
            continue

        effective_time = pd.Timestamp(dates[effective_idx]).normalize()
        signal_time = pd.Timestamp(dates[signal_idx]).normalize()
        signal_bar = weekly_df.iloc[signal_idx]
        week_map[iso_week_label(effective_time)] = {
            "effective_week_label": iso_week_label(effective_time),
            "effective_week_time": effective_time.date().isoformat(),
            "weekly_signal_time": signal_time.date().isoformat(),
            "weekly_open": float(signal_bar["open"]),
            "weekly_high": float(signal_bar["high"]),
            "weekly_low": float(signal_bar["low"]),
            "weekly_close": float(signal_bar["close"]),
        }
    return week_map


def build_signal_detail(
    data_root: Path,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    weekly_root = data_root / "weekly"
    daily_root = data_root / "daily"

    weekly_cfg = weekly_bullish_config()
    kong_cfg = kong_turn_bull_config()

    symbol_paths = sorted(daily_root.glob("*.parquet"))
    if limit is not None:
        symbol_paths = symbol_paths[:limit]

    rows: list[dict[str, object]] = []
    processed_symbols = 0
    missing_weekly = 0
    skipped_empty_daily = 0
    skipped_empty_weekly = 0
    latest_daily_date: pd.Timestamp | None = None

    for daily_path in symbol_paths:
        weekly_path = weekly_root / daily_path.name
        if not weekly_path.exists():
            missing_weekly += 1
            continue

        daily_df = pd.read_parquet(daily_path, columns=DAILY_COLUMNS).sort_index()
        if daily_df.empty:
            skipped_empty_daily += 1
            continue

        weekly_df = pd.read_parquet(weekly_path, columns=WEEKLY_COLUMNS).sort_index()
        weekly_df = dedupe_to_last_bar_per_week(weekly_df)
        if weekly_df.empty or len(weekly_df) < 2:
            skipped_empty_weekly += 1
            continue

        daily_df = daily_df.loc[daily_df.index >= signal_start - pd.Timedelta(days=10)].copy()
        if daily_df.empty:
            skipped_empty_daily += 1
            continue

        daily_dates = pd.to_datetime(daily_df.index).normalize()
        latest_symbol_daily_date = pd.Timestamp(daily_dates[-1]).normalize()
        if latest_daily_date is None or latest_symbol_daily_date > latest_daily_date:
            latest_daily_date = latest_symbol_daily_date

        weekly_signal = detect_bullish_up(weekly_df, config=weekly_cfg).fillna(False)
        weekly_pool_map = build_weekly_pool_week_map(weekly_df=weekly_df, weekly_signal=weekly_signal)
        if not weekly_pool_map:
            processed_symbols += 1
            continue

        daily_analysis = analyze_kong_turn_bull(daily_df, config=kong_cfg)
        name_series = daily_df["name"].dropna()
        symbol_series = daily_df["symbol"].dropna()
        file_symbol = daily_path.stem
        display_name = str(name_series.iloc[-1]) if not name_series.empty else file_symbol
        symbol = str(symbol_series.iloc[-1]) if not symbol_series.empty else file_symbol[-6:]

        for idx in range(len(daily_df)):
            signal_date = pd.Timestamp(daily_dates[idx]).normalize()
            if signal_date < signal_start or signal_date > signal_end:
                continue

            weekly_info = weekly_pool_map.get(iso_week_label(signal_date))
            if weekly_info is None:
                continue

            analysis_row = daily_analysis.iloc[idx]
            if not bool(analysis_row["kong_turn_bull"]):
                continue

            signal_bar = daily_df.iloc[idx]
            signal_close = float(signal_bar["close"])
            signal_row: dict[str, object] = {
                "file_symbol": file_symbol,
                "symbol": symbol,
                "name": display_name,
                "signal_date": signal_date.date().isoformat(),
                "signal_month": signal_date.strftime("%Y-%m"),
                "signal_week_label": iso_week_label(signal_date),
                "signal_open": float(signal_bar["open"]),
                "signal_high": float(signal_bar["high"]),
                "signal_low": float(signal_bar["low"]),
                "signal_close": signal_close,
                "kong_turn_bull_pattern": analysis_row.get("kong_turn_bull_pattern"),
                "three_day_kong_turn_bull": bool(analysis_row["three_day_kong_turn_bull"]),
                "two_day_kong_turn_bull": bool(analysis_row["two_day_kong_turn_bull"]),
                **weekly_info,
            }

            for hold_day in HOLD_DAYS:
                exit_idx = idx + hold_day
                if exit_idx >= len(daily_df):
                    signal_row[f"exit_date_{hold_day}d"] = pd.NA
                    signal_row[f"ret_{hold_day}d"] = pd.NA
                    continue

                exit_date = pd.Timestamp(daily_dates[exit_idx]).normalize()
                exit_close = float(daily_df.iloc[exit_idx]["close"])
                signal_row[f"exit_date_{hold_day}d"] = exit_date.date().isoformat()
                signal_row[f"ret_{hold_day}d"] = compute_return(exit_close, signal_close)

            rows.append(signal_row)

        processed_symbols += 1
        if processed_symbols % 300 == 0:
            print(f"processed_symbols={processed_symbols}")

    detail_df = pd.DataFrame(rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["signal_date", "file_symbol"]).reset_index(drop=True)

    meta = {
        "signal_start": signal_start.date().isoformat(),
        "signal_end": signal_end.date().isoformat(),
        "latest_daily_date": latest_daily_date.date().isoformat() if latest_daily_date is not None else None,
        "processed_symbols": processed_symbols,
        "missing_weekly": missing_weekly,
        "skipped_empty_daily": skipped_empty_daily,
        "skipped_empty_weekly": skipped_empty_weekly,
    }
    return detail_df, meta


def expand_groups(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    group_masks: list[tuple[str, pd.Series]] = [
        ("three_day_all", detail_df["three_day_kong_turn_bull"].fillna(False)),
        ("two_day_all", detail_df["two_day_kong_turn_bull"].fillna(False)),
        (
            "both",
            detail_df["three_day_kong_turn_bull"].fillna(False) & detail_df["two_day_kong_turn_bull"].fillna(False),
        ),
        (
            "three_day_only",
            detail_df["three_day_kong_turn_bull"].fillna(False) & ~detail_df["two_day_kong_turn_bull"].fillna(False),
        ),
        (
            "two_day_only",
            detail_df["two_day_kong_turn_bull"].fillna(False) & ~detail_df["three_day_kong_turn_bull"].fillna(False),
        ),
    ]

    frames: list[pd.DataFrame] = []
    for group_name, mask in group_masks:
        subset = detail_df.loc[mask].copy()
        if subset.empty:
            continue
        subset["group_name"] = group_name
        frames.append(subset)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_return_series(series: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "sample_count": 0,
            "win_rate": pd.NA,
            "avg_return": pd.NA,
            "median_return": pd.NA,
            "avg_gain": pd.NA,
            "avg_loss": pd.NA,
        }

    gains = numeric[numeric > 0]
    losses = numeric[numeric <= 0]
    return {
        "sample_count": int(len(numeric)),
        "win_rate": float((numeric > 0).mean()),
        "avg_return": float(numeric.mean()),
        "median_return": float(numeric.median()),
        "avg_gain": float(gains.mean()) if not gains.empty else pd.NA,
        "avg_loss": float(losses.mean()) if not losses.empty else pd.NA,
    }


def build_counts_summary(grouped_detail_df: pd.DataFrame) -> pd.DataFrame:
    if grouped_detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_name, group_df in grouped_detail_df.groupby("group_name", sort=False):
        rows.append(
            {
                "scope": "overall",
                "signal_month": pd.NA,
                "group_name": group_name,
                "signal_count": int(len(group_df)),
                "unique_symbols": int(group_df["file_symbol"].nunique()),
                "unique_signal_dates": int(group_df["signal_date"].nunique()),
            }
        )
        for signal_month, month_df in group_df.groupby("signal_month", sort=True):
            rows.append(
                {
                    "scope": "monthly",
                    "signal_month": signal_month,
                    "group_name": group_name,
                    "signal_count": int(len(month_df)),
                    "unique_symbols": int(month_df["file_symbol"].nunique()),
                    "unique_signal_dates": int(month_df["signal_date"].nunique()),
                }
            )

    return pd.DataFrame(rows)


def build_return_summary(grouped_detail_df: pd.DataFrame) -> pd.DataFrame:
    if grouped_detail_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_name, group_df in grouped_detail_df.groupby("group_name", sort=False):
        for hold_day in HOLD_DAYS:
            rows.append(
                {
                    "scope": "overall",
                    "signal_month": pd.NA,
                    "group_name": group_name,
                    "holding_days": hold_day,
                    **summarize_return_series(group_df[f"ret_{hold_day}d"]),
                }
            )
        for signal_month, month_df in group_df.groupby("signal_month", sort=True):
            for hold_day in HOLD_DAYS:
                rows.append(
                    {
                        "scope": "monthly",
                        "signal_month": signal_month,
                        "group_name": group_name,
                        "holding_days": hold_day,
                        **summarize_return_series(month_df[f"ret_{hold_day}d"]),
                    }
                )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(["scope", "signal_month", "group_name", "holding_days"], na_position="first").reset_index(drop=True)


def make_markdown_table(df: pd.DataFrame, columns: list[str], rename_map: dict[str, str] | None = None) -> str:
    if df.empty:
        return "无数据。"

    rename_map = rename_map or {}
    work_df = df.loc[:, columns].copy()
    work_df = work_df.rename(columns=rename_map)
    headers = list(work_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in work_df.iterrows():
        cells = [str(row[col]) if not pd.isna(row[col]) else "-" for col in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_return_table(summary_df: pd.DataFrame, group_names: tuple[str, ...]) -> str:
    if summary_df.empty:
        return "无数据。"

    work_df = summary_df[summary_df["group_name"].isin(group_names)].copy()
    if work_df.empty:
        return "无数据。"

    work_df["样本数"] = work_df["sample_count"].astype("Int64").astype(str)
    work_df["胜率"] = work_df["win_rate"].apply(format_pct)
    work_df["平均涨跌幅"] = work_df["avg_return"].apply(format_pct)
    work_df["中位数涨跌幅"] = work_df["median_return"].apply(format_pct)
    work_df["平均盈利"] = work_df["avg_gain"].apply(format_pct)
    work_df["平均亏损"] = work_df["avg_loss"].apply(format_pct)
    return make_markdown_table(
        work_df,
        columns=["group_name", "holding_days", "样本数", "胜率", "平均涨跌幅", "中位数涨跌幅", "平均盈利", "平均亏损"],
        rename_map={"group_name": "分组", "holding_days": "持有天数"},
    )


def render_counts_table(counts_df: pd.DataFrame) -> str:
    if counts_df.empty:
        return "无数据。"

    work_df = counts_df.copy()
    for column in ["signal_count", "unique_symbols", "unique_signal_dates"]:
        work_df[column] = work_df[column].astype("Int64").astype(str)
    return make_markdown_table(
        work_df,
        columns=["group_name", "signal_count", "unique_symbols", "unique_signal_dates"],
        rename_map={
            "group_name": "分组",
            "signal_count": "信号数",
            "unique_symbols": "去重股票数",
            "unique_signal_dates": "有信号交易日数",
        },
    )


def render_pairwise_comparison(summary_df: pd.DataFrame, signal_month: str | None = None) -> str:
    if summary_df.empty:
        return "无数据。"

    work_df = summary_df.copy()
    if signal_month is None:
        work_df = work_df[work_df["scope"] == "overall"]
    else:
        work_df = work_df[(work_df["scope"] == "monthly") & (work_df["signal_month"] == signal_month)]

    work_df = work_df[work_df["group_name"].isin(PRIMARY_GROUPS)].copy()
    if work_df.empty:
        return "无数据。"

    pivot_df = work_df.pivot(index="holding_days", columns="group_name")
    rows: list[dict[str, object]] = []
    for hold_day in HOLD_DAYS:
        if hold_day not in pivot_df.index:
            continue
        three_avg = pivot_df.at[hold_day, ("avg_return", "three_day_all")] if ("avg_return", "three_day_all") in pivot_df.columns else pd.NA
        two_avg = pivot_df.at[hold_day, ("avg_return", "two_day_all")] if ("avg_return", "two_day_all") in pivot_df.columns else pd.NA
        three_win = pivot_df.at[hold_day, ("win_rate", "three_day_all")] if ("win_rate", "three_day_all") in pivot_df.columns else pd.NA
        two_win = pivot_df.at[hold_day, ("win_rate", "two_day_all")] if ("win_rate", "two_day_all") in pivot_df.columns else pd.NA
        three_count = pivot_df.at[hold_day, ("sample_count", "three_day_all")] if ("sample_count", "three_day_all") in pivot_df.columns else pd.NA
        two_count = pivot_df.at[hold_day, ("sample_count", "two_day_all")] if ("sample_count", "two_day_all") in pivot_df.columns else pd.NA

        leader = "-"
        if not pd.isna(three_avg) and not pd.isna(two_avg):
            if float(three_avg) > float(two_avg):
                leader = "three_day_all"
            elif float(two_avg) > float(three_avg):
                leader = "two_day_all"
            else:
                leader = "持平"

        rows.append(
            {
                "holding_days": hold_day,
                "three_day_count": int(three_count) if not pd.isna(three_count) else "-",
                "three_day_avg": format_pct(three_avg),
                "three_day_win": format_pct(three_win),
                "two_day_count": int(two_count) if not pd.isna(two_count) else "-",
                "two_day_avg": format_pct(two_avg),
                "two_day_win": format_pct(two_win),
                "avg_diff": format_pct(float(three_avg) - float(two_avg)) if not pd.isna(three_avg) and not pd.isna(two_avg) else "-",
                "leader": leader,
            }
        )

    if not rows:
        return "无数据。"

    return make_markdown_table(
        pd.DataFrame(rows),
        columns=[
            "holding_days",
            "three_day_count",
            "three_day_avg",
            "three_day_win",
            "two_day_count",
            "two_day_avg",
            "two_day_win",
            "avg_diff",
            "leader",
        ],
        rename_map={
            "holding_days": "持有天数",
            "three_day_count": "3反转样本",
            "three_day_avg": "3反转平均",
            "three_day_win": "3反转胜率",
            "two_day_count": "2反转样本",
            "two_day_avg": "2反转平均",
            "two_day_win": "2反转胜率",
            "avg_diff": "平均差(3减2)",
            "leader": "平均收益领先",
        },
    )


def build_key_findings(overall_summary_df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if overall_summary_df.empty:
        return findings

    primary_df = overall_summary_df[overall_summary_df["group_name"].isin(PRIMARY_GROUPS)].copy()
    if primary_df.empty:
        return findings

    for group_name in PRIMARY_GROUPS:
        group_df = primary_df[primary_df["group_name"] == group_name].dropna(subset=["avg_return"])
        if group_df.empty:
            continue
        best_row = group_df.sort_values(["avg_return", "win_rate"], ascending=[False, False]).iloc[0]
        findings.append(
            f"`{group_name}` 在整体样本里平均收益最高的持有期是 {int(best_row['holding_days'])} 日，"
            f"平均涨跌幅 {format_pct(float(best_row['avg_return']))}，胜率 {format_pct(float(best_row['win_rate']))}。"
        )

    for hold_day in HOLD_DAYS:
        hold_df = primary_df[primary_df["holding_days"] == hold_day].set_index("group_name")
        if not {"three_day_all", "two_day_all"}.issubset(set(hold_df.index)):
            continue
        three_avg = hold_df.at["three_day_all", "avg_return"]
        two_avg = hold_df.at["two_day_all", "avg_return"]
        if pd.isna(three_avg) or pd.isna(two_avg):
            continue
        if float(three_avg) > float(two_avg):
            leader = "3反转"
            diff = float(three_avg) - float(two_avg)
        elif float(two_avg) > float(three_avg):
            leader = "2反转"
            diff = float(two_avg) - float(three_avg)
        else:
            leader = "两者持平"
            diff = 0.0
        if leader == "两者持平":
            findings.append(f"持有 {hold_day} 日时，2反转和3反转整体平均收益基本持平。")
        else:
            findings.append(
                f"持有 {hold_day} 日时，整体平均收益由{leader}领先，差值约 {format_pct(diff)}。"
            )

    return findings


def build_report(
    detail_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    meta: dict[str, object],
    output_paths: dict[str, Path],
) -> str:
    overall_counts_df = counts_df[counts_df["scope"] == "overall"].copy() if not counts_df.empty else pd.DataFrame()
    monthly_counts_df = counts_df[counts_df["scope"] == "monthly"].copy() if not counts_df.empty else pd.DataFrame()
    overall_summary_df = summary_df[summary_df["scope"] == "overall"].copy() if not summary_df.empty else pd.DataFrame()
    monthly_summary_df = summary_df[summary_df["scope"] == "monthly"].copy() if not summary_df.empty else pd.DataFrame()

    months = []
    if not detail_df.empty:
        months = sorted(detail_df["signal_month"].dropna().unique().tolist())

    lines: list[str] = []
    lines.append("# 2026年4-6月 周线多头 + 空转多 两套形态前瞻收益对比")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append(f"- 信号统计区间：`{meta['signal_start']}` 到 `{meta['signal_end']}`。")
    lines.append("- 先按项目现有 `weekly bullish` 周线母池口径，确定每个交易日所在周是否处于周线多头母池。")
    lines.append("- 只在母池内统计当日 `kong_turn_bull` 信号，再按 `three_day_kong_turn_bull` / `two_day_kong_turn_bull` 两套布尔字段分组。")
    lines.append("- 主收益口径：`后N日收盘价 / 信号日收盘价 - 1`，也就是相对信号日收盘的后 1 到 5 个交易日收盘涨跌幅。")
    lines.append("- 因为 2 日与 3 日形态可能同日同时满足，所以报告同时给出 `three_day_all`、`two_day_all`、`both`、`three_day_only`、`two_day_only`。")
    lines.append("- 晚于可用日线数据末尾的前瞻收益不会强行补值，因此不同持有期的样本数可能不同。")
    lines.append("")
    lines.append("## 样本概览")
    lines.append("")
    lines.append(f"- 处理股票文件数：{meta['processed_symbols']}")
    lines.append(f"- 缺少周线文件数：{meta['missing_weekly']}")
    lines.append(f"- 空日线文件数：{meta['skipped_empty_daily']}")
    lines.append(f"- 空周线或不足两根周K文件数：{meta['skipped_empty_weekly']}")
    lines.append(f"- 可用最新日线日期：`{meta['latest_daily_date'] or '-'}`")
    lines.append(f"- 总信号事件数：{len(detail_df)}")
    lines.append(f"- 去重股票数：{int(detail_df['file_symbol'].nunique()) if not detail_df.empty else 0}")
    lines.append(f"- 有信号的交易日数：{int(detail_df['signal_date'].nunique()) if not detail_df.empty else 0}")
    lines.append("")
    lines.append("### 整体分组样本数")
    lines.append("")
    lines.append(render_counts_table(overall_counts_df[overall_counts_df["group_name"].isin(ALL_GROUPS)] if not overall_counts_df.empty else pd.DataFrame()))
    lines.append("")
    for signal_month in months:
        month_counts_df = monthly_counts_df[monthly_counts_df["signal_month"] == signal_month].copy() if not monthly_counts_df.empty else pd.DataFrame()
        lines.append(f"### {signal_month} 分组样本数")
        lines.append("")
        lines.append(render_counts_table(month_counts_df[month_counts_df["group_name"].isin(ALL_GROUPS)] if not month_counts_df.empty else pd.DataFrame()))
        lines.append("")

    lines.append("## 2反转 vs 3反转：整体持有期对比")
    lines.append("")
    lines.append(render_pairwise_comparison(overall_summary_df))
    lines.append("")
    lines.append("## 整体分组表现")
    lines.append("")
    lines.append(render_return_table(overall_summary_df, ALL_GROUPS))
    lines.append("")

    for signal_month in months:
        month_summary_df = monthly_summary_df[monthly_summary_df["signal_month"] == signal_month].copy() if not monthly_summary_df.empty else pd.DataFrame()
        lines.append(f"## {signal_month} 月度对比")
        lines.append("")
        lines.append(render_pairwise_comparison(month_summary_df, signal_month=None) if month_summary_df.empty else render_pairwise_comparison(summary_df, signal_month=signal_month))
        lines.append("")
        lines.append(render_return_table(month_summary_df, ALL_GROUPS))
        lines.append("")

    findings = build_key_findings(overall_summary_df)
    lines.append("## 关键观察")
    lines.append("")
    if findings:
        for finding in findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- 没有足够样本生成稳定结论。")
    lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_start = pd.Timestamp(args.signal_start).normalize()
    signal_end = pd.Timestamp(args.signal_end).normalize()

    detail_df, meta = build_signal_detail(
        data_root=Path(args.data_root),
        signal_start=signal_start,
        signal_end=signal_end,
        limit=args.limit,
    )
    grouped_detail_df = expand_groups(detail_df)
    counts_df = build_counts_summary(grouped_detail_df)
    summary_df = build_return_summary(grouped_detail_df)

    suffix = f"{signal_start.date().isoformat()}_to_{signal_end.date().isoformat()}"
    output_paths = {
        "detail": output_dir / f"weekly_bullish_kong_turn_bull_forward_compare_detail_{suffix}.csv",
        "counts": output_dir / f"weekly_bullish_kong_turn_bull_forward_compare_counts_{suffix}.csv",
        "summary": output_dir / f"weekly_bullish_kong_turn_bull_forward_compare_summary_{suffix}.csv",
        "report": output_dir / f"weekly_bullish_kong_turn_bull_forward_compare_report_{suffix}.md",
    }

    detail_df.to_csv(output_paths["detail"], index=False, encoding="utf-8-sig")
    counts_df.to_csv(output_paths["counts"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_paths["summary"], index=False, encoding="utf-8-sig")

    report_text = build_report(
        detail_df=detail_df,
        counts_df=counts_df,
        summary_df=summary_df,
        meta=meta,
        output_paths=output_paths,
    )
    output_paths["report"].write_text(report_text, encoding="utf-8")

    print(f"detail_rows={len(detail_df)}")
    print(f"grouped_rows={len(grouped_detail_df)}")
    print(f"counts_path={output_paths['counts']}")
    print(f"summary_path={output_paths['summary']}")
    print(f"report_path={output_paths['report']}")
    if not summary_df.empty:
        print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
