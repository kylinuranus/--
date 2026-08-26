from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DATA_ROOT = Path(r"D:\a_stock_data")
DEFAULT_INPUT = SCREENING_RESULTS_DIR / "weekly_analysis_2026-06-01_to_2026-06-05.csv"
DAY_TYPE_LABELS = {
    "1_non_trend": "1.无趋势交易日",
    "2_normal_variation_up": "2.均衡延伸交易日-上延伸",
    "2_normal_variation_down": "2.均衡延伸交易日-下延伸",
    "3_neutral": "3.中心中性交易日",
    "4_balance": "4.均衡交易日",
    "4_balance_up": "4.均衡交易日-偏强收盘",
    "4_balance_down": "4.均衡交易日-偏弱收盘",
    "5_neutral_extreme_up": "5.端点中性交易日-收高",
    "5_neutral_extreme_down": "5.端点中性交易日-收低",
    "6_double_dist_up": "6.双头趋势交易日-上移",
    "6_double_dist_down": "6.双头趋势交易日-下移",
    "7_trend_up": "7.趋势交易日-上行",
    "7_trend_down": "7.趋势交易日-下行",
    "missing_30m": "缺少30分钟数据",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze weekly stocks by weekly trend and intraday trading-day type."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Weekly analysis CSV path.")
    parser.add_argument("--start", default="2026-06-01", help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-06-05", help="End date in YYYY-MM-DD.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Local parquet root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCREENING_RESULTS_DIR,
        help="Directory for generated CSV and text files.",
    )
    return parser.parse_args()


def safe_pct(change: float, base: float) -> float:
    if base == 0:
        return 0.0
    return change / base * 100.0


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return "; ".join(f"{label}={count}" for label, count in counter.items())


def format_float_counter(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return "; ".join(f"{label}={value:.4f}" for label, value in counter.items())


def classify_intraday_type(day_30m: pd.DataFrame) -> tuple[str, dict[str, float]]:
    if day_30m.empty or len(day_30m) < 4:
        return "missing_30m", {}

    bars = day_30m.sort_index().copy()
    open_price = float(bars["open"].iloc[0])
    close_price = float(bars["close"].iloc[-1])
    day_high = float(bars["high"].max())
    day_low = float(bars["low"].min())
    day_range = max(day_high - day_low, 1e-9)

    initial_balance = bars.iloc[:2]
    ib_high = float(initial_balance["high"].max())
    ib_low = float(initial_balance["low"].min())
    ib_range = max(ib_high - ib_low, 1e-9)

    up_extension = max(0.0, day_high - ib_high) / day_range
    down_extension = max(0.0, ib_low - day_low) / day_range
    close_position = (close_price - day_low) / day_range
    direction_efficiency = abs(close_price - open_price) / day_range
    ib_ratio = ib_range / day_range

    first_half = bars.iloc[: len(bars) // 2]
    second_half = bars.iloc[len(bars) // 2 :]
    first_low = float(first_half["low"].min())
    first_high = float(first_half["high"].max())
    second_low = float(second_half["low"].min())
    second_high = float(second_half["high"].max())
    first_range = max(first_high - first_low, 1e-9)
    second_range = max(second_high - second_low, 1e-9)
    first_mid = (first_high + first_low) / 2.0
    second_mid = (second_high + second_low) / 2.0
    overlap = max(0.0, min(first_high, second_high) - max(first_low, second_low))
    overlap_ratio = overlap / min(first_range, second_range)
    distribution_shift = abs(second_mid - first_mid) / day_range

    metrics = {
        "day_range": day_range,
        "ib_ratio": ib_ratio,
        "up_extension": up_extension,
        "down_extension": down_extension,
        "close_position": close_position,
        "direction_efficiency": direction_efficiency,
        "distribution_shift": distribution_shift,
        "distribution_overlap_ratio": overlap_ratio,
    }

    if up_extension >= 0.35 and down_extension < 0.10 and direction_efficiency >= 0.55 and close_position >= 0.82:
        return "7_trend_up", metrics
    if down_extension >= 0.35 and up_extension < 0.10 and direction_efficiency >= 0.55 and close_position <= 0.18:
        return "7_trend_down", metrics

    if distribution_shift >= 0.38 and overlap_ratio <= 0.35 and first_range <= 0.75 * day_range and second_range <= 0.75 * day_range:
        if second_mid > first_mid and close_position >= 0.58:
            return "6_double_dist_up", metrics
        if second_mid < first_mid and close_position <= 0.42:
            return "6_double_dist_down", metrics

    if up_extension >= 0.15 and down_extension >= 0.15:
        if 0.35 <= close_position <= 0.65:
            return "3_neutral", metrics
        if close_position > 0.65:
            return "5_neutral_extreme_up", metrics
        return "5_neutral_extreme_down", metrics

    if ib_ratio >= 0.85 and 0.35 <= close_position <= 0.65 and direction_efficiency < 0.20:
        return "1_non_trend", metrics

    if ib_ratio >= 0.65 and 0.30 <= close_position <= 0.70 and direction_efficiency < 0.35:
        if close_position > 0.55:
            return "4_balance_up", metrics
        if close_position < 0.45:
            return "4_balance_down", metrics
        return "4_balance", metrics

    if up_extension >= down_extension:
        if up_extension >= 0.12:
            return "2_normal_variation_up", metrics
        if close_position > 0.55:
            return "4_balance_up", metrics
        return "4_balance", metrics

    if down_extension >= 0.12:
        return "2_normal_variation_down", metrics
    if close_position < 0.45:
        return "4_balance_down", metrics
    return "4_balance", metrics


def classify_weekly_trend(detail_df: pd.DataFrame) -> str:
    ordered = detail_df.sort_values("date").copy()
    if ordered.empty:
        return "分析缺失"

    prev_close = float(ordered["prev_close"].iloc[0])
    week_open = float(ordered["open"].iloc[0])
    week_close = float(ordered["close"].iloc[-1])
    week_high = float(ordered["high"].max())
    week_low = float(ordered["low"].min())
    week_range = max(week_high - week_low, 1e-9)
    close_position = (week_close - week_low) / week_range
    body_efficiency = abs(week_close - week_open) / week_range
    changes = ordered["close_change_prev"].tolist()
    up_days = sum(change > 0 for change in changes)
    down_days = sum(change < 0 for change in changes)
    first_two = sum(changes[:2])
    last_two = sum(changes[-2:])
    total_change = sum(changes)

    if week_close < prev_close and week_close > week_open:
        return "周K收阳但相对上周仍弱"
    if total_change > 0 and body_efficiency >= 0.55 and close_position >= 0.75 and down_days <= 1:
        return "周强趋势上行"
    if total_change > 0 and first_two <= 0 and last_two > 0 and close_position >= 0.65:
        return "周内反转上行"
    if total_change > 0 and last_two >= max(total_change * 0.50, 0.0) and close_position >= 0.68:
        return "周后程加速上行"
    if total_change > 0 and up_days >= 2:
        return "周震荡抬升"
    if total_change > 0:
        return "周缓步抬升"
    return "周内偏弱震荡"


def analyze_stock(
    stock_row: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    data_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    file_symbol = str(stock_row["file_symbol"])
    symbol = str(stock_row["symbol"])
    name = stock_row["name"]
    daily_path = data_root / "daily" / f"{file_symbol}.parquet"
    intraday_path = data_root / "30m" / f"{file_symbol}.parquet"

    if not daily_path.exists():
        return [], {
            "file_symbol": file_symbol,
            "symbol": symbol,
            "name": name,
            "analysis_status": "daily_missing",
        }

    daily_df = pd.read_parquet(daily_path).sort_index()
    previous_bar = daily_df.loc[: start_ts - pd.Timedelta(days=1)].tail(1)
    week_df = daily_df.loc[str(start_ts.date()) : str(end_ts.date())].copy()

    if previous_bar.empty or len(week_df) == 0:
        return [], {
            "file_symbol": file_symbol,
            "symbol": symbol,
            "name": name,
            "analysis_status": "daily_window_missing",
        }

    intraday_df = pd.DataFrame()
    if intraday_path.exists():
        intraday_df = pd.read_parquet(intraday_path).sort_index()
        intraday_df = intraday_df.loc[str(start_ts.date()) : str(end_ts.date())]

    detail_rows: list[dict[str, object]] = []
    prev_close = float(previous_bar["close"].iloc[-1])
    for trade_time, day_row in week_df.iterrows():
        trade_day = pd.Timestamp(trade_time).normalize()
        intraday_day = intraday_df[intraday_df.index.normalize() == trade_day] if not intraday_df.empty else pd.DataFrame()
        day_type_code, metrics = classify_intraday_type(intraday_day)

        close_price = float(day_row["close"])
        open_price = float(day_row["open"])
        high_price = float(day_row["high"])
        low_price = float(day_row["low"])
        day_change = close_price - prev_close
        day_return_pct = safe_pct(day_change, prev_close)

        detail_rows.append(
            {
                "file_symbol": file_symbol,
                "symbol": symbol,
                "name": name,
                "date": trade_day.date().isoformat(),
                "week_dir": stock_row.get("week_dir"),
                "week_candle": stock_row.get("week_candle"),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "prev_close": prev_close,
                "close_change_prev": day_change,
                "close_return_prev_pct": day_return_pct,
                "green_candle": close_price > open_price,
                "day_up_prevclose": day_change > 0,
                "day_type_code": day_type_code,
                "day_type_label": DAY_TYPE_LABELS[day_type_code],
                "proxy_day_range": metrics.get("day_range"),
                "proxy_ib_ratio": metrics.get("ib_ratio"),
                "proxy_up_extension": metrics.get("up_extension"),
                "proxy_down_extension": metrics.get("down_extension"),
                "proxy_close_position": metrics.get("close_position"),
                "proxy_direction_efficiency": metrics.get("direction_efficiency"),
                "proxy_distribution_shift": metrics.get("distribution_shift"),
                "proxy_distribution_overlap_ratio": metrics.get("distribution_overlap_ratio"),
            }
        )
        prev_close = close_price

    detail_df = pd.DataFrame(detail_rows)
    week_close = float(detail_df["close"].iloc[-1])
    base_prev_close = float(detail_df["prev_close"].iloc[0])
    total_week_change = week_close - base_prev_close
    total_week_return_pct = safe_pct(total_week_change, base_prev_close)
    positive_detail = detail_df[detail_df["day_up_prevclose"]].copy()
    positive_change_sum = float(positive_detail["close_change_prev"].sum()) if not positive_detail.empty else 0.0
    up_day_type_counter = Counter(positive_detail["day_type_label"].tolist())
    up_day_type_contrib: Counter[str] = Counter()
    for label, value in positive_detail.groupby("day_type_label")["close_change_prev"].sum().items():
        up_day_type_contrib[label] = float(value)

    dominant_up_day_type = ""
    dominant_up_day_type_contribution = 0.0
    dominant_up_day_type_share = 0.0
    if up_day_type_contrib:
        dominant_up_day_type = max(
            up_day_type_contrib.items(),
            key=lambda item: (item[1], up_day_type_counter[item[0]], item[0]),
        )[0]
        dominant_up_day_type_contribution = up_day_type_contrib[dominant_up_day_type]
        if positive_change_sum > 0:
            dominant_up_day_type_share = dominant_up_day_type_contribution / positive_change_sum

    stock_summary = {
        "file_symbol": file_symbol,
        "symbol": symbol,
        "name": name,
        "analysis_status": "ok",
        "week_dir": stock_row.get("week_dir"),
        "week_candle": stock_row.get("week_candle"),
        "focus_weekly_bull_candle": stock_row.get("week_candle") == "bull",
        "week_open": float(detail_df["open"].iloc[0]),
        "week_close": week_close,
        "week_total_change_prevclose": total_week_change,
        "week_total_return_prevclose_pct": total_week_return_pct,
        "weekly_trend_class": classify_weekly_trend(detail_df),
        "up_days_count": int(detail_df["day_up_prevclose"].sum()),
        "green_candle_days_count": int(detail_df["green_candle"].sum()),
        "positive_change_pool": positive_change_sum,
        "dominant_up_day_type": dominant_up_day_type,
        "dominant_up_day_type_contribution": dominant_up_day_type_contribution,
        "dominant_up_day_type_share": dominant_up_day_type_share,
        "up_day_type_counts": format_counter(up_day_type_counter),
        "up_day_type_contributions": format_float_counter(up_day_type_contrib),
        "up_day_type_sequence": "|".join(
            f"{item.date}:{item.day_type_label}"
            for item in positive_detail[["date", "day_type_label"]].itertuples(index=False, name="Row")
        ),
        "all_day_type_sequence": "|".join(
            f"{item.date}:{item.day_type_label}"
            for item in detail_df[["date", "day_type_label"]].itertuples(index=False, name="Row")
        ),
    }
    return detail_rows, stock_summary


def build_stats_text(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    input_count: int,
    detail_df: pd.DataFrame,
    stock_summary_df: pd.DataFrame,
    bull_summary_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"分析区间: {start_ts.date()} 至 {end_ts.date()}")
    lines.append(f"输入样本数: {input_count}")
    lines.append(f"逐日明细行数: {len(detail_df)}")
    lines.append(f"成功生成个股汇总: {len(stock_summary_df)}")
    lines.append(f"周K收阳样本数: {len(bull_summary_df)}")
    lines.append("")

    if not bull_summary_df.empty:
        positive_day_df = detail_df[
            (detail_df["week_candle"] == "bull") & (detail_df["day_up_prevclose"])
        ].copy()

        lines.append("周趋势分布（仅周K收阳样本）:")
        for label, count in bull_summary_df["weekly_trend_class"].value_counts().items():
            lines.append(f"{label}: {count}")
        lines.append("")

        lines.append("上涨日交易日类型分布（仅周K收阳样本，按收盘高于前收定义）:")
        for label, count in positive_day_df["day_type_label"].value_counts().items():
            lines.append(f"{label}: {count}")
        lines.append("")

        lines.append("主要上涨交易日类型分布（按单只股票正向涨幅贡献最大者）:")
        for label, count in bull_summary_df["dominant_up_day_type"].value_counts().items():
            lines.append(f"{label}: {count}")
        lines.append("")

        lines.append("周K收阳但相对上周仍弱的股票数:")
        weaker_count = int((bull_summary_df["weekly_trend_class"] == "周K收阳但相对上周仍弱").sum())
        lines.append(str(weaker_count))

    return "\n".join(lines) + "\n"


def analyze_weekly_day_types(
    input_csv: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    data_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    input_df = pd.read_csv(input_csv, dtype={"symbol": str, "file_symbol": str})
    detail_rows: list[dict[str, object]] = []
    stock_summaries: list[dict[str, object]] = []

    for _, stock_row in input_df.iterrows():
        stock_detail_rows, stock_summary = analyze_stock(stock_row, start_ts, end_ts, data_root)
        detail_rows.extend(stock_detail_rows)
        stock_summaries.append(stock_summary)

    detail_df = pd.DataFrame(detail_rows).sort_values(["file_symbol", "date"]).reset_index(drop=True)
    stock_summary_df = pd.DataFrame(stock_summaries).sort_values(["analysis_status", "file_symbol"]).reset_index(drop=True)
    bull_summary_df = stock_summary_df[
        (stock_summary_df["analysis_status"] == "ok") & (stock_summary_df["focus_weekly_bull_candle"])
    ].copy()

    suffix = f"{start_ts.date()}_to_{end_ts.date()}"
    detail_path = output_dir / f"weekly_day_type_detail_{suffix}.csv"
    stock_summary_path = output_dir / f"weekly_day_type_stock_summary_{suffix}.csv"
    bull_summary_path = output_dir / f"weekly_bull_day_type_stock_summary_{suffix}.csv"
    stats_path = output_dir / f"weekly_bull_day_type_stats_{suffix}.txt"

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    stock_summary_df.to_csv(stock_summary_path, index=False, encoding="utf-8-sig")
    bull_summary_df.to_csv(bull_summary_path, index=False, encoding="utf-8-sig")

    stats_text = build_stats_text(
        start_ts=start_ts,
        end_ts=end_ts,
        input_count=len(input_df),
        detail_df=detail_df,
        stock_summary_df=stock_summary_df,
        bull_summary_df=bull_summary_df,
    )
    stats_path.write_text(stats_text, encoding="utf-8")
    return detail_df, stock_summary_df, bull_summary_df, detail_path, stock_summary_path, bull_summary_path


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_df, stock_summary_df, bull_summary_df, detail_path, stock_summary_path, bull_summary_path = analyze_weekly_day_types(
        input_csv=Path(args.input),
        start_ts=start_ts,
        end_ts=end_ts,
        data_root=Path(args.data_root),
        output_dir=output_dir,
    )

    print(f"detail_rows={len(detail_df)}")
    print(f"stock_summary_rows={len(stock_summary_df)}")
    print(f"weekly_bull_summary_rows={len(bull_summary_df)}")
    print(f"detail_path={detail_path}")
    print(f"stock_summary_path={stock_summary_path}")
    print(f"weekly_bull_summary_path={bull_summary_path}")


if __name__ == "__main__":
    main()
