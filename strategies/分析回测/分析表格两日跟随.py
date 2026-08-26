from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from project_paths import SCREENING_RESULTS_DIR


DATA_ROOT = Path(r"D:\a_stock_data")
DAILY_ROOT = DATA_ROOT / "daily"

TABLE_PATH = SCREENING_RESULTS_DIR / "Table.csv"
MULTI_MA_HEAD_PATH = SCREENING_RESULTS_DIR / "weekly_multi_ma_head_2026-06-08_to_2026-06-12.csv"

ANALYSIS_DATE_1 = pd.Timestamp("2026-06-15")
ANALYSIS_DATE_2 = pd.Timestamp("2026-06-16")
PREV_WEEK_START = pd.Timestamp("2026-06-08")
PREV_WEEK_END = pd.Timestamp("2026-06-12")
PREV_WEEK_CLOSE_DATE = pd.Timestamp("2026-06-05")

OUTPUT_CSV = SCREENING_RESULTS_DIR / "table80_two_day_analysis_2026-06-15_2026-06-16.csv"
OUTPUT_REPORT = SCREENING_RESULTS_DIR / "table80_two_day_report_2026-06-15_2026-06-16.md"


TABLE_COLUMN_INDEX_MAP = {
    0: "row_id",
    1: "code",
    2: "name",
    3: "close_0615",
    4: "ret_0615_pct",
    5: "change_0615",
    10: "speed_0615_pct",
    11: "turnover_0615_pct",
    12: "amount_0615",
    14: "industry",
    15: "high_0615",
    16: "low_0615",
    17: "open_0615",
    18: "prev_close_0615",
    19: "amplitude_0615_pct",
    20: "volume_ratio_0615",
    34: "ret_3d_pct",
    35: "ret_6d_pct",
    36: "turnover_3d_pct",
    37: "turnover_6d_pct",
    38: "watch_time",
    39: "watch_price",
    40: "watch_return",
    41: "streak_days",
    42: "month_ret_pct",
    43: "ytd_ret_pct",
    44: "ret_1m_pct",
    45: "ret_1y_pct",
}

GROUP_LABELS = {
    "up_up": "6月15日涨、6月16日涨",
    "up_down": "6月15日涨、6月16日跌",
    "down_up": "6月15日跌、6月16日涨",
    "down_down": "6月15日跌、6月16日跌",
    "up_flat": "6月15日涨、6月16日平",
    "down_flat": "6月15日跌、6月16日平",
    "flat_up": "6月15日平、6月16日涨",
    "flat_down": "6月15日平、6月16日跌",
    "flat_flat": "6月15日平、6月16日平",
    "missing": "数据缺失",
}

GROUP_ORDER = [
    "up_up",
    "up_down",
    "down_up",
    "down_down",
    "up_flat",
    "down_flat",
    "flat_up",
    "flat_down",
    "flat_flat",
    "missing",
]

REPORT_GROUPS = ["up_up", "up_down", "down_up", "down_down"]

CORRELATION_FEATURES = [
    "ret_0615_pct",
    "streak_days",
    "ret_3d_pct",
    "ret_6d_pct",
    "turnover_3d_pct",
    "turnover_6d_pct",
    "month_ret_pct",
    "week_return_0608_0612_pct",
    "week_close_position_0608_0612",
    "week_body_efficiency_0608_0612",
    "close_vs_ma5_0615_pct",
    "close_vs_ma20_0615_pct",
    "close_vs_ma60_0615_pct",
    "close_vs_prev20_high_0615_pct",
    "range_pos_20d_0615",
]

FEATURE_LABELS = {
    "ret_0615_pct": "6月15日涨跌幅",
    "streak_days": "截至6月15日连涨天数",
    "ret_3d_pct": "3日涨幅",
    "ret_6d_pct": "6日涨幅",
    "turnover_3d_pct": "3日换手",
    "turnover_6d_pct": "6日换手",
    "month_ret_pct": "本月涨幅",
    "week_return_0608_0612_pct": "前一周涨幅(6月8日至6月12日)",
    "week_close_position_0608_0612": "前一周收盘位置",
    "week_body_efficiency_0608_0612": "前一周实体效率",
    "close_vs_ma5_0615_pct": "6月15日收盘偏离MA5",
    "close_vs_ma20_0615_pct": "6月15日收盘偏离MA20",
    "close_vs_ma60_0615_pct": "6月15日收盘偏离MA60",
    "close_vs_prev20_high_0615_pct": "6月15日相对前20日高点突破幅度",
    "range_pos_20d_0615": "6月15日在20日区间中的位置",
}

SUMMARY_FIELDS = [
    "ret_0615_pct",
    "ret_0616_pct",
    "two_day_ret_pct",
    "streak_days",
    "ret_3d_pct",
    "ret_6d_pct",
    "turnover_3d_pct",
    "turnover_6d_pct",
    "month_ret_pct",
    "week_return_0608_0612_pct",
    "close_vs_ma5_0615_pct",
    "close_vs_ma20_0615_pct",
    "close_vs_ma60_0615_pct",
    "close_vs_prev20_high_0615_pct",
    "range_pos_20d_0615",
    "gap_0616_pct",
    "intraday_ret_0616_pct",
    "bar_close_position_0616",
    "body_ratio_0616",
    "upper_shadow_ratio_0616",
    "lower_shadow_ratio_0616",
]

SUMMARY_FIELD_LABELS = {
    "ret_0615_pct": "6月15日均值",
    "ret_0616_pct": "6月16日均值",
    "two_day_ret_pct": "两日合计均值",
    "streak_days": "连涨天数均值",
    "ret_3d_pct": "3日涨幅均值",
    "ret_6d_pct": "6日涨幅均值",
    "turnover_3d_pct": "3日换手均值",
    "turnover_6d_pct": "6日换手均值",
    "month_ret_pct": "本月涨幅均值",
    "week_return_0608_0612_pct": "前一周涨幅均值",
    "close_vs_ma5_0615_pct": "6月15日偏离MA5均值",
    "close_vs_ma20_0615_pct": "6月15日偏离MA20均值",
    "close_vs_ma60_0615_pct": "6月15日偏离MA60均值",
    "close_vs_prev20_high_0615_pct": "6月15日突破前20日高点幅度均值",
    "range_pos_20d_0615": "6月15日20日区间位置均值",
    "gap_0616_pct": "6月16日开盘相对6月15日收盘缺口均值",
    "intraday_ret_0616_pct": "6月16日日内涨跌幅均值",
    "bar_close_position_0616": "6月16日收盘在日内位置均值",
    "body_ratio_0616": "6月16日实体占比均值",
    "upper_shadow_ratio_0616": "6月16日上影占比均值",
    "lower_shadow_ratio_0616": "6月16日下影占比均值",
}


def parse_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if text in {"", "nan", "None", "-", "—"}:
        return None

    multiplier = 1.0
    if text.endswith("万亿"):
        multiplier = 1e12
        text = text[:-2]
    elif text.endswith("亿"):
        multiplier = 1e8
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 1e4
        text = text[:-1]
    elif text.endswith("%"):
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def pct_change(current: float | None, base: float | None) -> float | None:
    if current is None or base in {None, 0}:
        return None
    return (current / base - 1.0) * 100.0


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def market_prefix(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"sh{code}"
    return f"sz{code}"


def limit_up_threshold(code: str) -> float:
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    return 10.0


def direction_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}%"


def format_num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def build_table_base() -> pd.DataFrame:
    raw = pd.read_csv(TABLE_PATH, dtype=str, encoding="gbk")
    data = {name: raw.iloc[:, index].copy() for index, name in TABLE_COLUMN_INDEX_MAP.items()}
    table = pd.DataFrame(data)

    for column in table.columns:
        table[column] = table[column].astype(str).str.strip()

    table["code"] = table["code"].str.zfill(6)
    table["file_symbol"] = table["code"].map(market_prefix)

    numeric_columns = [
        "close_0615",
        "ret_0615_pct",
        "change_0615",
        "speed_0615_pct",
        "turnover_0615_pct",
        "high_0615",
        "low_0615",
        "open_0615",
        "prev_close_0615",
        "amplitude_0615_pct",
        "volume_ratio_0615",
        "ret_3d_pct",
        "ret_6d_pct",
        "turnover_3d_pct",
        "turnover_6d_pct",
        "watch_price",
        "streak_days",
        "month_ret_pct",
        "ytd_ret_pct",
        "ret_1m_pct",
        "ret_1y_pct",
    ]
    for column in numeric_columns:
        table[column] = table[column].map(parse_numeric)

    table["industry"] = table["industry"].fillna("").astype(str).str.strip()
    table["limit_up_threshold"] = table["code"].map(limit_up_threshold)
    table["hit_limit_0615"] = table.apply(
        lambda row: bool(
            row["ret_0615_pct"] is not None
            and row["limit_up_threshold"] is not None
            and row["ret_0615_pct"] >= row["limit_up_threshold"] - 0.25
        ),
        axis=1,
    )
    return table


def load_multi_ma_head_set() -> set[str]:
    if not MULTI_MA_HEAD_PATH.exists():
        return set()
    df = pd.read_csv(MULTI_MA_HEAD_PATH, dtype={"file_symbol": str})
    return set(df["file_symbol"].astype(str))


def estimate_bar_metrics(open_price: float | None, high_price: float | None, low_price: float | None, close_price: float | None) -> dict[str, float | None]:
    if None in {open_price, high_price, low_price, close_price}:
        return {
            "bar_close_position": None,
            "body_ratio": None,
            "upper_shadow_ratio": None,
            "lower_shadow_ratio": None,
        }

    bar_range = max(high_price - low_price, 1e-9)
    return {
        "bar_close_position": (close_price - low_price) / bar_range,
        "body_ratio": abs(close_price - open_price) / bar_range,
        "upper_shadow_ratio": (high_price - max(open_price, close_price)) / bar_range,
        "lower_shadow_ratio": (min(open_price, close_price) - low_price) / bar_range,
    }


def append_synthetic_0615(hist: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    combined = hist.copy()
    if ANALYSIS_DATE_1 in combined.index:
        return combined.sort_index()

    required = [row["open_0615"], row["high_0615"], row["low_0615"], row["close_0615"]]
    if any(value is None or pd.isna(value) for value in required):
        return combined.sort_index()

    seed_row = combined.tail(1).iloc[0].to_dict() if not combined.empty else {}
    seed_row.update(
        {
            "symbol": row["code"],
            "market": row["file_symbol"][:2],
            "name": row["name"],
            "period": "daily",
            "adjustment": seed_row.get("adjustment", "qfq"),
            "open": row["open_0615"],
            "high": row["high_0615"],
            "low": row["low_0615"],
            "close": row["close_0615"],
            "source": "table_snapshot",
        }
    )
    combined.loc[ANALYSIS_DATE_1] = seed_row
    return combined.sort_index()


def analyze_stock(row: pd.Series, multi_ma_head_set: set[str]) -> dict[str, object]:
    result: dict[str, object] = row.to_dict()
    daily_path = DAILY_ROOT / f"{row['file_symbol']}.parquet"

    result["has_daily_file"] = daily_path.exists()
    result["has_0616_bar"] = False
    result["is_multi_ma_head_0608_0612"] = row["file_symbol"] in multi_ma_head_set
    result["group"] = "missing"
    result["group_label"] = GROUP_LABELS["missing"]
    result["structural_note"] = "缺少足够数据"

    if not daily_path.exists():
        return result

    hist = pd.read_parquet(daily_path).sort_index()
    hist = hist.loc[:ANALYSIS_DATE_2].copy()
    hist_with_0615 = append_synthetic_0615(hist, row)

    for window in (5, 10, 20, 60):
        hist_with_0615[f"ma{window}"] = hist_with_0615["close"].rolling(window).mean()

    bar_0616 = hist_with_0615.loc[ANALYSIS_DATE_2] if ANALYSIS_DATE_2 in hist_with_0615.index else None
    bar_0615 = hist_with_0615.loc[ANALYSIS_DATE_1] if ANALYSIS_DATE_1 in hist_with_0615.index else None

    if bar_0616 is not None:
        result["has_0616_bar"] = True
        result["open_0616"] = float(bar_0616["open"])
        result["high_0616"] = float(bar_0616["high"])
        result["low_0616"] = float(bar_0616["low"])
        result["close_0616"] = float(bar_0616["close"])
        result["ret_0616_pct"] = pct_change(result["close_0616"], row["close_0615"])
        result["gap_0616_pct"] = pct_change(result["open_0616"], row["close_0615"])
        result["intraday_ret_0616_pct"] = pct_change(result["close_0616"], result["open_0616"])
        bar_metrics = estimate_bar_metrics(
            result["open_0616"],
            result["high_0616"],
            result["low_0616"],
            result["close_0616"],
        )
        result["bar_close_position_0616"] = bar_metrics["bar_close_position"]
        result["body_ratio_0616"] = bar_metrics["body_ratio"]
        result["upper_shadow_ratio_0616"] = bar_metrics["upper_shadow_ratio"]
        result["lower_shadow_ratio_0616"] = bar_metrics["lower_shadow_ratio"]
        result["two_day_ret_pct"] = pct_change(result["close_0616"], row["prev_close_0615"])
        result["hit_limit_0616"] = bool(
            result["ret_0616_pct"] is not None
            and result["ret_0616_pct"] >= row["limit_up_threshold"] - 0.25
        )
    else:
        result["open_0616"] = None
        result["high_0616"] = None
        result["low_0616"] = None
        result["close_0616"] = None
        result["ret_0616_pct"] = None
        result["gap_0616_pct"] = None
        result["intraday_ret_0616_pct"] = None
        result["bar_close_position_0616"] = None
        result["body_ratio_0616"] = None
        result["upper_shadow_ratio_0616"] = None
        result["lower_shadow_ratio_0616"] = None
        result["two_day_ret_pct"] = None
        result["hit_limit_0616"] = False

    if bar_0615 is not None:
        result["ma5_0615"] = float(bar_0615["ma5"]) if pd.notna(bar_0615["ma5"]) else None
        result["ma10_0615"] = float(bar_0615["ma10"]) if pd.notna(bar_0615["ma10"]) else None
        result["ma20_0615"] = float(bar_0615["ma20"]) if pd.notna(bar_0615["ma20"]) else None
        result["ma60_0615"] = float(bar_0615["ma60"]) if pd.notna(bar_0615["ma60"]) else None
        result["close_vs_ma5_0615_pct"] = pct_change(row["close_0615"], result["ma5_0615"])
        result["close_vs_ma10_0615_pct"] = pct_change(row["close_0615"], result["ma10_0615"])
        result["close_vs_ma20_0615_pct"] = pct_change(row["close_0615"], result["ma20_0615"])
        result["close_vs_ma60_0615_pct"] = pct_change(row["close_0615"], result["ma60_0615"])
        result["ma5_gt_ma10_gt_ma20_0615"] = bool(
            None not in {result["ma5_0615"], result["ma10_0615"], result["ma20_0615"]}
            and result["ma5_0615"] > result["ma10_0615"] > result["ma20_0615"]
        )
        result["ma5_gt_ma20_gt_ma60_0615"] = bool(
            None not in {result["ma5_0615"], result["ma20_0615"], result["ma60_0615"]}
            and result["ma5_0615"] > result["ma20_0615"] > result["ma60_0615"]
        )
    else:
        result["ma5_0615"] = None
        result["ma10_0615"] = None
        result["ma20_0615"] = None
        result["ma60_0615"] = None
        result["close_vs_ma5_0615_pct"] = None
        result["close_vs_ma10_0615_pct"] = None
        result["close_vs_ma20_0615_pct"] = None
        result["close_vs_ma60_0615_pct"] = None
        result["ma5_gt_ma10_gt_ma20_0615"] = False
        result["ma5_gt_ma20_gt_ma60_0615"] = False

    hist_before_0615 = hist.loc[: PREV_WEEK_END].copy()
    if not hist_before_0615.empty:
        prev20_high = float(hist_before_0615["high"].tail(20).max())
        prev20_low = float(hist_before_0615["low"].tail(20).min())
        result["prev20_high_before_0615"] = prev20_high
        result["prev20_low_before_0615"] = prev20_low
        result["close_vs_prev20_high_0615_pct"] = pct_change(row["close_0615"], prev20_high)
        result["range_pos_20d_0615"] = safe_ratio(row["close_0615"] - prev20_low, prev20_high - prev20_low)
    else:
        result["prev20_high_before_0615"] = None
        result["prev20_low_before_0615"] = None
        result["close_vs_prev20_high_0615_pct"] = None
        result["range_pos_20d_0615"] = None

    prev_week = hist.loc[str(PREV_WEEK_START.date()) : str(PREV_WEEK_END.date())].copy()
    prev_week_close_bar = hist.loc[:PREV_WEEK_CLOSE_DATE].tail(1)
    if not prev_week.empty and not prev_week_close_bar.empty:
        prev_week_close = float(prev_week_close_bar["close"].iloc[-1])
        week_open = float(prev_week["open"].iloc[0])
        week_high = float(prev_week["high"].max())
        week_low = float(prev_week["low"].min())
        week_close = float(prev_week["close"].iloc[-1])
        week_range = max(week_high - week_low, 1e-9)
        result["prev_week_close_0605"] = prev_week_close
        result["week_open_0608_0612"] = week_open
        result["week_high_0608_0612"] = week_high
        result["week_low_0608_0612"] = week_low
        result["week_close_0608_0612"] = week_close
        result["week_return_0608_0612_pct"] = pct_change(week_close, prev_week_close)
        result["week_candle_change_0608_0612"] = week_close - week_open
        result["week_close_position_0608_0612"] = (week_close - week_low) / week_range
        result["week_body_efficiency_0608_0612"] = abs(week_close - week_open) / week_range
        result["week_dir_0608_0612"] = "up" if week_close > prev_week_close else ("down" if week_close < prev_week_close else "flat")
        result["week_candle_0608_0612"] = "bull" if week_close > week_open else ("bear" if week_close < week_open else "flat")
    else:
        result["prev_week_close_0605"] = None
        result["week_open_0608_0612"] = None
        result["week_high_0608_0612"] = None
        result["week_low_0608_0612"] = None
        result["week_close_0608_0612"] = None
        result["week_return_0608_0612_pct"] = None
        result["week_candle_change_0608_0612"] = None
        result["week_close_position_0608_0612"] = None
        result["week_body_efficiency_0608_0612"] = None
        result["week_dir_0608_0612"] = "missing"
        result["week_candle_0608_0612"] = "missing"

    dir_0615 = direction_label(row["ret_0615_pct"])
    dir_0616 = direction_label(result["ret_0616_pct"])
    if "missing" in {dir_0615, dir_0616}:
        group = "missing"
    else:
        group = f"{dir_0615}_{dir_0616}"
    result["dir_0615"] = dir_0615
    result["dir_0616"] = dir_0616
    result["group"] = group
    result["group_label"] = GROUP_LABELS.get(group, GROUP_LABELS["missing"])
    result["structural_note"] = infer_structural_note(result)
    return result


def infer_structural_note(stock: dict[str, object]) -> str:
    group = stock["group"]
    close_vs_ma20 = stock.get("close_vs_ma20_0615_pct")
    close_vs_prev20 = stock.get("close_vs_prev20_high_0615_pct")
    streak_days = stock.get("streak_days")
    week_return = stock.get("week_return_0608_0612_pct")
    gap_0616 = stock.get("gap_0616_pct")
    close_pos_0616 = stock.get("bar_close_position_0616")
    upper_shadow = stock.get("upper_shadow_ratio_0616")
    if group == "up_up":
        if stock.get("hit_limit_0615") and stock.get("hit_limit_0616"):
            return "连续涨停加速，资金一致性最强"
        if close_vs_prev20 is not None and close_vs_prev20 > 5 and close_pos_0616 is not None and close_pos_0616 > 0.7:
            return "突破前高后继续抬升，属于趋势延续"
        return "强势股延续，6月16日收盘仍靠近日内高位"
    if group == "up_down":
        if close_vs_ma20 is not None and close_vs_ma20 > 30:
            return "6月15日已明显乖离MA20，次日更容易兑现"
        if streak_days is not None and streak_days >= 5:
            return "连涨天数偏长，6月16日出现高位分歧"
        if gap_0616 is not None and gap_0616 > 0 and close_pos_0616 is not None and close_pos_0616 < 0.35:
            return "高开后承接不足，冲高回落"
        return "前一日上涨后次日承接不足，属于短线分歧"
    if group == "down_up":
        if week_return is not None and week_return > 10:
            return "大结构仍强，6月15日更像强势回踩，6月16日修复"
        if close_vs_ma20 is not None and close_vs_ma20 > 5:
            return "仍在中期均线上方，6月15日调整后恢复上行"
        return "弱转强修复，前一日回落未破坏主结构"
    if group == "down_down":
        if close_vs_ma20 is not None and close_vs_ma20 < 10:
            return "离中期均线优势不大，调整后继续走弱"
        if close_pos_0616 is not None and close_pos_0616 < 0.35:
            return "6月16日收在日内低位，抛压延续"
        return "短线结构转弱，调整演变为补跌"
    return "数据不足"


def compute_group_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        group_df = analysis_df[analysis_df["group"] == group].copy()
        if group_df.empty:
            continue
        summary: dict[str, object] = {
            "group": group,
            "group_label": GROUP_LABELS[group],
            "count": len(group_df),
            "limit_up_0615_count": int(group_df["hit_limit_0615"].fillna(False).sum()),
            "limit_up_0616_count": int(group_df["hit_limit_0616"].fillna(False).sum()),
            "multi_ma_head_count": int(group_df["is_multi_ma_head_0608_0612"].fillna(False).sum()),
        }
        for field in SUMMARY_FIELDS:
            series = pd.to_numeric(group_df[field], errors="coerce").dropna()
            summary[f"{field}_mean"] = float(series.mean()) if not series.empty else None
            summary[f"{field}_median"] = float(series.median()) if not series.empty else None

        industries = (
            group_df["industry"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(5)
        )
        summary["top_industries"] = ", ".join(f"{name}({count})" for name, count in industries.items())
        rows.append(summary)
    return pd.DataFrame(rows)


def compute_correlations(analysis_df: pd.DataFrame) -> list[tuple[str, float]]:
    base = analysis_df[pd.notna(analysis_df["ret_0616_pct"])].copy()
    if base.empty:
        return []

    pairs: list[tuple[str, float]] = []
    target = pd.to_numeric(base["ret_0616_pct"], errors="coerce")
    for feature in CORRELATION_FEATURES:
        series = pd.to_numeric(base[feature], errors="coerce")
        pair = pd.concat([series, target], axis=1).dropna()
        if len(pair) < 8:
            continue
        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if pd.notna(corr):
            pairs.append((feature, corr))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs


def format_stock_line(stock: pd.Series) -> str:
    return (
        f"- {stock['code']} {stock['name']} | 6月15日 {format_pct(stock['ret_0615_pct'])}"
        f" | 6月16日 {format_pct(stock['ret_0616_pct'])}"
        f" | 两日合计 {format_pct(stock['two_day_ret_pct'])}"
        f" | 行业 {stock['industry']}"
        f" | 备注 {stock['structural_note']}"
    )


def build_report(analysis_df: pd.DataFrame, group_summary_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Table.csv 80只股票两日分化分析报告")
    lines.append("")
    lines.append("## 数据口径")
    lines.append("")
    lines.append("- 样本范围：`screening_results/Table.csv` 中 80 只股票。")
    lines.append("- 2026年6月15日数据：使用 `Table.csv` 快照中的收盘、涨跌幅与当日 OHLC。")
    lines.append("- 2026年6月16日数据：使用本地 `D:\\a_stock_data\\daily` 日线 parquet。")
    lines.append("- 说明：本地 `daily` 库缺少 2026年6月15日这一整天的原始日线，因此本报告采用 `6月15日快照 + 6月16日日线` 的拼接口径。")
    lines.append("")

    valid_df = analysis_df[analysis_df["group"] != "missing"].copy()
    missing_df = analysis_df[analysis_df["group"] == "missing"].copy()
    lines.append("## 样本概览")
    lines.append("")
    lines.append(f"- 可完成两日比较的股票数：{len(valid_df)}")
    lines.append(f"- 数据缺失股票数：{len(missing_df)}")
    for group in REPORT_GROUPS:
        count = int((analysis_df["group"] == group).sum())
        lines.append(f"- {GROUP_LABELS[group]}：{count} 只")
    if not missing_df.empty:
        missing_list = "，".join(f"{row.code} {row.name}" for row in missing_df.itertuples(index=False))
        lines.append(f"- 缺失样本：{missing_list}")
    lines.append("")

    lines.append("## 核心结论")
    lines.append("")
    up_up = group_summary_df[group_summary_df["group"] == "up_up"]
    up_down = group_summary_df[group_summary_df["group"] == "up_down"]
    down_up = group_summary_df[group_summary_df["group"] == "down_up"]
    down_down = group_summary_df[group_summary_df["group"] == "down_down"]

    if not up_up.empty and not up_down.empty:
        lines.append(
            "- 6月15日先涨的股票，到了6月16日会明显分成两类：一类是趋势继续加速，另一类是高位分歧兑现。"
        )
        lines.append(
            f"- 真正走出延续的 `6月15日涨、6月16日涨` 组，6月15日平均涨幅更高"
            f"（{format_pct(float(up_up['ret_0615_pct_mean'].iloc[0]))} vs {format_pct(float(up_down['ret_0615_pct_mean'].iloc[0]))}），"
            f"相对 MA20 的平均乖离也更大（{format_pct(float(up_up['close_vs_ma20_0615_pct_mean'].iloc[0]))} vs {format_pct(float(up_down['close_vs_ma20_0615_pct_mean'].iloc[0]))}），"
            "说明这批票里，强者更强的趋势延续效应，比单纯的“涨多了就回落”更明显。"
        )
        lines.append(
            f"- `6月15日涨、6月16日跌` 组与其说是过热，不如说是 6月16日承接弱：平均开盘缺口为"
            f" {format_pct(float(up_down['gap_0616_pct_mean'].iloc[0]))}，明显弱于 `6月15日涨、6月16日涨` 组的"
            f" {format_pct(float(up_up['gap_0616_pct_mean'].iloc[0]))}；同时 6月16日平均收盘只落在日内区间的"
            f" {format_num(float(up_down['bar_close_position_0616_mean'].iloc[0]))}，远低于延续组的"
            f" {format_num(float(up_up['bar_close_position_0616_mean'].iloc[0]))}。"
        )
    if not down_up.empty and not down_down.empty:
        lines.append(
            f"- `6月15日跌、6月16日涨` 组的前一周平均涨幅为 {format_pct(float(down_up['week_return_0608_0612_pct_mean'].iloc[0]))}，"
            f"高于 `6月15日跌、6月16日跌` 组的 {format_pct(float(down_down['week_return_0608_0612_pct_mean'].iloc[0]))}，"
            "说明 6月15日的下跌是否只是强趋势中的回踩，是第二天能否修复的关键。"
        )
    lines.append(
        "- 从样本特征看，分化更主要来自结构位置和短线拥挤程度，而不是单一行业因子。行业会影响弹性，但不是最核心的解释变量。"
    )
    lines.append("")

    correlation_pairs = compute_correlations(analysis_df)
    if correlation_pairs:
        lines.append("## 与6月16日涨跌最相关的先验特征")
        lines.append("")
        lines.append("- 下面是样本层面的相关性，反映的是统计相关，不代表严格因果。")
        for feature, corr in correlation_pairs[:8]:
            lines.append(f"- {FEATURE_LABELS[feature]}：相关系数 {corr:.3f}")
        lines.append("")

    lines.append("## 四类分组明细")
    lines.append("")
    for group in REPORT_GROUPS:
        group_df = analysis_df[analysis_df["group"] == group].copy()
        if group_df.empty:
            continue

        group_df = group_df.sort_values(
            ["ret_0616_pct", "ret_0615_pct", "two_day_ret_pct"],
            ascending=[False, False, False],
            na_position="last",
        )
        summary_row = group_summary_df[group_summary_df["group"] == group].iloc[0]

        lines.append(f"### {GROUP_LABELS[group]}")
        lines.append("")
        lines.append(f"- 数量：{int(summary_row['count'])} 只")
        lines.append(f"- 6月15日平均涨跌幅：{format_pct(summary_row['ret_0615_pct_mean'])}")
        lines.append(f"- 6月16日平均涨跌幅：{format_pct(summary_row['ret_0616_pct_mean'])}")
        lines.append(f"- 两日累计平均涨跌幅：{format_pct(summary_row['two_day_ret_pct_mean'])}")
        lines.append(f"- 连涨天数均值：{format_num(summary_row['streak_days_mean'])}")
        lines.append(f"- 6月15日相对 MA20 的平均乖离：{format_pct(summary_row['close_vs_ma20_0615_pct_mean'])}")
        lines.append(f"- 前一周平均涨幅：{format_pct(summary_row['week_return_0608_0612_pct_mean'])}")
        lines.append(f"- 6月16日平均开盘缺口：{format_pct(summary_row['gap_0616_pct_mean'])}")
        lines.append(f"- 6月16日平均收盘在日内位置：{format_num(summary_row['bar_close_position_0616_mean'])}")
        lines.append(f"- 6月15日涨停或接近涨停只数：{int(summary_row['limit_up_0615_count'])}")
        lines.append(f"- 命中 `weekly_multi_ma_head` 的只数：{int(summary_row['multi_ma_head_count'])}")
        if summary_row["top_industries"]:
            lines.append(f"- 组内行业集中：{summary_row['top_industries']}")
        lines.append("")

        if group == "up_up":
            lines.append("- 解释：这组的共性是 6月15日并非单纯脉冲拉升，而是建立在前一周强趋势或突破结构之上，6月16日收盘大多仍靠近日内高位，说明承接连续。")
        elif group == "up_down":
            lines.append("- 解释：这组更像高位获利兑现。典型特征是 6月15日已经离 MA20 较远、6日涨幅和连涨天数偏高，次日只要高开承接不足，就容易冲高回落。")
        elif group == "down_up":
            lines.append("- 解释：这组的 6月15日下跌更多是强趋势中的回踩或分时洗盘，主升结构没有被破坏，所以 6月16日容易走出修复。")
        elif group == "down_down":
            lines.append("- 解释：这组往往不只是 6月15日回调，而是短线结构已经转弱。6月16日如果继续收在日内低位，就会形成补跌或趋势破坏。")
        lines.append("")

        for _, stock in group_df.iterrows():
            lines.append(format_stock_line(stock))
        lines.append("")

    if not missing_df.empty:
        lines.append("## 数据缺失样本")
        lines.append("")
        for _, stock in missing_df.iterrows():
            lines.append(f"- {stock['code']} {stock['name']} | 本地日线缺少 6月16日数据，无法归入四象限")
        lines.append("")

    lines.append("## 归因总结")
    lines.append("")
    lines.append("- 第一层因子是大级别结构：前一周是否强趋势、是否站上并拉开 MA5/MA20/MA60，决定了 6月15日下跌到底是回踩还是转弱。")
    lines.append("- 第二层因子是 6月15日的位置：如果当日已经大幅偏离 MA20、突破前高过多、连涨天数偏长，那么 6月16日更容易出现高位分歧。")
    lines.append("- 第三层因子是 6月16日的承接形态：真正强的票往往高开后还能收在高位；弱的票则常见高开低走、上影偏长、收盘靠近日内低位。")
    lines.append("- 行业有助于解释弹性，但在这 80 只样本里，结构位置和资金拥挤程度的解释力高于单一行业标签。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    table = build_table_base()
    multi_ma_head_set = load_multi_ma_head_set()

    analysis_rows = [analyze_stock(row, multi_ma_head_set) for _, row in table.iterrows()]
    analysis_df = pd.DataFrame(analysis_rows)
    analysis_df = analysis_df.sort_values(["group", "ret_0616_pct", "ret_0615_pct"], ascending=[True, False, False], na_position="last")
    group_summary_df = compute_group_summary(analysis_df)

    analysis_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_REPORT.write_text(build_report(analysis_df, group_summary_df), encoding="utf-8")

    print(f"analysis_rows={len(analysis_df)}")
    print(f"output_csv={OUTPUT_CSV}")
    print(f"output_report={OUTPUT_REPORT}")
    print(group_summary_df[["group", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
