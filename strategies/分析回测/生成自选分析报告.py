from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STRATEGIES_ROOT))

from project_paths import SCREENING_RESULTS_DIR, ensure_project_dirs


DEFAULT_NOTICE_DAYS = 30
DEFAULT_PAGE_SIZE = 20
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)
JSONP_PATTERN = re.compile(r"^[^(]+\((.*)\)\s*;?\s*$", re.S)
DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

GENERIC_CONCEPT_KEYWORDS = [
    "融资融券",
    "沪股通",
    "深股通",
    "北交所",
    "富时罗素",
    "MSCI",
    "标普道琼斯",
    "中证",
    "上证",
    "深证",
    "创业板综",
    "机构重仓",
    "基金重仓",
    "社保重仓",
    "QFII重仓",
    "预盈预增",
    "连续亏损",
    "破净股",
    "标准普尔",
    "高股息",
    "低市净率",
    "中字头",
    "专精特新",
    "央国企",
    "中盘成长",
    "小盘成长",
    "微盘成长",
    "昨日连板",
    "昨日涨停",
    "昨日触板",
]

POSITIVE_NOTICE_KEYWORDS = [
    "业绩预增",
    "业绩快报",
    "业绩预告",
    "分配方案实施",
    "分配方案决议",
    "分红",
    "股东增持",
    "回购",
    "中标",
    "合同",
    "项目",
    "进展",
    "投资者关系活动记录",
    "调研活动",
    "获得补助",
    "战略合作",
    "股权激励",
    "募投",
    "现金管理到期赎回",
    "解除质押",
    "持股计划",
]

CAUTION_NOTICE_KEYWORDS = [
    "减持",
    "质押",
    "股票交易异常波动",
    "风险提示",
    "问询",
    "监管",
    "诉讼",
    "仲裁",
    "立案",
    "延期",
    "终止",
    "担保",
    "关联交易",
]


@dataclass
class StockReportRow:
    symbol: str
    name: str
    file_symbol: str
    api_code: str
    eastmoney_industry: str
    csrc_industry: str
    business_summary: str
    concepts: list[str]
    positive_items: list[str]
    caution_items: list[str]
    finance_report_name: str
    finance_notice_date: str
    revenue: float | None
    revenue_yoy: float | None
    parent_net_profit: float | None
    parent_net_profit_yoy: float | None
    gross_margin: float | None
    roe: float | None
    survey_url: str
    concept_url: str
    notice_url: str
    finance_url: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a markdown analysis report for a watchlist CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCREENING_RESULTS_DIR / "weekly_bullish_kong_turn_bull_2026-06-26_close.csv",
        help="Input screening CSV.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Analysis date in YYYY-MM-DD. Defaults to today in Asia/Shanghai.",
    )
    parser.add_argument(
        "--notice-days",
        type=int,
        default=DEFAULT_NOTICE_DAYS,
        help="Recent notice lookback window in days.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path.",
    )
    return parser


def normalize_as_of_date(text: str | None) -> pd.Timestamp:
    if not text:
        return pd.Timestamp.now(tz="Asia/Shanghai").normalize()

    normalized = text.strip()
    parsed = pd.Timestamp(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("Asia/Shanghai")
    else:
        parsed = parsed.tz_convert("Asia/Shanghai")

    if DATE_ONLY_PATTERN.fullmatch(normalized):
        parsed = parsed.normalize()

    return parsed.normalize()


def build_output_path(input_path: Path, as_of_date: pd.Timestamp, user_output: Path | None) -> Path:
    if user_output is not None:
        return user_output
    stem = input_path.stem
    return SCREENING_RESULTS_DIR / f"{stem}_analysis_{as_of_date.date().isoformat()}.md"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def parse_jsonp(text: str) -> dict:
    match = JSONP_PATTERN.match(text.strip())
    if not match:
        raise ValueError("Response is not valid JSONP")
    return json.loads(match.group(1))


def normalize_stock_api_code(file_symbol: str) -> str:
    text = str(file_symbol).strip().lower()
    if text.startswith("sh"):
        return f"SH{text[-6:]}"
    if text.startswith("sz"):
        return f"SZ{text[-6:]}"
    if text.startswith("bj"):
        return f"BJ{text[-6:]}"
    raise ValueError(f"Unsupported file_symbol: {file_symbol}")


def fetch_company_survey(session: requests.Session, api_code: str) -> dict:
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={api_code}"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_core_conception(session: requests.Session, api_code: str) -> dict:
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax?code={api_code}"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_announcements(session: requests.Session, symbol: str) -> list[dict]:
    params = {
        "cb": "jQuery_watchlist_report",
        "sr": "-1",
        "page_size": str(DEFAULT_PAGE_SIZE),
        "page_index": "1",
        "ann_type": "A",
        "client_source": "web",
        "stock_list": symbol,
    }
    response = session.get(
        "https://np-anotice-stock.eastmoney.com/api/security/ann",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = parse_jsonp(response.text)
    return payload.get("data", {}).get("list") or []


def fetch_finance_analysis(session: requests.Session, api_code: str) -> dict:
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew?type=0&code={api_code}"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8-sig"))


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def clean_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        value = text

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None
    return number


def format_amount_yi(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value / 100000000:.2f}亿元"


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def pick_business_summary(core_data: dict) -> str:
    for item in core_data.get("hxtc") or []:
        if item.get("KEY_CLASSIF") == "主营业务":
            keyword = clean_text(item.get("KEYWORD"))
            content = clean_text(item.get("MAINPOINT_CONTENT"))
            if keyword and content:
                first_sentence = content.split("。", 1)[0].strip()
                return f"{keyword}。{first_sentence}。"
            return keyword or content
    return ""


def is_generic_concept(board_name: str) -> bool:
    for keyword in GENERIC_CONCEPT_KEYWORDS:
        if keyword in board_name:
            return True
    if board_name.endswith("板块") and any(
        board_name.startswith(prefix)
        for prefix in ["北京", "上海", "广东", "陕西", "浙江", "江苏", "湖北", "四川", "福建"]
    ):
        return True
    return False


def extract_concepts(core_data: dict, limit: int = 6) -> list[str]:
    concepts: list[str] = []
    for item in core_data.get("ssbk") or []:
        board_name = clean_text(item.get("BOARD_NAME"))
        if not board_name or is_generic_concept(board_name):
            continue
        if board_name in concepts:
            continue
        concepts.append(board_name)
        if len(concepts) >= limit:
            break
    return concepts


def within_notice_window(notice_date: pd.Timestamp, as_of_date: pd.Timestamp, notice_days: int) -> bool:
    window_start = as_of_date - pd.Timedelta(days=notice_days)
    return window_start <= notice_date <= as_of_date


def classify_notice_strength(column_name: str, title: str) -> str:
    text = f"{column_name} {title}"
    if any(keyword in text for keyword in CAUTION_NOTICE_KEYWORDS):
        return "caution"
    if any(keyword in text for keyword in POSITIVE_NOTICE_KEYWORDS):
        return "positive"
    return "neutral"


def collect_notice_items(
    notices: list[dict],
    as_of_date: pd.Timestamp,
    notice_days: int,
) -> tuple[list[str], list[str]]:
    positive_items: list[str] = []
    caution_items: list[str] = []
    fallback_items: list[str] = []

    for item in notices:
        notice_date = pd.Timestamp(item.get("notice_date")).tz_localize(None).normalize()
        if not within_notice_window(notice_date, as_of_date.tz_localize(None), notice_days):
            continue

        columns = item.get("columns") or []
        column_name = clean_text(columns[0].get("column_name")) if columns else ""
        title = clean_text(item.get("title_ch") or item.get("title"))
        label = f"{notice_date.date().isoformat()} {column_name}：{title}"
        strength = classify_notice_strength(column_name, title)

        if strength == "positive" and label not in positive_items:
            positive_items.append(label)
        elif strength == "caution" and label not in caution_items:
            caution_items.append(label)
        elif label not in fallback_items:
            fallback_items.append(label)

    if not positive_items and fallback_items:
        positive_items = fallback_items[:2]

    return positive_items[:3], caution_items[:2]


def summarize_industry_distribution(rows: list[StockReportRow]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.eastmoney_industry or row.csrc_industry or "未识别"
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{industry}({count})" for industry, count in ordered[:8]]


def summarize_concept_distribution(rows: list[StockReportRow]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        for concept in row.concepts:
            counts[concept] = counts.get(concept, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{concept}({count})" for concept, count in ordered[:12]]


def build_source_urls(api_code: str, symbol: str) -> tuple[str, str, str, str]:
    survey_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/PageAjax?code={api_code}"
    concept_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax?code={api_code}"
    notice_url = (
        "https://np-anotice-stock.eastmoney.com/api/security/ann"
        f"?cb=jQuery_watchlist_report&sr=-1&page_size={DEFAULT_PAGE_SIZE}"
        f"&page_index=1&ann_type=A&client_source=web&stock_list={symbol}"
    )
    finance_url = f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew?type=0&code={api_code}"
    return survey_url, concept_url, notice_url, finance_url


def build_stock_report_row(
    session: requests.Session,
    symbol: str,
    name: str,
    file_symbol: str,
    as_of_date: pd.Timestamp,
    notice_days: int,
) -> StockReportRow:
    api_code = normalize_stock_api_code(file_symbol)
    survey_data = fetch_company_survey(session, api_code)
    core_data = fetch_core_conception(session, api_code)
    notices = fetch_announcements(session, symbol)
    finance_data = fetch_finance_analysis(session, api_code)

    survey_row = (survey_data.get("jbzl") or [{}])[0]
    finance_row = (finance_data.get("data") or [{}])[0]
    eastmoney_industry = clean_text(survey_row.get("EM2016"))
    csrc_industry = clean_text(survey_row.get("INDUSTRYCSRC1"))
    business_summary = pick_business_summary(core_data)
    concepts = extract_concepts(core_data)
    positive_items, caution_items = collect_notice_items(notices, as_of_date, notice_days)
    survey_url, concept_url, notice_url, finance_url = build_source_urls(api_code, symbol)

    return StockReportRow(
        symbol=symbol,
        name=name,
        file_symbol=file_symbol,
        api_code=api_code,
        eastmoney_industry=eastmoney_industry,
        csrc_industry=csrc_industry,
        business_summary=business_summary,
        concepts=concepts,
        positive_items=positive_items,
        caution_items=caution_items,
        finance_report_name=clean_text(finance_row.get("REPORT_DATE_NAME")),
        finance_notice_date=clean_text(finance_row.get("NOTICE_DATE")),
        revenue=clean_float(finance_row.get("TOTALOPERATEREVE")),
        revenue_yoy=clean_float(finance_row.get("TOTALOPERATEREVETZ")),
        parent_net_profit=clean_float(finance_row.get("PARENTNETPROFIT")),
        parent_net_profit_yoy=clean_float(finance_row.get("PARENTNETPROFITTZ")),
        gross_margin=clean_float(finance_row.get("XSMLL")),
        roe=clean_float(finance_row.get("ROEJQ")),
        survey_url=survey_url,
        concept_url=concept_url,
        notice_url=notice_url,
        finance_url=finance_url,
    )


def render_report(
    rows: list[StockReportRow],
    input_path: Path,
    as_of_date: pd.Timestamp,
    notice_days: int,
) -> str:
    window_start = (as_of_date - pd.Timedelta(days=notice_days)).date().isoformat()
    window_end = as_of_date.date().isoformat()
    industry_summary = "、".join(summarize_industry_distribution(rows))
    concept_summary = "、".join(summarize_concept_distribution(rows))

    lines: list[str] = []
    lines.append(f"# {input_path.stem} 行业/概念/近期催化/业绩分析")
    lines.append("")
    lines.append(f"- 分析对象来源：`{input_path.name}`")
    lines.append(f"- 统计口径：公告窗口 `{window_start}` 至 `{window_end}`，行业/概念快照按 `{window_end}` 拉取")
    lines.append("- 数据说明：")
    lines.append("  行业来自东方财富公司资料页字段 `EM2016` 与证监会行业字段 `INDUSTRYCSRC1`。")
    lines.append("  概念来自东方财富核心题材/所属板块标签，已剔除部分纯风格、通道与地域标签。")
    lines.append("  “近期利好”指近 30 天正式公告中偏正面的潜在催化；若未见明确利好型公告，则回退展示最近的中性经营类公告。")
    lines.append("  “业绩”使用东方财富最近一期已披露财报口径，展示营收、归母净利润、同比、毛利率和 ROE。")
    lines.append("  若近 30 天出现减持、异常波动、担保等事项，会单独列在“需留意”。")
    lines.append("")
    lines.append("## 组合概览")
    lines.append("")
    lines.append(f"- 行业分布：{industry_summary}")
    lines.append(f"- 高频概念：{concept_summary}")
    lines.append("")
    lines.append("## 个股明细")
    lines.append("")

    for row in rows:
        lines.append(f"### {row.symbol} {row.name}")
        lines.append("")
        lines.append(f"- 行业：{row.eastmoney_industry or '未识别'}")
        if row.csrc_industry:
            lines.append(f"- 证监会行业：{row.csrc_industry}")
        if row.business_summary:
            lines.append(f"- 主营主线：{row.business_summary}")
        if row.concepts:
            lines.append(f"- 主要概念：{'、'.join(row.concepts)}")
        else:
            lines.append("- 主要概念：未识别到较强业务相关板块标签")

        if row.finance_report_name or row.revenue is not None or row.parent_net_profit is not None:
            finance_notice = row.finance_notice_date[:10] if row.finance_notice_date else "-"
            finance_items = [
                f"报告期 {row.finance_report_name or '-'}",
                f"披露日 {finance_notice}",
                f"营收 {format_amount_yi(row.revenue)}",
                f"营收同比 {format_pct(row.revenue_yoy)}",
                f"归母净利润 {format_amount_yi(row.parent_net_profit)}",
                f"归母同比 {format_pct(row.parent_net_profit_yoy)}",
                f"毛利率 {format_pct(row.gross_margin)}",
                f"ROE {format_pct(row.roe)}",
            ]
            lines.append(f"- 最近业绩：{'；'.join(finance_items)}")

        if row.positive_items:
            lines.append("- 近30天潜在正面催化：")
            for item in row.positive_items:
                lines.append(f"  - {item}")
        else:
            lines.append("- 近30天潜在正面催化：未见明确利好型正式公告")

        if row.caution_items:
            lines.append("- 需留意：")
            for item in row.caution_items:
                lines.append(f"  - {item}")

        lines.append(
            f"- 数据源：[公司资料]({row.survey_url}) | "
            f"[概念板块]({row.concept_url}) | "
            f"[最近公告]({row.notice_url}) | "
            f"[财务分析]({row.finance_url})"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ensure_project_dirs()
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    as_of_date = normalize_as_of_date(args.as_of_date)
    output_path = build_output_path(
        input_path=input_path,
        as_of_date=as_of_date,
        user_output=args.output.expanduser().resolve() if args.output else None,
    )

    df = pd.read_csv(input_path, dtype={"symbol": str, "file_symbol": str})
    required_columns = {"symbol", "name", "file_symbol"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    session = create_session()
    rows: list[StockReportRow] = []
    for _, item in df.iterrows():
        rows.append(
            build_stock_report_row(
                session=session,
                symbol=clean_text(item["symbol"]),
                name=clean_text(item["name"]),
                file_symbol=clean_text(item["file_symbol"]),
                as_of_date=as_of_date,
                notice_days=args.notice_days,
            )
        )

    report_text = render_report(
        rows=rows,
        input_path=input_path,
        as_of_date=as_of_date,
        notice_days=args.notice_days,
    )
    output_path.write_text(report_text, encoding="utf-8")

    print(f"report_path={output_path}")
    print(f"stock_count={len(rows)}")
    print(f"as_of_date={as_of_date.date().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
