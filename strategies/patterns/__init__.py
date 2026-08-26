# 作用：作为 patterns 包的统一导出入口。
# 内容：集中暴露 Bullish Up 与 Dow Pullback 相关的配置类、识别函数和扫描接口，方便示例脚本直接导入使用。
"""Reusable candlestick pattern modules."""

from .bullish_up import (
    BullishUpConfig,
    detect_bullish_up,
    filter_bullish_up_signals,
    load_stock_bars,
    scan_period_for_bullish_up,
)
from .dow_pullback import (
    DowPullbackConfig,
    analyze_dow_pullback,
    detect_dow_pullback,
    detect_latest_dow_pullback,
    evaluate_latest_dow_pullback,
    filter_dow_pullback_signals,
    get_latest_confirmed_swings,
    scan_period_for_dow_pullback,
    scan_period_for_latest_dow_pullback,
)
from .kong_turn_bull import (
    KongTurnBullConfig,
    analyze_kong_turn_bull,
    detect_kong_turn_bull,
    detect_latest_kong_turn_bull,
    evaluate_latest_kong_turn_bull,
    filter_kong_turn_bull_signals,
    scan_period_for_kong_turn_bull,
    scan_period_for_latest_kong_turn_bull,
)
from .strong_weak_strong import (
    StrongWeakStrongConfig,
    analyze_strong_weak_strong,
    detect_latest_strong_weak_strong,
    detect_strong_weak_strong,
    evaluate_latest_strong_weak_strong,
    filter_strong_weak_strong_signals,
    scan_period_for_latest_strong_weak_strong,
    scan_period_for_strong_weak_strong,
)

__all__ = [
    "BullishUpConfig",
    "DowPullbackConfig",
    "KongTurnBullConfig",
    "StrongWeakStrongConfig",
    "analyze_kong_turn_bull",
    "analyze_strong_weak_strong",
    "analyze_dow_pullback",
    "detect_bullish_up",
    "detect_dow_pullback",
    "detect_kong_turn_bull",
    "detect_latest_dow_pullback",
    "detect_latest_kong_turn_bull",
    "detect_latest_strong_weak_strong",
    "detect_strong_weak_strong",
    "evaluate_latest_dow_pullback",
    "evaluate_latest_kong_turn_bull",
    "evaluate_latest_strong_weak_strong",
    "filter_bullish_up_signals",
    "filter_dow_pullback_signals",
    "filter_kong_turn_bull_signals",
    "filter_strong_weak_strong_signals",
    "get_latest_confirmed_swings",
    "load_stock_bars",
    "scan_period_for_bullish_up",
    "scan_period_for_dow_pullback",
    "scan_period_for_latest_dow_pullback",
    "scan_period_for_kong_turn_bull",
    "scan_period_for_latest_kong_turn_bull",
    "scan_period_for_latest_strong_weak_strong",
    "scan_period_for_strong_weak_strong",
]
