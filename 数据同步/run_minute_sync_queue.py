from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
WATCHLISTS_DIR = SCRIPT_DIR / "watchlists"
DEFAULT_CONFIG = SCRIPT_DIR / "market_store_manual_sync.json"
DEFAULT_STOCKS = WATCHLISTS_DIR / "stock_universe.csv"
DEFAULT_STATE = LOGS_DIR / "minute_queue_state.csv"
DEFAULT_TARGET_DATE = time.strftime("%Y-%m-%d")


def load_symbols(csv_path: Path) -> list[str]:
    frame = pd.read_csv(csv_path, dtype={"symbol": str}).fillna("")
    markets = frame["market"].astype(str).str.strip().str.lower()
    symbols = frame["symbol"].astype(str).str.strip()
    return [f"{market}{symbol}" for market, symbol in zip(markets, symbols) if market and symbol]


def normalize_target_date(value: str | None) -> str:
    text = str(value or "").strip()
    return text[:10] if text else ""


def parse_symbol_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = {str(item).strip().lower() for item in str(value).split(",") if str(item).strip()}
    return items or None


def load_latest_rows(state_path: Path, periods: str, target_date: str) -> dict[str, dict[str, str]]:
    if not state_path.exists():
        return {}
    frame = pd.read_csv(state_path, dtype=str).fillna("")
    if "symbol" not in frame.columns or "status" not in frame.columns or "periods" not in frame.columns:
        return {}

    scoped = frame.loc[frame["periods"].astype(str) == periods].copy()
    if "logged_at" in scoped.columns and target_date:
        # Treat target_date as a floor so overnight runs can resume/retry without
        # discarding progress written after midnight.
        scoped = scoped.loc[scoped["logged_at"].astype(str).str.slice(0, 10) >= target_date]
    if scoped.empty:
        return {}

    latest_by_symbol: dict[str, dict[str, str]] = {}
    for row in scoped.to_dict(orient="records"):
        symbol = str(row.get("symbol", "")).strip()
        if symbol:
            latest_by_symbol[symbol] = {key: str(value) for key, value in row.items()}
    return latest_by_symbol


def load_completed(state_path: Path, periods: str, target_date: str) -> set[str]:
    latest_rows = load_latest_rows(state_path=state_path, periods=periods, target_date=target_date)
    completed: set[str] = set()
    for symbol, row in latest_rows.items():
        if row.get("status", "").strip() == "ok":
            completed.add(f"{symbol}|{periods}")
    return completed


STATE_COLUMNS = ["symbol", "periods", "status", "detail", "logged_at"]
STATE_WRITE_LOCK = Lock()


def append_state(state_path: Path, symbol: str, periods: str, status: str, detail: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "symbol": symbol,
        "periods": periods,
        "status": status,
        "detail": detail,
        "logged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with STATE_WRITE_LOCK:
        file_exists = state_path.exists()
        with state_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=STATE_COLUMNS)
            if not file_exists or state_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)


def run_one_symbol(
    config_path: Path,
    symbol: str,
    periods: str,
    timeout_seconds: int,
    start_date: str,
    end_date: str,
) -> tuple[bool, str]:
    command = [
        sys.executable,
        "sync_market_data_store.py",
        str(config_path),
        "--periods",
        periods,
        "--stocks",
        symbol,
        "--skip-store-metadata",
        "--skip-concepts",
    ]
    if start_date:
        command.extend(["--start-date", start_date])
    if end_date:
        command.extend(["--end-date", end_date])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        return False, f"timeout after {elapsed:.1f}s"

    elapsed = time.perf_counter() - started
    stdout_text = completed.stdout.strip().splitlines()
    stderr_text = completed.stderr.strip().splitlines()
    tail = stdout_text[-1] if stdout_text else (stderr_text[-1] if stderr_text else "")
    detail = f"code={completed.returncode} elapsed={elapsed:.1f}s tail={tail}"
    return completed.returncode == 0, detail


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minute sync one symbol at a time with timeout protection.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stocks-file", default=str(DEFAULT_STOCKS))
    parser.add_argument("--stocks", default="", help="Comma-separated full symbols to process, e.g. sh600519,sz000001")
    parser.add_argument("--periods", default="60m,30m")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE))
    parser.add_argument("--target-date", default=DEFAULT_TARGET_DATE, help="Only treat state rows from this date onward as completed, format YYYY-MM-DD")
    parser.add_argument("--start-from-symbol", default="", help="Resume from this full symbol, e.g. sz301308")
    parser.add_argument("--start-date", default="", help="Optional inclusive sync start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="Optional inclusive sync end date, format YYYY-MM-DD")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Only process at most this many pending symbols; 0 means all")
    parser.add_argument("--workers", type=int, default=1, help="How many symbols to process in parallel")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config_path = Path(args.config).resolve()
    stocks_path = Path(args.stocks_file).resolve()
    state_path = Path(args.state_file).resolve()
    target_date = normalize_target_date(args.target_date) or DEFAULT_TARGET_DATE

    symbols = load_symbols(stocks_path)
    selected_symbols = parse_symbol_arg(args.stocks)
    if selected_symbols:
        symbols = [symbol for symbol in symbols if symbol.lower() in selected_symbols]
    start_from_symbol = str(args.start_from_symbol or "").strip().lower()
    if start_from_symbol:
        try:
            start_index = next(index for index, symbol in enumerate(symbols) if symbol.lower() == start_from_symbol)
        except StopIteration:
            print(f"start_from_symbol_not_found={args.start_from_symbol}", file=sys.stderr, flush=True)
            return 1
        symbols = symbols[start_index:]

    completed = load_completed(state_path=state_path, periods=args.periods, target_date=target_date)
    pending = [symbol for symbol in symbols if f"{symbol}|{args.periods}" not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    worker_count = max(1, min(int(args.workers), len(pending) if pending else 1))

    print(f"target_date={target_date}", flush=True)
    if start_from_symbol:
        print(f"start_from_symbol={symbols[0]}", flush=True)
    print(f"total_symbols={len(symbols)}", flush=True)
    print(f"completed_symbols={len(completed)}", flush=True)
    print(f"pending_symbols={len(pending)}", flush=True)
    print(f"workers={worker_count}", flush=True)

    if not pending:
        return 0

    if worker_count == 1:
        for index, symbol in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] syncing {symbol} ...", flush=True)
            ok, detail = run_one_symbol(
                config_path=config_path,
                symbol=symbol,
                periods=args.periods,
                timeout_seconds=args.timeout_seconds,
                start_date=str(args.start_date or "").strip(),
                end_date=str(args.end_date or "").strip(),
            )
            status = "ok" if ok else "failed"
            append_state(state_path=state_path, symbol=symbol, periods=args.periods, status=status, detail=detail)
            print(f"[{index}/{len(pending)}] {symbol} {status}: {detail}", flush=True)
            time.sleep(args.sleep_seconds)
        return 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_symbol = {
            executor.submit(
                run_one_symbol,
                config_path,
                symbol,
                args.periods,
                args.timeout_seconds,
                str(args.start_date or "").strip(),
                str(args.end_date or "").strip(),
            ): symbol
            for symbol in pending
        }
        for index, future in enumerate(as_completed(future_to_symbol), start=1):
            symbol = future_to_symbol[future]
            try:
                ok, detail = future.result()
            except Exception as exc:
                ok = False
                detail = f"worker_exception={exc}"
            status = "ok" if ok else "failed"
            append_state(state_path=state_path, symbol=symbol, periods=args.periods, status=status, detail=detail)
            print(f"[{index}/{len(pending)}] {symbol} {status}: {detail}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
