"""Fetch 1m OHLCV history from the Binance public archive (data.binance.vision).

Monthly zip files are cached in data/raw/ so re-runs only download what is
missing, then everything is consolidated into one .npz per symbol via ohlcv.py.

Usage:
    python data/fetch_binance.py --symbol BTCUSDT --start 2024-01 --end 2026-06
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ohlcv import Bars, save_bars  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
BASE = "https://data.binance.vision/data/spot/monthly/klines"


def month_range(start: str, end: str) -> list[str]:
    """['2024-01', ..., '2026-06'] inclusive."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def download_month(symbol: str, interval: str, month: str, session: requests.Session) -> str | None:
    """Download one monthly zip into RAW_DIR (skip if present). None if 404."""
    os.makedirs(RAW_DIR, exist_ok=True)
    name = f"{symbol}-{interval}-{month}.zip"
    dest = os.path.join(RAW_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = f"{BASE}/{symbol}/{interval}/{name}"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with open(dest + ".part", "wb") as f:
        f.write(r.content)
    os.replace(dest + ".part", dest)
    return dest


def parse_zip(path: str) -> np.ndarray:
    """Parse one monthly kline zip -> (N, 6) float64 array [ts_ms, o, h, l, c, v]."""
    rows = []
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f, "utf-8")):
                if not row or not row[0].strip().isdigit():
                    continue  # header line in newer files
                ts = int(row[0])
                if ts > 10**14:  # 2025+ archives use microseconds
                    ts //= 1000
                rows.append((ts, float(row[1]), float(row[2]), float(row[3]),
                             float(row[4]), float(row[5])))
    return np.array(rows, dtype=np.float64)


def fetch(symbol: str, interval: str, start: str, end: str) -> Bars:
    session = requests.Session()
    chunks = []
    months = month_range(start, end)
    for i, month in enumerate(months):
        path = download_month(symbol, interval, month, session)
        if path is None:
            print(f"  {month}: not on archive (skipped)")
            continue
        arr = parse_zip(path)
        chunks.append(arr)
        print(f"  {month}: {len(arr)} bars  [{i + 1}/{len(months)}]")
    if not chunks:
        raise RuntimeError(f"no data downloaded for {symbol}")
    all_ = np.concatenate(chunks)
    order = np.argsort(all_[:, 0], kind="stable")
    all_ = all_[order]
    # drop duplicate timestamps if any month boundaries overlap
    keep = np.concatenate([[True], np.diff(all_[:, 0]) > 0])
    all_ = all_[keep]
    return Bars(
        symbol, interval,
        all_[:, 0].astype(np.int64),
        *(all_[:, i].astype(np.float32) for i in range(1, 6)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", default="2024-01", help="YYYY-MM")
    ap.add_argument("--end", default="2026-06", help="YYYY-MM inclusive")
    args = ap.parse_args()

    print(f"fetching {args.symbol} {args.interval} {args.start}..{args.end}")
    bars = fetch(args.symbol, args.interval, args.start, args.end)
    path = save_bars(bars)
    print(f"saved {len(bars)} bars -> {path}")


if __name__ == "__main__":
    main()
