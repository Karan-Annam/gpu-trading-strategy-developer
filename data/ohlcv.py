"""OHLCV bar cache: structure-of-arrays float32 columns + int64 ms timestamps.

Storage is one .npz per (symbol, interval) under data/cache/. Engines consume
the raw numpy arrays directly (contiguous float32), so this is the only place
that knows about files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class Bars:
    symbol: str
    interval: str
    ts: np.ndarray      # int64, epoch milliseconds, strictly increasing
    open: np.ndarray    # float32
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def slice(self, start: int, stop: int) -> "Bars":
        return Bars(
            self.symbol, self.interval, self.ts[start:stop],
            self.open[start:stop], self.high[start:stop], self.low[start:stop],
            self.close[start:stop], self.volume[start:stop],
        )

    def validate(self) -> None:
        n = len(self.ts)
        for c in COLUMNS:
            a = getattr(self, c)
            assert a.dtype == np.float32 and len(a) == n, c
        assert self.ts.dtype == np.int64
        if n > 1:
            assert bool(np.all(np.diff(self.ts) > 0)), "timestamps not strictly increasing"
        assert bool(np.all(self.high >= self.low)), "high < low"
        assert bool(np.all(self.volume >= 0)), "negative volume"


def cache_path(symbol: str, interval: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}.npz")


def save_bars(bars: Bars) -> str:
    bars.validate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(bars.symbol, bars.interval)
    tmp = path + ".tmp"
    np.savez_compressed(
        tmp, ts=bars.ts,
        **{c: getattr(bars, c) for c in COLUMNS},
    )
    # np.savez appends .npz to the tmp name
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    return path


def load_bars(symbol: str, interval: str = "1m") -> Bars:
    path = cache_path(symbol, interval)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached bars for {symbol} {interval}. "
            f"Run: python data/fetch_binance.py --symbol {symbol} --interval {interval}"
        )
    with np.load(path) as z:
        bars = Bars(
            symbol, interval, z["ts"].astype(np.int64),
            *(np.ascontiguousarray(z[c], dtype=np.float32) for c in COLUMNS),
        )
    bars.validate()
    return bars


def list_cached() -> list[dict]:
    """Enumerate cached datasets with basic metadata (for the Lab UI)."""
    out = []
    if not os.path.isdir(CACHE_DIR):
        return out
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".npz"):
            continue
        symbol, interval = fn[:-4].rsplit("_", 1)
        with np.load(os.path.join(CACHE_DIR, fn)) as z:
            ts = z["ts"]
            out.append({
                "symbol": symbol, "interval": interval, "bars": int(len(ts)),
                "start_ms": int(ts[0]), "end_ms": int(ts[-1]),
            })
    return out
