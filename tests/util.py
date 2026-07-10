"""Shared test helpers: synthetic bar construction."""

from __future__ import annotations

import numpy as np

from data.ohlcv import Bars


def make_bars(close, open_=None, high=None, low=None, volume=None) -> Bars:
    c = np.asarray(close, dtype=np.float32)
    n = len(c)
    o = np.asarray(open_, dtype=np.float32) if open_ is not None else c.copy()
    h = np.asarray(high, dtype=np.float32) if high is not None else np.maximum(o, c)
    l = np.asarray(low, dtype=np.float32) if low is not None else np.minimum(o, c)
    v = np.asarray(volume, dtype=np.float32) if volume is not None else np.ones(n, np.float32)
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000
    return Bars("TEST", "1m", ts, o, h, l, c, v)


def random_walk_bars(n: int, seed: int = 0, start: float = 100.0,
                     vol: float = 0.002, drift: float = 0.0) -> Bars:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n).astype(np.float64)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(0.5, 2.0, n)
    return make_bars(close, open_, high, low, volume)
