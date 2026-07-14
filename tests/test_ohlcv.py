"""Validation tests for the engine's market-data boundary."""

import numpy as np
import pytest

from data.ohlcv import Bars


def bars() -> Bars:
    return Bars(
        "TEST", "1m", np.array([1, 2], dtype=np.int64),
        np.array([100, 101], dtype=np.float32),
        np.array([102, 103], dtype=np.float32),
        np.array([99, 100], dtype=np.float32),
        np.array([101, 102], dtype=np.float32),
        np.array([10, 20], dtype=np.float32),
    )


def test_valid_bars():
    bars().validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("close", np.nan, "non-finite"),
        ("open", np.inf, "non-finite"),
        ("low", 0.0, "positive"),
        ("high", 98.0, "below open or close"),
        ("low", 102.0, "above open or close"),
        ("volume", -1.0, "negative"),
    ],
)
def test_rejects_invalid_values(field, value, message):
    data = bars()
    getattr(data, field)[0] = value
    with pytest.raises(ValueError, match=message):
        data.validate()


def test_rejects_bad_timestamps_and_empty_data():
    data = bars()
    data.ts[1] = data.ts[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        data.validate()

    empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        Bars("TEST", "1m", np.array([], dtype=np.int64),
             empty, empty, empty, empty, empty).validate()
