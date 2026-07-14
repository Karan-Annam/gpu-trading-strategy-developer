"""Input-boundary checks shared by the native engine wrappers."""

import numpy as np
import pytest

from dsl.compiler import compile_source
from engine import cpu_engine, gpu_engine
from tests.util import make_bars


PROGRAM = compile_source(
    "param n = 3 in [2, 5]\n"
    "enter_long when close > sma(close, n)\n"
)


def bars():
    return make_bars(np.linspace(100.0, 101.0, 20, dtype=np.float32))


@pytest.mark.parametrize("engine", [cpu_engine, gpu_engine])
def test_rejects_wrong_parameter_width(engine):
    if not engine.available():
        pytest.skip(f"{engine.__name__} is unavailable")
    with pytest.raises(ValueError, match="shape"):
        engine.run_batch(PROGRAM, bars(), np.zeros((2, 0), np.float32))


@pytest.mark.parametrize("engine", [cpu_engine, gpu_engine])
def test_rejects_empty_parameter_batch(engine):
    if not engine.available():
        pytest.skip(f"{engine.__name__} is unavailable")
    with pytest.raises(ValueError, match="at least one"):
        engine.run_batch(PROGRAM, bars(), np.zeros((0, 1), np.float32))


def test_rejects_bad_single_parameter_count():
    if not cpu_engine.available():
        pytest.skip("CPU engine is unavailable")
    with pytest.raises(ValueError, match="1 values"):
        cpu_engine.run(PROGRAM, bars(), [])


def test_rejects_mismatched_bar_lengths():
    if not cpu_engine.available():
        pytest.skip("CPU engine is unavailable")
    data = bars()
    data.open = data.open[:-1]
    with pytest.raises(ValueError, match="length-20"):
        cpu_engine.run_batch(PROGRAM, data, np.zeros((1, 1), np.float32))
