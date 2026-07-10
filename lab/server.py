"""Strategy Lab backend.

Run from the repo root:
    python -m uvicorn lab.server:app --port 8321
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ohlcv import list_cached, load_bars          # noqa: E402
from dsl import refengine                              # noqa: E402
from dsl.compiler import compile_source                # noqa: E402
from dsl.errors import CompileError                    # noqa: E402
import threading                                       # noqa: E402

from engine import cpu_engine, gpu_engine              # noqa: E402
from evolve import genome, llm                         # noqa: E402
from evolve import sweep as sweeplib                   # noqa: E402
from evolve.loop import EvolutionRun, Individual       # noqa: E402

app = FastAPI(title="Strategy Lab")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")
SEEDS = os.path.join(ROOT, "strategies")

REF_ENGINE_BAR_CAP = 60_000   # pure python is ~15k bars/s; keep the UI responsive

_bars_cache: dict[tuple[str, str], object] = {}


def get_bars(symbol: str, interval: str):
    key = (symbol, interval)
    if key not in _bars_cache:
        _bars_cache[key] = load_bars(symbol, interval)
    return _bars_cache[key]


# ---------- models ----------

class CompileReq(BaseModel):
    source: str


class RunReq(BaseModel):
    source: str
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    last_bars: int = 43_200          # default: last 30 days of 1m bars
    params: dict[str, float] = {}
    engine: str = "auto"             # auto | cpu | ref


class SaveReq(BaseModel):
    name: str
    source: str


class AxisSpec(BaseModel):
    name: str
    steps: int = 25


class SweepReq(BaseModel):
    source: str
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    last_bars: int = 129_600
    x: AxisSpec
    y: AxisSpec | None = None
    fixed: dict[str, float] = {}
    metric: str = "sharpe"
    n_folds: int = 0             # 0 = single period, >1 = mean OOS across folds


# ---------- helpers ----------

def compile_info(source: str) -> dict:
    try:
        prog = compile_source(source)
    except CompileError as e:
        return {"ok": False,
                "error": {"line": e.line, "col": e.col, "message": e.message}}
    return {"ok": True, "params": prog.params, "locals": prog.local_names,
            "n_ops": len(prog.code), "state_floats": prog.state_floats}


def decimate_idx(T: int, target: int = 6000) -> np.ndarray:
    if T <= target:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, target).astype(np.int64))


def pick_engine(name: str) -> str:
    if name == "cpu" or (name == "auto" and cpu_engine.available()):
        return "cpu"
    return "ref"


# ---------- endpoints ----------

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/datasets")
def datasets():
    return list_cached()


@app.get("/api/status")
def status():
    return {"cpu_engine": cpu_engine.available()}


@app.post("/api/compile")
def api_compile(req: CompileReq):
    return compile_info(req.source)


@app.post("/api/run")
def api_run(req: RunReq):
    info = compile_info(req.source)
    if not info["ok"]:
        return {"ok": False, "error": info["error"]}
    prog = compile_source(req.source)

    try:
        bars = get_bars(req.symbol, req.interval)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None

    engine_name = pick_engine(req.engine)
    notice = None
    n = min(req.last_bars, len(bars.close)) if req.last_bars > 0 else len(bars.close)
    if engine_name == "ref" and n > REF_ENGINE_BAR_CAP:
        n = REF_ENGINE_BAR_CAP
        notice = (f"C++ engine not built yet - python reference capped to "
                  f"last {REF_ENGINE_BAR_CAP:,} bars")
    view = bars.slice(len(bars.close) - n, len(bars.close))

    pvals = []
    for p in prog.params:
        v = float(req.params.get(p["name"], p["default"]))
        pvals.append(min(max(v, p["lo"]), p["hi"]))

    eng = cpu_engine if engine_name == "cpu" else refengine
    res = eng.run(prog, view, pvals or None, record_locals=bool(prog.local_names))

    ts = (view.ts // 1000).astype(np.float64)
    idx = decimate_idx(len(view.ts))
    equity = np.asarray(res.equity)

    out = {
        "ok": True,
        "engine": engine_name,
        "notice": notice,
        "bars_used": int(n),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in res.metrics.items()},
        "params_used": {p["name"]: pvals[i] for i, p in enumerate(prog.params)},
        "chart": {
            "ts": ts[idx].tolist(),
            "close": np.asarray(view.close, dtype=np.float64)[idx].tolist(),
            "equity": equity.astype(np.float64)[idx].tolist(),
        },
        "trades": [
            {"side": tr.side, "entry_ts": float(ts[tr.entry_t]),
             "exit_ts": float(ts[tr.exit_t]), "entry_px": tr.entry_px,
             "exit_px": tr.exit_px, "pnl": tr.pnl, "reason": tr.reason}
            for tr in res.trades[:3000]
        ],
        "n_trades_total": len(res.trades),
    }

    if res.locals_curve is not None and prog.local_names:
        med = float(np.median(view.close))
        overlays = {}
        for i, name in enumerate(prog.local_names):
            col = res.locals_curve[:, i].astype(np.float64)
            m = float(np.median(np.abs(col[np.isfinite(col)]))) if len(col) else 0.0
            if med > 0 and 0.2 * med <= m <= 5.0 * med:
                overlays[name] = col[idx].tolist()
        out["chart"]["overlays"] = overlays
    return out


@app.post("/api/sweep")
def api_sweep(req: SweepReq):
    info = compile_info(req.source)
    if not info["ok"]:
        return {"ok": False, "error": info["error"]}
    prog = compile_source(req.source)
    pspec = {p["name"]: p for p in prog.params}
    if req.x.name not in pspec:
        raise HTTPException(400, f"unknown param {req.x.name!r}")
    if req.y and req.y.name not in pspec:
        raise HTTPException(400, f"unknown param {req.y.name!r}")
    if req.y and req.y.name == req.x.name:
        raise HTTPException(400, "x and y must differ")
    if req.metric not in sweeplib.COL:
        raise HTTPException(400, f"unknown metric {req.metric!r}")

    bars = get_bars(req.symbol, req.interval)
    n = min(req.last_bars, len(bars.close)) if req.last_bars > 0 else len(bars.close)
    view = bars.slice(len(bars.close) - n, len(bars.close))

    xv = sweeplib.axis_values(pspec[req.x.name], req.x.steps)
    yv = sweeplib.axis_values(pspec[req.y.name], req.y.steps) if req.y else np.array([0.0])
    nx, ny = len(xv), len(yv)

    pm = np.zeros((nx * ny, len(prog.params)), dtype=np.float32)
    for j, p in enumerate(prog.params):
        v = float(req.fixed.get(p["name"], p["default"]))
        pm[:, j] = min(max(v, p["lo"]), p["hi"])
    xi = [p["name"] for p in prog.params].index(req.x.name)
    grid_x, grid_y = np.meshgrid(xv, yv, indexing="ij")     # (nx, ny)
    pm[:, xi] = grid_x.ravel()
    if req.y:
        yi = [p["name"] for p in prog.params].index(req.y.name)
        pm[:, yi] = grid_y.ravel()

    col = sweeplib.COL[req.metric]
    if req.n_folds > 1:
        _, test = sweeplib.fold_metrics(prog, view, pm, req.n_folds)
        vals = test[:, :, col].mean(axis=0)
        sharpes = test[:, :, sweeplib.COL["sharpe"]].mean(axis=0)
    else:
        eng = sweeplib.pick_engine()
        m = eng.run_batch(prog, view, pm)
        vals = m[:, col]
        sharpes = m[:, sweeplib.COL["sharpe"]]

    matrix = vals.reshape(nx, ny)
    best_flat = int(np.nanargmax(vals))
    dsr = sweeplib.deflated_sharpe(float(sharpes[best_flat]), sharpes, n)

    return {
        "ok": True,
        "engine": "gpu" if gpu_engine.available() else "cpu",
        "x": {"name": req.x.name, "values": xv.astype(float).tolist()},
        "y": ({"name": req.y.name, "values": yv.astype(float).tolist()}
              if req.y else None),
        "metric": req.metric,
        "matrix": [[float(v) for v in row] for row in matrix],
        "best": {"params": {p["name"]: float(pm[best_flat, j])
                            for j, p in enumerate(prog.params)},
                 "value": float(vals[best_flat]),
                 "deflated_sharpe": None if np.isnan(dsr) else float(dsr)},
        "combos": int(nx * ny),
        "folds": req.n_folds,
    }


# ---------- evolution ----------

class EvolveStartReq(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    last_bars: int = 129_600
    pop_size: int = 24
    generations: int = 10
    n_param_samples: int = 48
    n_folds: int = 3
    seed: int = 0
    llm_inject: bool = False        # Claude proposes novel variants every few gens
    llm_hint: str = ""              # optional human guidance passed to Claude
    llm_max_calls: int = 8          # budget guard


_evo_lock = threading.Lock()
_evo_run: EvolutionRun | None = None
_evo_thread: threading.Thread | None = None


def _make_injector(hint: str, max_calls: int):
    """Every 3rd generation, ask Claude for novel variants seeded from the
    current top performers. Budget-guarded; failures never break the run."""
    calls = {"n": 0}

    def inject(run: EvolutionRun, gen: int) -> list[Individual]:
        if gen % 3 != 2 or calls["n"] >= max_calls:
            return []
        try:
            top = run.population_json()[:3]
            calls["n"] += 2
            out = []
            for src in llm.inject_novelty(top, hint=hint or None, n=2):
                out.append(Individual(genome.from_source(src), src))
            return out
        except Exception:
            return []

    return inject


@app.post("/api/evolve/start")
def evolve_start(req: EvolveStartReq):
    global _evo_run, _evo_thread
    with _evo_lock:
        if _evo_thread is not None and _evo_thread.is_alive():
            raise HTTPException(409, "an evolution run is already active")
        bars = get_bars(req.symbol, req.interval)
        n = min(req.last_bars, len(bars.close)) if req.last_bars > 0 else len(bars.close)
        view = bars.slice(len(bars.close) - n, len(bars.close))
        inject = (_make_injector(req.llm_hint, req.llm_max_calls)
                  if req.llm_inject and llm.available() else None)
        _evo_run = EvolutionRun(
            view, pop_size=req.pop_size, generations=req.generations,
            n_param_samples=req.n_param_samples, n_folds=req.n_folds,
            seed=req.seed, prefer_gpu=gpu_engine.available(), inject=inject,
            label="lab")
        _evo_thread = threading.Thread(target=_evo_run.run, daemon=True)
        _evo_thread.start()
    return {"ok": True, "llm_inject": inject is not None}


@app.get("/api/evolve/status")
def evolve_status():
    if _evo_run is None:
        return {"state": "idle"}
    return _evo_run.status()


@app.get("/api/evolve/population")
def evolve_population():
    if _evo_run is None:
        return []
    return _evo_run.population_json()


@app.post("/api/evolve/stop")
def evolve_stop():
    if _evo_run is not None:
        _evo_run.stop_requested = True
    return {"ok": True}


# ---------- Claude (idea box + conversational improvement) ----------

class IdeaReq(BaseModel):
    idea: str


class ImproveReq(BaseModel):
    source: str
    instruction: str
    metrics: dict | None = None


@app.get("/api/ai/status")
def ai_status():
    return {"available": llm.available(), "model": llm.MODEL}


@app.post("/api/ai/idea")
def ai_idea(req: IdeaReq):
    if not llm.available():
        raise HTTPException(503, "Claude API not configured (set ANTHROPIC_API_KEY)")
    try:
        return llm.generate_strategy(req.idea)
    except Exception as e:
        raise HTTPException(502, f"Claude request failed: {e}") from None


@app.post("/api/ai/improve")
def ai_improve(req: ImproveReq):
    if not llm.available():
        raise HTTPException(503, "Claude API not configured (set ANTHROPIC_API_KEY)")
    try:
        return llm.improve_strategy(req.source, req.instruction, req.metrics)
    except Exception as e:
        raise HTTPException(502, f"Claude request failed: {e}") from None


# ---------- strategy library ----------

def _safe(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,60}", name):
        raise HTTPException(400, "name must be [A-Za-z0-9_-], max 60 chars")
    return name


@app.get("/api/strategies")
def strategies():
    out = []
    for group, folder in (("seed", SEEDS), ("library", LIBRARY)):
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.endswith(".dsl"):
                    out.append({"group": group, "name": fn[:-4]})
    return out


@app.get("/api/strategy/{group}/{name}")
def strategy_get(group: str, name: str):
    folder = SEEDS if group == "seed" else LIBRARY
    path = os.path.join(folder, _safe(name) + ".dsl")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, encoding="utf-8") as f:
        return {"source": f.read()}


@app.post("/api/strategy")
def strategy_save(req: SaveReq):
    os.makedirs(LIBRARY, exist_ok=True)
    path = os.path.join(LIBRARY, _safe(req.name) + ".dsl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(req.source)
    return {"ok": True, "name": req.name}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
