"""Claude integration: idea -> DSL, conversational improvement, and novelty
injection for the evolution loop.

The DSL compiler is the safety gate for everything the model returns — LLM
text is never executed, only compiled. One retry with compiler feedback.
"""

from __future__ import annotations

import os
import re

from dsl.compiler import compile_source
from dsl.errors import CompileError

MODEL = os.environ.get("BT_LLM_MODEL", "claude-opus-4-8")
MAX_TOKENS = 4000

DSL_GUIDE = """\
You write trading strategies in a small DSL. Complete language reference:

STATEMENTS (one per line, # comments):
  param NAME = DEFAULT in [LO, HI]        # tunable number (add `step S` optionally)
  let NAME = EXPR                          # per-bar value
  enter_long when BOOLEXPR                 # aliases: buy / sell / short / cover
  exit_long when BOOLEXPR
  enter_short when BOOLEXPR
  exit_short when BOOLEXPR
  set stop_loss = EXPR                     # fraction, e.g. 0.02 = 2%. 0 disables
  set take_profit = EXPR
  set trail_stop = EXPR
  set size = EXPR                          # fraction of equity in [0,1]

EXPRESSIONS:
  series: open high low close volume; lagged: close[3] (integer literal lag)
  context: bar_index, position (-1/0/1), entry_price, equity
  operators: + - * /   > < >= <= == !=   and or not   parentheses
  stateless: abs(x) min(a,b) max(a,b) sqrt(x) log(x)
  indicators: sma(x,n) ema(x,n) rsi(x,n) atr(n) highest(x,n) lowest(x,n)
              stddev(x,n) delay(x,k) crossover(a,b) crossunder(a,b)
              change(x) roc(x,n)

HARD RULES:
- Window args (n, k) must be a literal, a param, or simple arithmetic of
  those, with upper bound <= 4096. Never a series or indicator value.
- Booleans are floats; comparisons yield 1/0. No if/else, no loops, no lets
  referencing later lets. Entries fire only when flat; an exit closes the
  position before re-entry. Fills happen at NEXT bar open (no lookahead).
- Fees (~0.1%/trade) are charged by the backtester: strategies that trade
  every few bars on 1-minute data lose to fees. Prefer fewer, larger moves.
- Declare 2-4 params max, with sensible [lo, hi] ranges around the default.
- Keep strategies scale-free: compare prices to prices (or ATR multiples),
  oscillators to constants in their natural range. Never compare price to a
  hardcoded absolute level.

Example:
param lookback = 120 in [20, 500] step 20
param trail_mult = 3 in [1, 6]
let hi_ch = highest(high, lookback)
let vol = atr(14)
enter_long when close > delay(hi_ch, 1)
exit_long when crossunder(close, sma(close, lookback))
set trail_stop = trail_mult * vol / close

Respond with ONLY the DSL program (comments with # are fine). No prose, no
markdown fences unless you wrap the whole program in one ```dsl block.
"""

_client = None


def available() -> bool:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.exists(os.path.expanduser("~/.config/anthropic"))
                or os.path.exists(os.path.join(os.environ.get("APPDATA", ""), "Anthropic")))


def _get_client():
    global _client
    if _client is None:
        import anthropic  # noqa: PLC0415
        _client = anthropic.Anthropic()
    return _client


def _extract_dsl(text: str) -> str:
    m = re.search(r"```(?:dsl)?\s*\n(.*?)```", text, re.DOTALL)
    src = m.group(1) if m else text
    return src.strip() + "\n"


def _ask(messages: list[dict]) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=DSL_GUIDE,
        messages=messages,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    return next(b.text for b in resp.content if b.type == "text")


def _generate_with_gate(messages: list[dict]) -> dict:
    """Ask, compile-check, retry once with the compiler error."""
    text = _ask(messages)
    src = _extract_dsl(text)
    try:
        compile_source(src)
        return {"ok": True, "source": src}
    except CompileError as e:
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                f"That does not compile: line {e.line}: {e.message}\n"
                f"Fix it and respond with only the corrected DSL program."},
        ]
        text2 = _ask(retry)
        src2 = _extract_dsl(text2)
        try:
            compile_source(src2)
            return {"ok": True, "source": src2}
        except CompileError as e2:
            return {"ok": False,
                    "error": f"Claude's output failed to compile twice: {e2}"}


def generate_strategy(idea: str) -> dict:
    """Plain-English idea -> compilable DSL."""
    return _generate_with_gate([{
        "role": "user",
        "content": f"Write a strategy implementing this idea:\n\n{idea}",
    }])


def improve_strategy(source: str, instruction: str, metrics: dict | None = None) -> dict:
    """Revise an existing strategy per the user's instruction."""
    stats = ""
    if metrics:
        keys = ("total_return", "sharpe", "max_drawdown", "n_trades",
                "win_rate", "exposure")
        stats = "\nRecent backtest metrics:\n" + "\n".join(
            f"  {k}: {metrics[k]:.4f}" if isinstance(metrics.get(k), float)
            else f"  {k}: {metrics.get(k)}" for k in keys if k in metrics)
    return _generate_with_gate([{
        "role": "user",
        "content": (f"Current strategy:\n```dsl\n{source}```\n{stats}\n\n"
                    f"Revise it per this instruction: {instruction}\n"
                    f"Keep unrelated parts unchanged."),
    }])


def inject_novelty(top: list[dict], hint: str | None = None, n: int = 2) -> list[str]:
    """Given top performers (src + stats), propose n novel variants.
    Returns compilable sources only (failed generations are dropped)."""
    summary = "\n\n".join(
        f"### Strategy {i + 1} (fitness {t.get('fitness', 0):.3f}, "
        f"oos sharpe {t.get('mean_oos_sharpe', 0):.2f}, "
        f"{t.get('median_oos_trades', 0):.0f} trades/fold)\n```dsl\n{t['src']}```"
        for i, t in enumerate(top))
    hint_txt = f"\nGuidance from the user: {hint}\n" if hint else ""
    out = []
    for k in range(n):
        res = _generate_with_gate([{
            "role": "user",
            "content": (
                f"These are the current best strategies in an evolutionary "
                f"search (fitness = out-of-sample robustness across "
                f"walk-forward folds, penalized for complexity):\n\n{summary}\n"
                f"{hint_txt}\n"
                f"Propose ONE new strategy (variant #{k + 1}) that is "
                f"meaningfully different from all of the above — a different "
                f"signal family or regime filter, not a parameter tweak. "
                f"It should still respect the fee constraint."),
        }])
        if res["ok"]:
            out.append(res["source"])
    return out
