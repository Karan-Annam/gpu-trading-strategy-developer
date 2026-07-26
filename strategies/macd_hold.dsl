# MACD momentum with a time-based exit: ride the cross, but never sit in a
# stale position longer than max_hold bars.
param fast = 12 in [5, 50]
param slow = 26 in [10, 100]
param max_hold = 240 in [30, 2000] step 30

let m = macd(close, fast, slow)

enter_long when crossover(m, 0)
exit_long when crossunder(m, 0) or bars_held >= max_hold

set stop_loss = 0.03
