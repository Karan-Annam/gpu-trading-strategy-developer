# Classic SMA crossover, long-only, with a hard stop.
param fast = 20 in [5, 100] step 5
param slow = 100 in [20, 400] step 20
param sl = 0.02 in [0.005, 0.1]

let f = sma(close, fast)
let s = sma(close, slow)

enter_long when crossover(f, s)
exit_long when crossunder(f, s)

set stop_loss = sl
