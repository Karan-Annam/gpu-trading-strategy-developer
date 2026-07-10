param p1 = 20 in [7, 60]
param p2 = 100 in [33, 300]
param p3 = 0.02 in [0.005, 0.08]

let f = sma(close, p1)
let s = sma(close, p2)
enter_long when crossover(f, s)
exit_long when crossunder(f, s)
set stop_loss = p3
