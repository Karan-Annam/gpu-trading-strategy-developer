param p1 = 14 in [5, 42]
param p2 = 0.03 in [0.0075, 0.12]
param p3 = 0.5 in [0.1875, 0.5]

let r = rsi(close, p1)
enter_long when crossover(r, 30)
exit_long when crossover(r, 70)
set stop_loss = p2
set size = p3
