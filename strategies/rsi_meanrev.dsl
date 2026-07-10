# RSI mean reversion: buy the dip when RSI recovers through the oversold line.
param period = 14 in [5, 50]
param oversold = 30 in [10, 45]
param overbought = 70 in [55, 90]

let r = rsi(close, period)

enter_long when crossover(r, oversold)
exit_long when crossover(r, overbought)

set stop_loss = 0.03
set size = 0.75
