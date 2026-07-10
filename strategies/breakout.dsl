# Donchian-style breakout with an ATR trailing stop.
param lookback = 120 in [20, 500] step 20
param atr_n = 14 in [5, 50]
param trail_mult = 3 in [1, 6]

let hi_ch = highest(high, lookback)
let vol = atr(atr_n)

enter_long when close > delay(hi_ch, 1)
exit_long when crossunder(close, sma(close, lookback))

set trail_stop = trail_mult * vol / close
