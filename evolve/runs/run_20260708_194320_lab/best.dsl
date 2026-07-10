param p1 = 80 in [27, 240]
param p2 = 120 in [40, 360]
param p3 = -0.0276 in [-0.04416, -0.01104]

enter_long when roc(close, p1) < p3
exit_long when high > sma(close, p2)
