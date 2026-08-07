# Future Research Directions

The current cumulative evidence contains 1,775 coordinates, 449,850 trades, and five compatible stages. Continuation Round 2 moved both cost-adjusted total-return leaders into the E320/BH240/TRW12/K1.25/W6/M4.5 neighborhood; their main difference is the S340 versus S400 speed window. The items below are testable research questions, not backtest-execution or HTML-delivery rules.

## 1. Determine whether the leaders sit on a stable parameter region

Examine nearby W, M, and S combinations around the shared E320/BH240/TRW12/K1.25/W6/M4.5 structure. The main question is whether S340 through S400 contains a continuous plateau or parameter ridge rather than two isolated peaks. Interpret the Scenario-1 and unrestricted objectives separately.

## 2. Run time-slice and out-of-sample validation

Freeze the current leaders as candidates and test their return, maximum drawdown, trade count, and rank on data that does not overlap the training period. Time-sliced analysis within the existing sample can also show whether the results are concentrated in a few dates or one market regime.

## 3. Quantify dependence on gap-spanning trades

The current Scenario-1 and unrestricted leaders have gross gap-excluded returns of -0.3590% and +0.1949%, respectively. Separate the return contribution, trade count, and drawdown of gap-spanning and non-gap trades to determine whether the current advantage mainly comes from overnight or gap behavior. Gap-excluded return remains diagnostic under the current method; changing its ranking role would require a separate research scope.

## 4. Test cost and trading-frequency robustness

Use the hash-bound K200M current-notional cost reference to evaluate reasonable slippage and commission scenarios. Check whether the leaders remain profitable and similarly ranked as cost assumptions change, and compare very high-frequency combinations with the current 113- and 130-trade candidates. Raw fills and raw returns remain unchanged.

## 5. Extend cross-instrument transfer evidence

The first frozen-candidate transfer to SImain SIH6 is complete for 2026-01-29 through 2026-02-23. Future work may add another non-overlapping SImain interval, another instrument, or a separately labeled SImain full-grid post-hoc diagnostic. Each extension must freeze candidates before target evaluation and use the target contract multiplier, tick size, trading hours, commission, and data-quality assumptions.

## 6. Compare alternative method definitions

Use the current net-drop, rebound-confirmation, and W-baseline definitions as the control, then test whether alternative definitions improve time stability, reduce gap dependence, or lower parameter sensitivity. An alternative definition creates a new method identity and must be interpreted separately from the current V4.4 evidence.
