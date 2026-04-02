# XAGUSD GARCH forecast report

## Setup
- Source CSV: `D:\Code\data\20260326\yearly_30s\xagusd_30s_2025.csv`
- Resampled bar: `5m`
- Training window: `2025-02-01 00:00:00 -> 2025-05-31 23:55:00`
- Forecast window: `2025-06-01 00:00:00 -> 2025-06-15 23:55:00`
- Forecast horizon: `12 bars / 60 minutes`
- Figure: `D:\Code\backtest-release\Backtest v2 ratio\GARCH\garch_validate_output\xagusd_garch_5m_20250601_20250615.png`
- Evaluation CSV: `D:\Code\backtest-release\Backtest v2 ratio\GARCH\garch_validate_output\xagusd_garch_5m_20250601_20250615_eval.csv`

## Model
- mu: `0.00100174`
- omega: `0.00030683`
- alpha: `0.12183673`
- beta: `0.85616004`
- alpha + beta: `0.97799677`

## Evaluation
- Train samples: `22846`
- Forecast samples: `2821`
- Eval samples: `2701`
- Mean predicted volatility: `0.345298`
- Mean realized volatility: `0.303995`
- Volatility correlation: `0.451884`
- Volatility RMSE: `0.166083`
- MZ intercept a: `0.059952`
- MZ slope b: `0.468353`
- MZ R^2: `0.108991`
- MZ joint F-test p-value: `0.000000`
- QLIKE: `-1.199340`
- Peak predicted volatility time: `2025-06-12 03:45:00`
- Peak predicted volatility: `1.132201`
- Peak realized volatility time: `2025-06-12 03:00:00`
- Peak realized volatility: `1.149793`

## Reading
- The forecast volatility level stays above the realized level for most of the window.
- The model captures part of the volatility swings, though the fit is still loose.
- The Mincer-Zarnowitz slope is far from 1, so the calibration needs caution.
- The QLIKE value is -1.199340. Lower values indicate a tighter forecast.
