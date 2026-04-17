@echo off
set PYTHONUNBUFFERED=1
set LM_RUN_MODE=grid
set LM_CLOSE_WITHDRAWAL_MODE=fixed_high_pct
set LM_GRID_SHARD_TAG=shard80a
set LM_GRID_OPEN_BAR_VALUES=80,90
"C:\ProgramData\anaconda3\python.exe" -u "D:\Code\backtest-release\Backtest v2 ratio\long_momentum.py"
pause
