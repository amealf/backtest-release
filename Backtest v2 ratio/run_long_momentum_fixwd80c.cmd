@echo off
setlocal
cd /d "D:\Code\backtest-release\Backtest v2 ratio"
set "LM_RUN_MODE=grid"
set "LM_CLOSE_WITHDRAWAL_MODE=legacy_low_to_high_2over3"
set "LM_GRID_SHARD_TAG=fixwd80c"
set "LM_GRID_OPEN_BAR_VALUES=130,140"
set "LM_MANUAL_OPEN_WD_THRESHOLD=0"
C:\ProgramData\anaconda3\python.exe -u "D:\Code\backtest-release\Backtest v2 ratio\long_momentum.py"

