@echo off
cd /d "D:\Code\backtest-release\Backtest v2 ratio"
set LM_GRID_SHARD_TAG=shard70c
set LM_GRID_OPEN_BAR_VALUES=95,100
echo LM_GRID_SHARD_TAG=%LM_GRID_SHARD_TAG%
echo LM_GRID_OPEN_BAR_VALUES=%LM_GRID_OPEN_BAR_VALUES%
python -u long_momentum.py
