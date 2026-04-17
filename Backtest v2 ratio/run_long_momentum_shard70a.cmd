@echo off
cd /d "D:\Code\backtest-release\Backtest v2 ratio"
set LM_GRID_SHARD_TAG=shard70a
set LM_GRID_OPEN_BAR_VALUES=70,75,80
echo LM_GRID_SHARD_TAG=%LM_GRID_SHARD_TAG%
echo LM_GRID_OPEN_BAR_VALUES=%LM_GRID_OPEN_BAR_VALUES%
python -u long_momentum.py
