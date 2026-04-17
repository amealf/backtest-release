$ErrorActionPreference = 'Stop'
Set-Location 'D:\Code\backtest-release\Backtest v2 ratio'
$env:LM_GRID_SHARD_TAG = 'shard70a'
$env:LM_GRID_OPEN_BAR_VALUES = '70,75,80'
Write-Host ('LM_GRID_SHARD_TAG=' + $env:LM_GRID_SHARD_TAG)
Write-Host ('LM_GRID_OPEN_BAR_VALUES=' + $env:LM_GRID_OPEN_BAR_VALUES)
python -u '.\long_momentum.py'
