# Repaired K200 to SImain Exact-Transfer Report

## Frozen source selection

- Source snapshot: `0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626` with 4,747 K200 coordinates.
- Eligibility: top 20% by K200 cost-adjusted total return, at least 10 K200 trades, W/M/S families already present in previous frozen transfers, and exclusion of 266 previously transferred coordinates plus three current champions.
- The eligible pool contains 225 coordinates. Within-family Pareto selection on higher source return, higher median actual entry threshold, and lower source trade count retains 64 coordinates across 11 W/M/S families.
- The 64 candidates were frozen before target evaluation with content SHA-256 `0623937812e4f669799b7eeca30f9f1d7201d05762a6669cbefcd146e2d50d68`.

## SImain result

- Exact SIH6 15-second transfer, 2026-01-29 through 2026-02-23, fixed 3.57-bps research cost, four workers.
- 6,755 SImain trades across 64 candidates; 58 candidates are cost-positive (90.625%).
- Median candidate result: 4.5883% target cost-adjusted total return, 111 target trades, 0.0769% mean trade, and 27.1303% maximum drawdown.
- Twenty-nine candidates are in target-positive stable regions; twenty-six positive candidates are isolated.
- K200-to-SImain cost-return rank Spearman is -0.45728. Source rank therefore does not transfer as target rank.

## Leaders and interpretation

- SImain total-return leader: E320/BH240/TRW21/K1.05/W6/M4.5/S340, 85 trades, 19.6892% target cost-adjusted total return, 0.2508% mean trade, and 17.5481% maximum drawdown. It is isolated in the transferred set.
- Best clearly stable high-return point: E290/BH240/TRW20/K1.1/W6/M4.5/S330, 86 trades, 18.0685% target return, 0.2307% mean trade, and 16.2734% drawdown.
- The K200 leader E480/BH171/TRW12/K1.26/W6/M4.5/S388 transfers to 120 SImain trades and only 1.1311% target return with 30.3146% drawdown.
- The repaired transfer has a higher positive fraction than the historical 266-candidate set (90.63% versus 87.59%), but weaker median target return (4.59% versus 5.80%) and higher median drawdown (27.13% versus 19.10%). It supports a broad positive-transfer tendency, not a better target ranking or parameter acceptance.

`parameter_acceptance=none`; no SImain local grid was run.
