# K200 New-Rules Multi-Round Exploration Report

## Scope

- Current unrestricted anchor: E480, BH171, TRW12, K1.26, W6, M4.5, S388.
- Ten rounds changed one parameter at a time and tested three new points per round with every other field fixed.
- Four compute workers were used. Intermediate rounds closed raw evidence and compact summaries without HTML.
- The cumulative main and shared per-trade HTML were published once after the series ended.

## Round comparison

| Round | Tested direction | Best point in round | Cost-adjusted return | Average trade | Maximum drawdown | Gap-excluded gross return | Classification |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Anchor | — | BH171 / TRW12 / K1.26 / W6 | 82.4352% | 0.5618% | 15.1770% | 4.6795% | reference |
| 3 | BH expansion: 205, 257, 480 | BH205 | 69.6174% | 0.4885% | 17.0915% | 0.3884% | not_improved |
| 4 | TRW expansion: 13, 14, 15 | TRW13 | 71.9266% | 0.5273% | 16.2463% | 1.0059% | not_improved |
| 5 | K expansion: 1.4, 1.5, 1.6 | K1.4 | 71.1573% | 0.5330% | 16.6701% | 1.2043% | not_improved |
| 6 | K contraction: 1.15, 1.05, 0.95 | K1.15 | 62.0151% | 0.4171% | 19.9481% | -1.9047% | not_improved |
| 7 | W expansion: 7, 8, 9 | W7 | 82.4664% | 0.5571% | 15.2433% | 3.2002% | mixed |
| 8 | TRW contraction: 11, 10, 9 | TRW11 | 61.9046% | 0.4199% | 20.0330% | -1.8834% | not_improved |
| 9 | BH contraction: 145, 137, 128 | BH145 | 69.8267% | 0.4778% | 17.3873% | -1.2200% | not_improved |
| 10 | M expansion: 4.75, 5.0, 5.5 | M4.75 | 79.6419% | 0.5432% | 15.8120% | 3.2814% | not_improved |
| 11 | M contraction: 4.25, 4.0, 3.75 | M3.75 | 65.2421% | 0.4231% | 16.2942% | 0.3383% | not_improved |
| 12 | S expansion: 427, 466, 520 | S466 | 52.6175% | 0.4282% | 16.1588% | -4.4748% | not_improved |

## Interpretation

BH expansion, TRW expansion, and both K directions are consistently weaker than the anchor. K1.26 is therefore a local peak under the fixed surrounding parameters, without becoming an accepted parameter.

W7 becomes the cumulative cost-adjusted total-return leader by 0.0312 percentage points. The gain is isolated: W8 and W9 are materially weaker, while W7 has lower average trade, slightly higher drawdown, and lower gap-excluded return than W6. This does not establish a useful W expansion direction.

The additional five rounds further define the local surface. TRW and BH contraction both weaken; together with the prior expansion rounds, TRW12 and BH171 are near local peaks. Both sides of M4.5 are weaker, and expanding S from 388 also falls sharply. None of the fifteen added points forms a continuing direction.

The series closes with `parameter_acceptance=none`. W7 remains an isolated cumulative total-return leader, while W6 remains the more stable representative anchor.

## Final publication

- Snapshot: `0126cd77b436aef1434e7072bac0d6dfa15b3d2ad4dc2cf1b2fafe936ee1e626`.
- Population: 4,747 coordinates, 713,886 trades, and thirteen compatible stages.
- Ten-round evidence: 30 coordinates and 3,485 trades.
- Duplicate coordinates: zero.
- Stable main and shared per-trade entries were refreshed once at the end.
