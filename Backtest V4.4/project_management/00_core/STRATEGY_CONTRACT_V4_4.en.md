# Strategy Contract V4.4

## Purpose

This document defines the instrument-neutral trading protocol. Instrument data, sessions, costs, gap rules, low-activity thresholds, and event dates belong to instrument profiles or campaign manifests. The current strategy sampling contract is one completed 15-second bar per execution atom; a transfer target must use the same granularity.

## Entry contract

- Find the ordered high anchor H and measure the E-window decline from that anchor.
- Build the entry baseline from completed TR atoms inside BH, grouped by TRW.
- The entry threshold is `max(K × baseline, absolute floor)`.
- Entry qualification requires finite `baseline`, `drop`, and `threshold`, with `baseline > 0`, `drop > 0`, `threshold > 0`, and `drop >= threshold`. Equality remains a valid positive-boundary signal; zero baseline/drop/threshold cannot open a trade.
- A calculated-threshold signal fills at the threshold unless the real bar opens through it, in which case the open is used.
- A signal formed on a non-real bar is retained and fills at the first real-trade open within 120 continuous candidate bars.

## Exit contract

- W uses available prefixes from 1 through W and measures `open[start] - low[end]`.
- The W start is no earlier than H, the continuous-segment start, or `end-W+1`.
- M converts the maximum completed W decline into the rebound threshold.
- A prior trigger exits at `open >= trigger`; otherwise `high >= trigger` exits at the trigger. Equality exits.
- A strict same-bar new low is confirmed with the close and exits at that close.
- S detects a stopped decline. Non-real exits wait for the next real-trade open.
- Any position remaining at the declared sample end closes at that sample-end close.

## Shared state machine

Flat reset, calculated-threshold fill, synthetic bars, pending entry, pending exit, and the confirmed low-activity gate are part of the core. A low-volume run has no strategy effect while pending: its atoms remain baseline-eligible and entries remain allowed. If it recovers before confirmation, the run is discarded. At the 120th consecutive low-volume 15-second atom, the run confirms: all atoms from its first atom through confirmation become unavailable to every later baseline calculation, unfilled entry orders are cancelled, and new entries are blocked through the final low-volume atom. The first normal-volume atom ends the gate and is immediately eligible for new baseline sampling and entry evaluation. Existing positions continue their normal exit rules. This confirmation-time rule uses no future state.

## Instrument boundary

The core contains no instrument name, exchange, currency, multiplier, commission, slippage, scenario date, or session clock. It does bind the shared 15-second execution granularity and positive-entry qualification. The executable JSON authority is `research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json`.
