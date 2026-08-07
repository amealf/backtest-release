# Instrument Profile Contract

## Required profile fields

Each executable instrument profile binds:

- `instrument_id`, display name, timezone, and the shared strategy-contract ID;
- market data and preparation-manifest path, size, and SHA-256, plus explicit `bar_seconds`;
- a frozen cost-model reference;
- a supported gap-policy mode with hash-bound configuration and implementation identities;
- a supported low-activity policy with hash-bound configuration and implementation identities plus the causal baseline-availability rule;
- an optional scenario set;
- a sample-specific `ranking_lineage_id`.

Missing cost or gap decisions keep a profile at `requires_user_input`; execution must stop and request those values. SImain and NQ templates deliberately contain no invented fees, slippage, FX, or gap policy. The profile, preparation manifest, and observed market-data grid must all declare the strategy's 15-second granularity; any difference stops execution and reports the expected and observed values.

## Cost model

Notional is calculated as `reference price × point value`. Commission is converted into the quote currency with the frozen FX reference when needed. Round-trip cost is:

`slippage bps + 10,000 × round-trip commission in quote currency / contract notional`.

The normalized output fields are `point_value`, `quote_currency`, `commission`, `slippage_bps`, `contract_notional_quote`, and `round_trip_total_cost_bps`. The K200 legacy aliases remain available for historical tests and pages.

Existing K200 results keep their frozen cost reference. A later K200 stage may bind a newer price-derived reference. Each coordinate retains its source-stage cost model and can remain in the same K200 ranking lineage with its actual `round_trip_cost_bps` disclosed.

## Scenario and low-activity rules

`scenario_set` is optional. Under schema-v5+ `profile_optional`, a profile with no scenario set produces an empty scenario population and cannot fall back to K200 events. Only an explicit legacy policy may load legacy K200 events. K200 event intervals stay in the K200 profile and remain separate filters or diagnostic views. A transfer or fresh-search primary result cannot require K200 scenario qualification.

Gap and low-activity policy labels are executable identities, not free-form descriptions. The profile policy IDs, configuration hashes, implementation hashes, and preparation-manifest declarations must match before compute. Those identities enter the plan fingerprint, stage contract, completion evidence, and result-semantics boundary. The shared low-activity lifecycle remains instrument-neutral; instrument thresholds and enable/disable decisions come only from the bound policy.

## Current files

- Ready K200 profile: `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`.
- SImain input template: `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\simain.template.json`.
- NQ input template: `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\nq.template.json`.
