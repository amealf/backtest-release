"""Instrument, cost, and experiment contracts for cross-instrument V4.4 work.

The legacy K200 cost file remains a valid input.  This module normalizes it to
currency-neutral field names while retaining its historical aliases so current
K200 analysis bytes and tests can remain stable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import csv
from pathlib import Path
from typing import Any


INSTRUMENT_PROFILE_SCHEMA_VERSION = 2
SUPPORTED_INSTRUMENT_PROFILE_SCHEMA_VERSIONS = (1, 2)
CAMPAIGN_MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_CAMPAIGN_MANIFEST_SCHEMA_VERSIONS = (1, 2)
EXPERIMENT_MODES = frozenset(
    {
        "transfer_exact",
        "target_local_refinement",
        "continuation_search",
        "fresh_search",
    }
)
SUPPORTED_GAP_POLICY_MODES = frozenset(
    {"preserve_existing_v4_4_behavior", "session_calendar_and_real_gap", "none"}
)
SUPPORTED_LOW_ACTIVITY_POLICY_MODES = frozenset(
    {"causal_pending_buffer", "pending_no_effect_confirmed_gate", "none"}
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
STRATEGY_BAR_SECONDS = 15
EXPLORATION_GUIDE = (
    Path(__file__).resolve().parents[3]
    / "project_management"
    / "05_domains"
    / "research"
    / "PARAMETER_EXPLORATION_GUIDE.en.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result or not ID_PATTERN.fullmatch(result):
        raise ValueError(f"{field} contains an unsupported identifier")
    return result


def _positive_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _nonnegative_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _resolve(path_value: Any, parent: Path, field: str) -> Path:
    if path_value in (None, ""):
        raise ValueError(f"{field} is required")
    raw = Path(str(path_value))
    return raw.resolve() if raw.is_absolute() else (parent / raw).resolve()


def _verify_bound_file(record: dict[str, Any], parent: Path, field: str) -> Path:
    path = _resolve(record.get("path"), parent, f"{field}.path")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(record.get("sha256", "")).strip()
    expected_size = record.get("size_bytes")
    if expected_hash and sha256_file(path) != expected_hash:
        raise ValueError(f"{field} SHA-256 does not match")
    if expected_size not in (None, "") and path.stat().st_size != int(expected_size):
        raise ValueError(f"{field} size does not match")
    return path


def _load_bound_json(
    record: dict[str, Any], parent: Path, field: str
) -> tuple[Path, dict[str, Any]]:
    path = _verify_bound_file(record, parent, field)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return path, payload


def _observed_bar_seconds(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "bar_seconds" in fields:
            observed = {int(float(row["bar_seconds"])) for row in reader if row["bar_seconds"] not in (None, "")}
            if len(observed) != 1:
                raise ValueError(f"market data contains mixed bar_seconds: {sorted(observed)}")
            return next(iter(observed))
        time_field = next(
            (name for name in ("datetime", "timestamp", "time") if name in fields),
            None,
        )
        if time_field is None:
            raise ValueError("market data lacks a datetime/timestamp/time column")
        from datetime import datetime

        previous = None
        interval_gcd = 0
        positive_count = 0
        for row in reader:
            current = datetime.fromisoformat(str(row[time_field]).replace("Z", "+00:00"))
            if previous is not None:
                delta = int((current - previous).total_seconds())
                if delta > 0:
                    interval_gcd = math.gcd(interval_gcd, delta)
                    positive_count += 1
            previous = current
    if positive_count == 0 or interval_gcd <= 0:
        raise ValueError("market data has no positive bar interval")
    return interval_gcd


def _policy_contract(
    record: dict[str, Any],
    parent: Path,
    field: str,
    supported_modes: frozenset[str],
) -> dict[str, Any]:
    if str(record.get("status", "")) != "ready":
        raise ValueError(f"{field} is not ready")
    policy_id = _safe_id(record.get("policy_id", record.get("id")), f"{field}.policy_id")
    mode = str(record.get("mode", "")).strip()
    if mode not in supported_modes:
        raise ValueError(f"unsupported {field}.mode: {mode}")
    config = record.get("config")
    implementation = record.get("implementation")
    if not isinstance(config, dict) or not isinstance(implementation, dict):
        raise ValueError(f"{field} requires bound config and implementation files")
    config_path = _verify_bound_file(config, parent, f"{field}.config")
    implementation_path = _verify_bound_file(
        implementation, parent, f"{field}.implementation"
    )
    return {
        "policy_id": policy_id,
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": str(implementation_path),
        "implementation_sha256": sha256_file(implementation_path),
    }


def load_cost_model(path: Path) -> dict[str, Any]:
    """Load a frozen cost reference and derive notional-dependent round-trip bps."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("contract")
    market_price = payload.get("market_price")
    inputs = payload.get("cost_inputs")
    if not all(isinstance(item, dict) for item in (contract, market_price, inputs)):
        raise ValueError("cost reference requires contract, market_price, and cost_inputs")

    legacy_krw = "contract_multiplier_krw_per_point" in contract
    point_value = _positive_number(
        contract.get("contract_multiplier_krw_per_point")
        if legacy_krw
        else contract.get("point_value"),
        "contract.point_value",
    )
    quote_currency = str(
        "KRW" if legacy_krw else contract.get("quote_currency", "")
    ).strip().upper()
    if not quote_currency:
        raise ValueError("contract.quote_currency is required")
    price = _positive_number(market_price.get("price"), "market_price.price")
    slippage_bps = _nonnegative_number(
        inputs.get("round_trip_slippage_bps"),
        "cost_inputs.round_trip_slippage_bps",
    )

    if legacy_krw:
        commission = _nonnegative_number(
            inputs.get("round_trip_commission_usd"),
            "cost_inputs.round_trip_commission_usd",
        )
        commission_currency = "USD"
        fx_payload = payload.get("fx")
        if not isinstance(fx_payload, dict):
            raise ValueError("legacy K200 cost reference requires fx")
        quote_per_commission_currency = _positive_number(
            fx_payload.get("krw_per_usd"), "fx.krw_per_usd"
        )
    else:
        commission = _nonnegative_number(
            inputs.get("round_trip_commission"),
            "cost_inputs.round_trip_commission",
        )
        commission_currency = str(inputs.get("commission_currency", "")).strip().upper()
        if not commission_currency:
            raise ValueError("cost_inputs.commission_currency is required")
        if commission_currency == quote_currency:
            quote_per_commission_currency = 1.0
        else:
            fx_payload = payload.get("fx")
            if not isinstance(fx_payload, dict):
                raise ValueError("cross-currency commission requires fx")
            quote_per_commission_currency = _positive_number(
                fx_payload.get("quote_per_commission_currency"),
                "fx.quote_per_commission_currency",
            )

    notional_quote = price * point_value
    commission_quote = commission * quote_per_commission_currency
    commission_bps = 10000.0 * commission_quote / notional_quote
    slippage_quote = notional_quote * slippage_bps / 10000.0
    total_quote = commission_quote + slippage_quote
    total_bps = slippage_bps + commission_bps
    derived = payload.get("derived")
    if isinstance(derived, dict):
        aliases = (
            {
                "contract_notional_krw": notional_quote,
                "round_trip_commission_krw": commission_quote,
                "round_trip_commission_bps": commission_bps,
                "round_trip_slippage_krw": slippage_quote,
                "round_trip_total_cost_krw": total_quote,
                "round_trip_total_cost_bps": total_bps,
            }
            if legacy_krw
            else {
                "contract_notional_quote": notional_quote,
                "round_trip_commission_quote": commission_quote,
                "round_trip_commission_bps": commission_bps,
                "round_trip_slippage_quote": slippage_quote,
                "round_trip_total_cost_quote": total_quote,
                "round_trip_total_cost_bps": total_bps,
            }
        )
        for key, value in aliases.items():
            if key in derived and not math.isclose(
                float(derived[key]), value, rel_tol=0.0, abs_tol=1e-10
            ):
                raise ValueError(f"cost reference derived value differs: {key}")

    result = {
        "id": _safe_id(payload.get("cost_model_id"), "cost_model_id"),
        "role": str(payload.get("role", "derived_ranking_and_display_only")),
        "reference_date": str(payload.get("reference_date", "")),
        "reference_path": str(path),
        "reference_sha256": sha256_file(path),
        "instrument_id": str(payload.get("instrument_id", "k200m" if legacy_krw else "")),
        "instrument_name": str(contract.get("instrument", "")),
        "reference_price": price,
        "reference_price_currency": str(market_price.get("currency", quote_currency)),
        "reference_price_observed_at": str(market_price.get("observed_at", "")),
        "reference_price_selection_rule": str(market_price.get("selection_rule", "")),
        "point_value": point_value,
        "quote_currency": quote_currency,
        "contract_notional_quote": notional_quote,
        "commission_currency": commission_currency,
        "round_trip_commission": commission,
        "quote_per_commission_currency": quote_per_commission_currency,
        "round_trip_slippage_bps": slippage_bps,
        "round_trip_commission_bps": commission_bps,
        "round_trip_commission_quote": commission_quote,
        "round_trip_slippage_quote": slippage_quote,
        "round_trip_total_cost_quote": total_quote,
        "round_trip_total_cost_bps": total_bps,
        "ranking_display_default": "cost_adjusted",
        "ranking_basis": "cost_adjusted",
        "available_ranking_display_modes": ["cost_adjusted", "gross"],
    }
    if legacy_krw:
        result.update(
            {
                "contract_multiplier_krw_per_point": point_value,
                "contract_notional_krw": notional_quote,
                "usdkrw": quote_per_commission_currency,
                "round_trip_commission_usd": commission,
                "round_trip_commission_krw": commission_quote,
                "round_trip_slippage_krw": slippage_quote,
                "round_trip_total_cost_krw": total_quote,
            }
        )
    return result


def load_instrument_profile(
    path: Path,
    *,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Load an instrument profile and resolve all executable file bindings."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_schema_version = int(payload.get("schema_version", 0))
    if profile_schema_version not in SUPPORTED_INSTRUMENT_PROFILE_SCHEMA_VERSIONS:
        raise ValueError("unsupported instrument profile schema_version")
    status = str(payload.get("status", "")).strip()
    if require_ready and status != "ready":
        raise ValueError("instrument profile is not ready for execution")
    instrument_id = _safe_id(payload.get("instrument_id"), "instrument_id")
    strategy_contract_id = _safe_id(
        payload.get("strategy_contract_id"), "strategy_contract_id"
    )
    ranking_lineage_id = _safe_id(
        payload.get("ranking_lineage_id"), "ranking_lineage_id"
    )
    bar_seconds = int(payload.get("bar_seconds", STRATEGY_BAR_SECONDS if profile_schema_version == 1 else 0))
    if bar_seconds != STRATEGY_BAR_SECONDS:
        raise ValueError(
            f"instrument profile bar_seconds differs: expected={STRATEGY_BAR_SECONDS}, actual={bar_seconds}"
        )
    data = payload.get("data")
    cost = payload.get("cost_model")
    gap = payload.get("gap_policy")
    low_activity = payload.get("low_activity_policy")
    if not all(isinstance(item, dict) for item in (data, cost, gap, low_activity)):
        raise ValueError(
            "instrument profile requires data, cost_model, gap_policy, and low_activity_policy"
        )
    if require_ready:
        market_data = _verify_bound_file(data, path.parent, "data")
        preparation_record = data.get("preparation_manifest")
        if not isinstance(preparation_record, dict):
            raise ValueError("data.preparation_manifest is required")
        preparation = _verify_bound_file(
            preparation_record, path.parent, "data.preparation_manifest"
        )
        cost_path = _verify_bound_file(cost, path.parent, "cost_model")
        cost_model = load_cost_model(cost_path)
        if cost_model["instrument_id"] not in ("", instrument_id):
            raise ValueError("cost model instrument_id differs from instrument profile")
        if profile_schema_version >= 2:
            preparation_payload = json.loads(preparation.read_text(encoding="utf-8"))
            prepared_bar_seconds = int(preparation_payload.get("bar_seconds", 0))
            if prepared_bar_seconds not in (0, bar_seconds):
                raise ValueError(
                    f"preparation bar_seconds differs: expected={bar_seconds}, actual={prepared_bar_seconds}"
                )
            observed_bar_seconds = _observed_bar_seconds(market_data)
            if observed_bar_seconds != bar_seconds:
                raise ValueError(
                    f"market-data bar_seconds differs: expected={bar_seconds}, actual={observed_bar_seconds}"
                )
            gap_contract = _policy_contract(
                gap, path.parent, "gap_policy", SUPPORTED_GAP_POLICY_MODES
            )
            low_activity_contract = _policy_contract(
                low_activity,
                path.parent,
                "low_activity_policy",
                SUPPORTED_LOW_ACTIVITY_POLICY_MODES,
            )
            declared = preparation_payload.get("policy_contracts")
            policy_attestation_path: Path | None = None
            if not isinstance(declared, dict):
                attestation_record = data.get("policy_attestation")
                if not isinstance(attestation_record, dict):
                    raise ValueError(
                        "preparation manifest lacks policy_contracts and profile lacks policy_attestation"
                    )
                policy_attestation_path, attestation = _load_bound_json(
                    attestation_record, path.parent, "data.policy_attestation"
                )
                if str(attestation.get("status", "")) != "complete":
                    raise ValueError("policy attestation is not complete")
                if str(attestation.get("preparation_manifest_sha256", "")) != sha256_file(preparation):
                    raise ValueError("policy attestation binds a different preparation manifest")
                if int(attestation.get("bar_seconds", 0)) != bar_seconds:
                    raise ValueError("policy attestation bar_seconds differs")
                declared = attestation.get("policy_contracts")
            elif prepared_bar_seconds == 0:
                raise ValueError(
                    "preparation manifest lacks bar_seconds and requires policy_attestation"
                )
            if not isinstance(declared, dict):
                raise ValueError("preparation manifest lacks policy_contracts")
            for name, contract in (
                ("gap_policy", gap_contract),
                ("low_activity_policy", low_activity_contract),
            ):
                prepared = declared.get(name)
                if not isinstance(prepared, dict):
                    raise ValueError(f"preparation manifest lacks {name}")
                for key in ("policy_id", "mode", "config_sha256", "implementation_sha256"):
                    if str(prepared.get(key, "")) != str(contract[key]):
                        raise ValueError(
                            f"instrument profile {name} differs from preparation manifest: {key}"
                        )
        else:
            if str(gap.get("status", "")) != "ready":
                raise ValueError("gap policy is not ready")
            if str(low_activity.get("status", "")) != "ready":
                raise ValueError("low-activity policy is not ready")
            gap_contract = {"policy_id": str(gap.get("id", "")), "mode": str(gap.get("mode", ""))}
            low_activity_contract = {
                "policy_id": str(low_activity.get("id", "")),
                "mode": "legacy_profile_label_only",
            }
    else:
        market_data = None
        preparation = None
        cost_path = None
        cost_model = None

    scenario = payload.get("scenario_set")
    scenario_path: Path | None = None
    if isinstance(scenario, dict) and scenario.get("path") not in (None, ""):
        scenario_path = _verify_bound_file(scenario, path.parent, "scenario_set")
    elif scenario not in (None, {}):
        raise ValueError("scenario_set must be null or a file binding")

    return {
        **payload,
        "instrument_id": instrument_id,
        "strategy_contract_id": strategy_contract_id,
        "ranking_lineage_id": ranking_lineage_id,
        "bar_seconds": bar_seconds,
        "profile_path": str(path),
        "profile_sha256": sha256_file(path),
        "resolved_market_data_path": str(market_data) if market_data else None,
        "resolved_preparation_manifest_path": str(preparation) if preparation else None,
        "resolved_cost_model_path": str(cost_path) if cost_path else None,
        "resolved_scenario_set_path": str(scenario_path) if scenario_path else None,
        "normalized_cost_model": cost_model,
        "gap_policy_contract": gap_contract if require_ready else None,
        "low_activity_policy_contract": low_activity_contract if require_ready else None,
        "resolved_policy_attestation_path": (
            str(policy_attestation_path)
            if require_ready and profile_schema_version >= 2 and policy_attestation_path
            else None
        ),
    }


def load_campaign_manifest(path: Path, *, require_ready: bool = True) -> dict[str, Any]:
    """Validate one transfer, local-refinement, or fresh-search campaign contract."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_schema_version = int(payload.get("schema_version", 0))
    if manifest_schema_version not in SUPPORTED_CAMPAIGN_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError("unsupported campaign manifest schema_version")
    campaign_id = _safe_id(payload.get("campaign_id"), "campaign_id")
    mode = str(payload.get("mode", "")).strip()
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"unsupported campaign mode: {mode}")
    if require_ready and str(payload.get("status", "")) != "ready":
        raise ValueError("campaign manifest is not ready for execution")
    profile_record = payload.get("instrument_profile")
    if not isinstance(profile_record, dict):
        raise ValueError("campaign manifest requires instrument_profile")
    profile_path = _verify_bound_file(profile_record, path.parent, "instrument_profile")
    profile = load_instrument_profile(profile_path, require_ready=require_ready)
    if manifest_schema_version >= 2 and int(profile.get("schema_version", 0)) < 2:
        raise ValueError("schema-v2 campaign manifests require a schema-v2 instrument profile")
    ranking = payload.get("ranking")
    if not isinstance(ranking, dict):
        raise ValueError("campaign manifest requires ranking")
    if str(ranking.get("lineage_id", "")) != profile["ranking_lineage_id"]:
        raise ValueError("campaign ranking lineage differs from instrument profile")
    if ranking.get("merge_policy") != "same_instrument_compatible_lineage_only":
        raise ValueError("campaign ranking merge policy is not supported")

    source = payload.get("source")
    search = payload.get("search")
    resolved_contract: dict[str, Any] = {}
    if mode == "transfer_exact":
        if not isinstance(source, dict) or not isinstance(source.get("candidate_freeze"), dict):
            raise ValueError("transfer_exact requires a frozen source candidate set")
        freeze_path, freeze_payload = _load_bound_json(
            source["candidate_freeze"], path.parent, "source.candidate_freeze"
        )
        if payload.get("target_tuning_allowed") is not False:
            raise ValueError("transfer_exact must forbid target tuning")
        resolved_contract.update(candidate_freeze_path=str(freeze_path), candidate_freeze_payload=freeze_payload)
    elif mode == "target_local_refinement":
        if not isinstance(source, dict) or not isinstance(source.get("parent_transfer"), dict):
            raise ValueError("target_local_refinement requires a completed transfer parent")
        parent_path, parent_payload = _load_bound_json(
            source["parent_transfer"], path.parent, "source.parent_transfer"
        )
        freeze_record = source.get("parent_candidate_freeze")
        if manifest_schema_version >= 2 and not isinstance(freeze_record, dict):
            raise ValueError("target_local_refinement requires parent_candidate_freeze")
        freeze_path = freeze_payload = None
        if isinstance(freeze_record, dict):
            freeze_path, freeze_payload = _load_bound_json(
                freeze_record, path.parent, "source.parent_candidate_freeze"
            )
        if not isinstance(search, dict) or search.get("scope") != "bounded_neighborhood":
            raise ValueError("target_local_refinement requires a bounded neighborhood")
        if manifest_schema_version >= 2 and not isinstance(search.get("neighborhood"), dict):
            raise ValueError("target_local_refinement requires machine-readable neighborhood")
        resolved_contract.update(
            parent_path=str(parent_path), parent_payload=parent_payload,
            candidate_freeze_path=str(freeze_path) if freeze_path else None,
            candidate_freeze_payload=freeze_payload,
        )
    elif mode == "continuation_search":
        if not isinstance(source, dict) or not isinstance(source.get("parent_stage"), dict):
            raise ValueError("continuation_search requires a completed parent_stage")
        parent_path, parent_payload = _load_bound_json(
            source["parent_stage"], path.parent, "source.parent_stage"
        )
        if not isinstance(search, dict) or search.get("scope") != "continued_exploration":
            raise ValueError("continuation_search requires continued_exploration")
        resolved_contract.update(parent_path=str(parent_path), parent_payload=parent_payload)
    else:
        if source not in (None, {}):
            raise ValueError("fresh_search must not import source-instrument candidates")
        if not isinstance(search, dict) or search.get("scope") != "from_scratch":
            raise ValueError("fresh_search requires a from-scratch search definition")
        if manifest_schema_version >= 2:
            exploration = search.get("exploration_contract")
            if not isinstance(exploration, dict):
                raise ValueError("fresh_search requires exploration_contract")
            exploration_path = _verify_bound_file(
                exploration, path.parent, "search.exploration_contract"
            )
            if exploration_path != EXPLORATION_GUIDE.resolve():
                raise ValueError("fresh_search must bind the canonical parameter exploration guide")
            if not isinstance(search.get("parameter_space"), dict):
                raise ValueError("fresh_search requires parameter_space")
            resolved_contract["exploration_contract_path"] = str(exploration_path)

    return {
        **payload,
        "campaign_id": campaign_id,
        "mode": mode,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "resolved_instrument_profile_path": str(profile_path),
        "instrument_profile_contract": profile,
        "manifest_schema_version": manifest_schema_version,
        "resolved_mode_contract": resolved_contract,
    }
