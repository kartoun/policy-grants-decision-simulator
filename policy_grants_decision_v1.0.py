"""
Policy / Grants Decision Simulator (v1.0) — single-file Python generator
Runs with NO arguments. Users configure via the CONFIG dict ONLY.

Outputs (CSV) to CONFIG["output_dir"]:
  - policy_runs.csv
  - decision_traces.csv
  - artifacts.csv
  - run_summary_by_knob.csv
  - phase_bottlenecks.csv
  - reason_code_breakdown.csv
  - assumptions.csv
  - config_snapshot.json

VERSION policy:
- VERSION="1.0" stays forever.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# USER CONFIG (EDIT ONLY THIS BLOCK)
# ============================================================

CONFIG = {
    # Pick one: "sam_onegov" | "noaa_baa" | "darpa_eris"
    "regime": "sam_onegov",

    # Run controls
    "seed": 123,
    "runs": 80,
    "episodes_per_run": 400,
    "vendors_per_episode": [25, 70],   # [min,max] sampled uniformly each episode

    # Sensitivity experiment (policy knob perturbations)
    "sensitivity": {
        "enabled": True,
        "delta": 0.10,  # +/- 10% (additive for 0..1 knobs; multiplicative for base_award_days if used)
        "knobs": ["documentation_burden", "oem_direct_adoption", "sb_preference"],
    },

    # Output
    "output_dir": "output",
    "write_artifacts": True,
    "artifact_verbosity": "short",  # "short" | "medium"

    # Actor population (optional overrides)
    # If omitted, defaults are used.
    "vendor_population": {
        "sb_rate": 0.65,
        "capability_mean": 0.55,
        "capability_sd": 0.18,
        "risk_mean": 0.45,
        "risk_sd": 0.20,
        "pricing_mean": 0.55,
        "pricing_sd": 0.18,
        "capacity_mean": 0.55,
        "capacity_sd": 0.18,
    },

    "reviewer": {
        "load_mean": 0.45,
        "load_sd": 0.20,
        "strictness_mean": 0.55,
        "strictness_sd": 0.18,
    },

    # Optional: override policy knob ranges for the chosen regime
    # Example:
    # "policy_knob_ranges": {
    #   "documentation_burden": {"min": 0.7, "max": 0.95},
    #   "sb_preference": {"min": 0.2, "max": 0.6}
    # }
    "policy_knob_ranges": {}
}

# ============================================================
# END USER CONFIG
# ============================================================


# =============================
# Constants
# =============================

VERSION = "1.0"

# Reason codes (stable enums)
RC_OK = "OK"
RC_DROP_OUT = "DROP_OUT"
RC_DQ_MISSING_DOCS = "DQ_MISSING_DOCS"
RC_DQ_BUDGET_MISMATCH = "DQ_BUDGET_MISMATCH"
RC_DQ_EXPORT_CUI = "DQ_EXPORT_CUI"
RC_DQ_RESEARCH_SECURITY = "DQ_RESEARCH_SECURITY"
RC_DQ_INELIGIBLE = "DQ_INELIGIBLE"
RC_NO_BIDS = "NO_BIDS"
RC_NO_AWARD = "NO_AWARD"
RC_PROTEST = "PROTEST"

# CSV schemas (fixed column order)
POLICY_RUNS_HEADER = [
    "version","scenario","regime","run_id","policy_config_id","ts_utc",
    "episodes","awards","no_awards","avg_bids_per_episode","avg_bids_per_award",
    "dropouts","dq_count","avg_award_days","stdev_award_days","sb_win_rate",
    "avg_cost_score","avg_risk_score","avg_tech_score","avg_admin_load",
    "policy_json","effects_json",
]

DECISION_TRACES_HEADER = [
    "version","scenario","regime","run_id","episode_id","step_idx","phase","actor",
    "vendor_id","is_small_business","action","reason_code","p_value","days_added",
    "constraints_json","state_json","score_json","counterfactual_json",
]

ARTIFACTS_HEADER = [
    "version","scenario","regime","run_id","episode_id","artifact_type",
    "text","ts_utc","meta_json",
]

RUN_SUMMARY_BY_KNOB_HEADER = [
    "version","scenario","regime","run_id","baseline_policy_config_id","knob","direction","delta",
    "delta_awards","delta_no_awards","delta_avg_award_days","delta_sb_win_rate",
    "delta_avg_cost","delta_avg_risk","delta_avg_tech","delta_avg_admin_load",
    "ci95_low_avg_award_days","ci95_high_avg_award_days","ci95_low_sb_win_rate","ci95_high_sb_win_rate",
]

PHASE_BOTTLENECKS_HEADER = [
    "version","scenario","regime","phase","count","mean_days","p90_days",
    "share_of_total_days","top_reason_codes",
]

REASON_BREAKDOWN_HEADER = [
    "version","scenario","regime","reason_code","count","rate_per_episode","sb_share",
    "avg_complexity","avg_doc_burden","avg_budget_scale","avg_cyber",
]

ASSUMPTIONS_HEADER = ["version","scenario","regime","run_id","key","value"]


# =============================
# Utilities
# =============================

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(math.ceil((len(ys) - 1) * p))
    k = int(clamp(k, 0, len(ys) - 1))
    return ys[k]

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)

def ci95_normal(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], xs[0])
    m = mean(xs)
    s = stdev(xs)
    se = s / math.sqrt(len(xs))
    return (m - 1.96 * se, m + 1.96 * se)


# =============================
# Built-in regimes
# =============================

def built_in_regimes() -> Dict[str, Dict[str, Any]]:
    return {
        "sam_onegov": {
            "description": "Synthetic SAM/OneGov-style procurement workflow (non-operational).",
            "phases": [
                "POST","QNA","BID_DECISION","SUBMIT","COMPLIANCE_SCREEN",
                "TECH_REVIEW","COST_REALISM","RISK_REVIEW","AWARD","POST_AWARD",
            ],
            "policy_knobs": {
                "documentation_burden": {"min": 0.1, "max": 0.9},
                "oem_direct_adoption": {"min": 0.0, "max": 0.8},
                "sb_preference": {"min": 0.0, "max": 0.6},
                "base_award_days": {"min": 20, "max": 120},
                "w_technical": {"min": 0.35, "max": 0.75},
                "w_cost": {"min": 0.15, "max": 0.55},
                "w_risk": {"min": 0.05, "max": 0.25},
                "export_cui_strictness": {"min": 0.0, "max": 0.4},
                "protest_rate": {"min": 0.0, "max": 0.15},
            },
            "gates": {"export_cui": True, "research_security": False},
            "narrative_style": "procurement",
        },
        "noaa_baa": {
            "description": "Synthetic NOAA BAA-like grants workflow (non-operational).",
            "phases": [
                "POST","BID_DECISION","SUBMIT","COMPLIANCE_SCREEN","BUDGET_CHECK",
                "TECH_REVIEW","MERIT_REVIEW","AWARD","POST_AWARD",
            ],
            "policy_knobs": {
                "documentation_burden": {"min": 0.2, "max": 0.95},
                "sb_preference": {"min": 0.0, "max": 0.3},
                "base_award_days": {"min": 45, "max": 180},
                "w_technical": {"min": 0.45, "max": 0.80},
                "w_cost": {"min": 0.05, "max": 0.25},
                "w_risk": {"min": 0.10, "max": 0.35},
                "budget_scrutiny": {"min": 0.2, "max": 0.9},
                "cost_share_pressure": {"min": 0.0, "max": 0.4},
            },
            "gates": {"export_cui": False, "research_security": False},
            "narrative_style": "grant",
        },
        "darpa_eris": {
            "description": "Synthetic DARPA ERIS-style fast intake workflow (non-operational).",
            "phases": [
                "POST","BID_DECISION","SUBMIT","COMPLIANCE_SCREEN",
                "PITCH_REVIEW","TECH_REVIEW","TRANSITION_REVIEW","AWARD","POST_AWARD",
            ],
            "policy_knobs": {
                "documentation_burden": {"min": 0.05, "max": 0.6},
                "sb_preference": {"min": 0.0, "max": 0.2},
                "base_award_days": {"min": 15, "max": 90},
                "w_technical": {"min": 0.55, "max": 0.90},
                "w_cost": {"min": 0.02, "max": 0.18},
                "w_risk": {"min": 0.05, "max": 0.30},
                "research_security_strictness": {"min": 0.0, "max": 0.4},
            },
            "gates": {"export_cui": False, "research_security": True},
            "narrative_style": "darpa",
        },
    }


# =============================
# Build internal cfg from CONFIG
# =============================

def build_cfg_from_user_config(user: Dict[str, Any]) -> Dict[str, Any]:
    regimes = built_in_regimes()
    regime = str(user.get("regime", "sam_onegov"))
    if regime not in regimes:
        regime = "sam_onegov"

    cfg = {
        "meta": {
            "scenario_name": f"policy_grants_decision_simulator_v{VERSION}",
            "schema_version": VERSION,
        },
        "simulation": {
            "runs": int(user.get("runs", 8)),
            "episodes_per_run": int(user.get("episodes_per_run", 25)),
            "vendors_per_episode": user.get("vendors_per_episode", [6, 22]),
            "seed": int(user.get("seed", 123)),
            "sensitivity": {
                "enabled": bool(user.get("sensitivity", {}).get("enabled", True)),
                "delta": float(user.get("sensitivity", {}).get("delta", 0.10)),
                "knobs": list(user.get("sensitivity", {}).get("knobs", [])),
            },
        },
        "regime": {"name": regime},
        "actors": {
            "vendor_population": dict(user.get("vendor_population", {})),
            "reviewer": dict(user.get("reviewer", {})),
        },
        "outputs": {
            "write_artifacts": bool(user.get("write_artifacts", True)),
            "artifact_verbosity": str(user.get("artifact_verbosity", "short")).lower(),
            "out_dir": str(user.get("output_dir", "output")),
        },
        "policy_knob_ranges": dict(user.get("policy_knob_ranges", {})),
    }

    # Fill defaults if missing
    vp = cfg["actors"]["vendor_population"]
    vp.setdefault("sb_rate", 0.65)
    vp.setdefault("capability_mean", 0.55); vp.setdefault("capability_sd", 0.18)
    vp.setdefault("risk_mean", 0.45);      vp.setdefault("risk_sd", 0.20)
    vp.setdefault("pricing_mean", 0.55);   vp.setdefault("pricing_sd", 0.18)
    vp.setdefault("capacity_mean", 0.55);  vp.setdefault("capacity_sd", 0.18)

    rv = cfg["actors"]["reviewer"]
    rv.setdefault("load_mean", 0.45); rv.setdefault("load_sd", 0.20)
    rv.setdefault("strictness_mean", 0.55); rv.setdefault("strictness_sd", 0.18)

    # Validate vendors_per_episode
    vpe = cfg["simulation"]["vendors_per_episode"]
    if (not isinstance(vpe, list)) or len(vpe) != 2:
        cfg["simulation"]["vendors_per_episode"] = [6, 22]
    else:
        lo, hi = int(vpe[0]), int(vpe[1])
        if hi < lo:
            lo, hi = hi, lo
        cfg["simulation"]["vendors_per_episode"] = [max(1, lo), max(1, hi)]

    # Validate verbosity
    if cfg["outputs"]["artifact_verbosity"] not in ("short", "medium"):
        cfg["outputs"]["artifact_verbosity"] = "short"

    return cfg


# =============================
# Data models
# =============================

@dataclass
class PolicyConfig:
    policy_config_id: str
    knobs: Dict[str, float]

    def get(self, k: str, default: float = 0.0) -> float:
        return float(self.knobs.get(k, default))

    def normalized_weights(self) -> Tuple[float, float, float]:
        wt = self.get("w_technical", 0.6)
        wc = self.get("w_cost", 0.25)
        wr = self.get("w_risk", 0.15)
        s = max(1e-9, wt + wc + wr)
        return (wt / s, wc / s, wr / s)

@dataclass
class Vendor:
    vendor_id: str
    is_small_business: bool
    capability: float
    risk: float
    pricing: float
    capacity: float

@dataclass
class Episode:
    run_id: str
    episode_id: str
    complexity: float
    cyber_sensitivity: float
    budget_scale: float
    competition_heat: float
    export_cui_flag: bool
    research_security_flag: bool

@dataclass
class Budget:
    personnel: float
    fringe: float
    travel: float
    equipment: float
    supplies: float
    contractual: float
    other: float
    indirect: float
    total: float
    indirect_rate: float
    allowability_flags: List[str]
    mismatch_flag: bool

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "personnel": self.personnel,
            "fringe": self.fringe,
            "travel": self.travel,
            "equipment": self.equipment,
            "supplies": self.supplies,
            "contractual": self.contractual,
            "other": self.other,
            "indirect": self.indirect,
            "total": self.total,
            "indirect_rate": self.indirect_rate,
            "allowability_flags": self.allowability_flags,
            "mismatch_flag": self.mismatch_flag,
        }


# =============================
# Simulator core
# =============================

class PolicySimulatorV1_0:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.regimes = built_in_regimes()
        self.regime_name = cfg["regime"]["name"]
        self.regime = self.regimes[self.regime_name]
        self.rng = random.Random(int(cfg["simulation"]["seed"]))

        # Apply per-user knob range overrides (if any)
        # This changes the sampling ranges, not the semantics.
        self.knob_range_overrides: Dict[str, Dict[str, float]] = cfg.get("policy_knob_ranges", {}) or {}

    # ---------- Sampling helpers ----------

    def rand_norm01(self, mean_v: float, sd_v: float) -> float:
        return clamp(self.rng.gauss(mean_v, sd_v), 0.0, 1.0)

    def sample_int(self, lo_hi: List[int]) -> int:
        lo, hi = int(lo_hi[0]), int(lo_hi[1])
        return self.rng.randint(lo, hi)

    def _knob_range(self, knob: str, default_range: Dict[str, float]) -> Dict[str, float]:
        o = self.knob_range_overrides.get(knob)
        if isinstance(o, dict) and "min" in o and "max" in o:
            return {"min": float(o["min"]), "max": float(o["max"])}
        return {"min": float(default_range.get("min", 0.0)), "max": float(default_range.get("max", 1.0))}

    def sample_policy_knobs(self) -> Dict[str, float]:
        knobs_spec = self.regime.get("policy_knobs", {})
        knobs: Dict[str, float] = {}
        for k, mm in knobs_spec.items():
            rr = self._knob_range(k, mm)
            mn = safe_float(rr.get("min"), 0.0)
            mx = safe_float(rr.get("max"), 1.0)
            if mx < mn:
                mn, mx = mx, mn
            knobs[k] = self.rng.uniform(mn, mx)

        # clamps/defaults
        knobs["documentation_burden"] = clamp(knobs.get("documentation_burden", 0.5), 0.0, 1.0)
        knobs["sb_preference"] = clamp(knobs.get("sb_preference", 0.0), 0.0, 1.0)
        knobs["oem_direct_adoption"] = clamp(knobs.get("oem_direct_adoption", 0.0), 0.0, 1.0)
        knobs["export_cui_strictness"] = clamp(knobs.get("export_cui_strictness", 0.0), 0.0, 1.0)
        knobs["research_security_strictness"] = clamp(knobs.get("research_security_strictness", 0.0), 0.0, 1.0)
        knobs["budget_scrutiny"] = clamp(knobs.get("budget_scrutiny", 0.5), 0.0, 1.0)
        knobs["cost_share_pressure"] = clamp(knobs.get("cost_share_pressure", 0.0), 0.0, 1.0)
        knobs["protest_rate"] = clamp(knobs.get("protest_rate", 0.0), 0.0, 1.0)

        knobs.setdefault("w_technical", 0.6)
        knobs.setdefault("w_cost", 0.25)
        knobs.setdefault("w_risk", 0.15)
        knobs["base_award_days"] = clamp(knobs.get("base_award_days", 60.0), 5.0, 365.0)

        return knobs

    def sample_policy(self) -> PolicyConfig:
        pid = f"pc_{int(time.time()*1000)}_{self.rng.randint(1000, 9999)}"
        return PolicyConfig(policy_config_id=pid, knobs=self.sample_policy_knobs())

    def sample_episode(self, run_id: str, idx: int) -> Episode:
        ep_id = f"{run_id}_ep{idx:04d}"
        complexity = self.rng.random()
        cyber = self.rng.random()
        budget_scale = self.rng.random()
        heat = self.rng.random()

        export_flag = False
        rs_flag = False
        if self.regime.get("gates", {}).get("export_cui", False):
            export_flag = (self.rng.random() < 0.20 + 0.30 * cyber)
        if self.regime.get("gates", {}).get("research_security", False):
            rs_flag = (self.rng.random() < 0.10 + 0.15 * complexity)

        return Episode(
            run_id=run_id,
            episode_id=ep_id,
            complexity=complexity,
            cyber_sensitivity=cyber,
            budget_scale=budget_scale,
            competition_heat=heat,
            export_cui_flag=export_flag,
            research_security_flag=rs_flag,
        )

    def sample_vendors(self, episode_id: str, n: int) -> List[Vendor]:
        pop = self.cfg["actors"]["vendor_population"]
        sb_rate = clamp(safe_float(pop.get("sb_rate"), 0.65), 0.0, 1.0)

        vendors: List[Vendor] = []
        for i in range(n):
            vid = f"{episode_id}_v{i:03d}"
            is_sb = (self.rng.random() < sb_rate)

            cap = self.rand_norm01(safe_float(pop.get("capability_mean"), 0.55),
                                   safe_float(pop.get("capability_sd"), 0.18))
            risk = self.rand_norm01(safe_float(pop.get("risk_mean"), 0.45),
                                    safe_float(pop.get("risk_sd"), 0.20))
            pricing = self.rand_norm01(safe_float(pop.get("pricing_mean"), 0.55),
                                       safe_float(pop.get("pricing_sd"), 0.18))
            capacity = self.rand_norm01(safe_float(pop.get("capacity_mean"), 0.55),
                                        safe_float(pop.get("capacity_sd"), 0.18))

            # SBs: more variance + slightly lower capacity on average
            if is_sb:
                cap = clamp(cap + self.rng.gauss(0, 0.03), 0.0, 1.0)
                risk = clamp(risk + self.rng.gauss(0, 0.03), 0.0, 1.0)
                pricing = clamp(pricing + self.rng.gauss(0, 0.03), 0.0, 1.0)
                capacity = clamp(capacity + self.rng.gauss(-0.03, 0.05), 0.0, 1.0)

            vendors.append(Vendor(vid, is_sb, cap, risk, pricing, capacity))
        return vendors

    # ---------- Budget model ----------

    def generate_budget(self, policy: PolicyConfig, episode: Episode, vendor: Vendor) -> Budget:
        total = 50_000 + (2_450_000 * episode.budget_scale)

        personnel = total * clamp(0.35 + 0.25 * vendor.capability, 0.25, 0.65)
        fringe = personnel * clamp(0.20 + 0.20 * self.rng.random(), 0.18, 0.45)
        travel = total * clamp(0.02 + 0.06 * self.rng.random(), 0.00, 0.12)
        equipment = total * clamp(0.00 + 0.12 * episode.complexity, 0.00, 0.25)
        supplies = total * clamp(0.02 + 0.05 * self.rng.random(), 0.00, 0.10)
        contractual = total * clamp(0.05 + 0.25 * (1.0 - vendor.capacity), 0.02, 0.35)
        other = total * clamp(0.02 + 0.08 * self.rng.random(), 0.00, 0.12)

        subtotal_direct = personnel + fringe + travel + equipment + supplies + contractual + other

        indirect_rate = clamp(
            0.10 + 0.35 * vendor.capacity - 0.10 * policy.get("documentation_burden", 0.5),
            0.05, 0.55
        )
        indirect = subtotal_direct * indirect_rate

        allow_flags: List[str] = []
        if travel / total > 0.10:
            allow_flags.append("TRAVEL_HIGH")
        if equipment / total > 0.20:
            allow_flags.append("EQUIPMENT_HIGH")
        if contractual / total > 0.30:
            allow_flags.append("CONTRACTUAL_HIGH")
        if indirect_rate > 0.40:
            allow_flags.append("INDIRECT_HIGH")

        scrutiny = policy.get("budget_scrutiny", 0.5)
        burden = policy.get("documentation_burden", 0.5)
        mismatch_p = clamp(0.02 + 0.18 * burden + 0.12 * scrutiny - 0.08 * vendor.capacity, 0.0, 0.60)
        mismatch = (self.rng.random() < mismatch_p)

        computed_total = subtotal_direct + indirect
        if mismatch:
            computed_total = computed_total * (1.0 + self.rng.uniform(-0.06, 0.06))

        def rd(x: float) -> float:
            return float(int(round(x)))

        return Budget(
            personnel=rd(personnel), fringe=rd(fringe), travel=rd(travel), equipment=rd(equipment),
            supplies=rd(supplies), contractual=rd(contractual), other=rd(other),
            indirect=rd(indirect), total=rd(computed_total),
            indirect_rate=float(round(indirect_rate, 4)),
            allowability_flags=allow_flags,
            mismatch_flag=mismatch,
        )

    # ---------- Strategy + Gates ----------

    def vendor_bid_decision(self, policy: PolicyConfig, episode: Episode, vendor: Vendor, expected_competitors: int) -> Tuple[bool, float]:
        burden = policy.get("documentation_burden", 0.5)
        heat = episode.competition_heat
        complexity = episode.complexity

        win_logit = (
            -0.5
            + 2.0 * (vendor.capability - 0.5)
            - 1.2 * (vendor.risk - 0.5)
            - 0.6 * complexity
            - 0.4 * heat
        )
        win_p = sigmoid(win_logit)

        effort = (0.15 + 0.55 * burden + 0.25 * complexity + 0.30 * (1.0 - vendor.capacity))
        if vendor.is_small_business:
            effort += 0.08

        reward = 0.6 + 0.4 * episode.budget_scale
        comp_penalty = 0.04 * math.log(1 + max(1, expected_competitors))
        utility = win_p * reward - effort - comp_penalty

        p_bid = sigmoid(4.0 * utility)
        will_bid = (self.rng.random() < p_bid)
        return will_bid, p_bid

    def dropout_probability(self, policy: PolicyConfig, episode: Episode, vendor: Vendor) -> float:
        burden = policy.get("documentation_burden", 0.5)
        complexity = episode.complexity
        base = 0.03
        sb_boost = 0.06 if vendor.is_small_business else 0.02
        capacity_relief = -0.10 * vendor.capacity
        cap_relief = -0.05 * vendor.capability
        p = base + 0.30 * burden + 0.18 * complexity + sb_boost + capacity_relief + cap_relief
        return clamp(p, 0.0, 0.85)

    def compliance_screen(self, policy: PolicyConfig, episode: Episode, vendor: Vendor, budget: Budget) -> Tuple[bool, str, float]:
        burden = policy.get("documentation_burden", 0.5)

        p_missing_docs = clamp(0.03 + 0.28 * burden + 0.18 * (1.0 - vendor.capacity), 0.0, 0.80)
        if self.rng.random() < p_missing_docs:
            return False, RC_DQ_MISSING_DOCS, p_missing_docs

        if "BUDGET_CHECK" in self.regime["phases"]:
            scrutiny = policy.get("budget_scrutiny", 0.5)
            p_budget_fail = clamp((0.02 + 0.10 * scrutiny) + (0.35 if budget.mismatch_flag else 0.0), 0.0, 0.95)
            if self.rng.random() < p_budget_fail:
                return False, RC_DQ_BUDGET_MISMATCH, p_budget_fail

        if episode.export_cui_flag and self.regime.get("gates", {}).get("export_cui", False):
            strict = policy.get("export_cui_strictness", 0.0)
            p_export_fail = clamp(0.02 + 0.35 * strict + 0.12 * vendor.risk, 0.0, 0.90)
            if self.rng.random() < p_export_fail:
                return False, RC_DQ_EXPORT_CUI, p_export_fail

        if episode.research_security_flag and self.regime.get("gates", {}).get("research_security", False):
            strict = policy.get("research_security_strictness", 0.0)
            p_rs_fail = clamp(0.02 + 0.35 * strict + 0.10 * vendor.risk, 0.0, 0.90)
            if self.rng.random() < p_rs_fail:
                return False, RC_DQ_RESEARCH_SECURITY, p_rs_fail

        return True, RC_OK, 0.0

    # ---------- Scoring ----------

    def reviewer_state(self) -> Tuple[float, float]:
        r = self.cfg["actors"]["reviewer"]
        load = self.rand_norm01(safe_float(r.get("load_mean"), 0.45), safe_float(r.get("load_sd"), 0.20))
        strict = self.rand_norm01(safe_float(r.get("strictness_mean"), 0.55), safe_float(r.get("strictness_sd"), 0.18))
        return load, strict

    def score_vendor(self, policy: PolicyConfig, episode: Episode, vendor: Vendor, budget: Budget,
                     reviewer_load: float, reviewer_strict: float) -> Dict[str, float]:
        wT, wC, wR = policy.normalized_weights()

        complexity = episode.complexity
        cyber = episode.cyber_sensitivity
        burden = policy.get("documentation_burden", 0.5)
        oem = policy.get("oem_direct_adoption", 0.0)
        sb_pref = policy.get("sb_preference", 0.0)

        noise = 0.04 + 0.10 * reviewer_load + 0.10 * complexity

        tech = vendor.capability
        tech = clamp(tech - 0.30 * complexity * (1.0 - vendor.capability), 0.0, 1.0)
        tech = clamp(tech + self.rng.gauss(0, noise), 0.0, 1.0)

        price_noise_sd = 0.10 * (1.0 - oem) + 0.02
        eff_price = clamp(vendor.pricing + self.rng.gauss(0, price_noise_sd), 0.0, 1.0)
        cost = clamp(1.0 - eff_price, 0.0, 1.0)

        budget_penalty = 0.0
        if budget.mismatch_flag:
            budget_penalty += 0.08 + 0.12 * policy.get("budget_scrutiny", 0.5)
        if "INDIRECT_HIGH" in budget.allowability_flags:
            budget_penalty += 0.06
        if "CONTRACTUAL_HIGH" in budget.allowability_flags:
            budget_penalty += 0.04
        cost = clamp(cost - budget_penalty + self.rng.gauss(0, noise * 0.8), 0.0, 1.0)

        rsk = vendor.risk
        rsk = clamp(rsk + 0.18 * cyber * rsk, 0.0, 1.0)
        rsk = clamp(rsk + 0.08 * burden * (1.0 - vendor.capacity), 0.0, 1.0)
        rsk = clamp(rsk - 0.08 * oem, 0.0, 1.0)
        risk_score = clamp(1.0 - rsk + self.rng.gauss(0, noise * 0.9), 0.0, 1.0)

        sb_bonus = sb_pref * (0.08 if vendor.is_small_business else 0.0)

        total = clamp(wT * tech + wC * cost + wR * risk_score + sb_bonus, 0.0, 1.0)
        total = clamp(total - 0.06 * reviewer_strict * (1.0 - total), 0.0, 1.0)

        return {
            "tech": tech, "cost": cost, "risk": risk_score, "total": total,
            "eff_price": eff_price, "wT": wT, "wC": wC, "wR": wR,
            "sb_bonus": sb_bonus, "budget_penalty": budget_penalty,
        }

    # ---------- Timing model ----------

    def phase_time_days(self, policy: PolicyConfig, episode: Episode, phase: str, bids_in_play: int) -> float:
        burden = policy.get("documentation_burden", 0.5)
        base = policy.get("base_award_days", 60.0)
        complexity = episode.complexity
        oem = policy.get("oem_direct_adoption", 0.0)

        phase_weights = {
            "POST": 0.05, "QNA": 0.06, "BID_DECISION": 0.02, "SUBMIT": 0.08,
            "COMPLIANCE_SCREEN": 0.10, "BUDGET_CHECK": 0.08, "TECH_REVIEW": 0.22,
            "MERIT_REVIEW": 0.16, "COST_REALISM": 0.12, "RISK_REVIEW": 0.10,
            "PITCH_REVIEW": 0.08, "TRANSITION_REVIEW": 0.10, "AWARD": 0.10, "POST_AWARD": 0.03,
        }
        w = phase_weights.get(phase, 0.08)

        admin_amp = 1.0 + 0.9 * burden + 0.6 * complexity
        volume_amp = 1.0 + 0.25 * math.log(1 + max(1, bids_in_play))
        oem_relief = 1.0 - 0.20 * oem if phase in ("COST_REALISM","COMPLIANCE_SCREEN","AWARD") else 1.0

        days = base * w * admin_amp * volume_amp * oem_relief
        return clamp(days, 0.2, 120.0)

    # ---------- Analyst helpers ----------

    def _assumption_rows_for_run(self, scenario_name: str, run_id: str, policy: PolicyConfig) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        sim = self.cfg["simulation"]
        base_items = {
            "seed": sim.get("seed", ""),
            "runs": sim.get("runs", ""),
            "episodes_per_run": sim.get("episodes_per_run", ""),
            "vendors_per_episode": sim.get("vendors_per_episode", ""),
            "regime_description": self.regime.get("description", ""),
            "phases": self.regime.get("phases", []),
            "policy_config_id": policy.policy_config_id,
            "policy_knobs": policy.knobs,
            "vendor_population_params": self.cfg["actors"]["vendor_population"],
            "reviewer_params": self.cfg["actors"]["reviewer"],
        }
        for k, v in base_items.items():
            rows.append({
                "version": VERSION, "scenario": scenario_name, "regime": self.regime_name, "run_id": run_id,
                "key": str(k), "value": jdump(v) if isinstance(v, (dict, list)) else str(v),
            })
        return rows

    def _reason_event(self, reason_code: str, policy: PolicyConfig, episode: Episode, vendor: Optional[Vendor]) -> Dict[str, Any]:
        return {
            "reason_code": reason_code,
            "is_small_business": int(vendor.is_small_business) if vendor else "",
            "complexity": episode.complexity,
            "doc_burden": policy.get("documentation_burden", 0.0),
            "budget_scale": episode.budget_scale,
            "cyber": episode.cyber_sensitivity,
        }

    def _build_phase_bottlenecks_rows(self, scenario_name: str,
                                      phase_days: Dict[str, List[float]],
                                      total_days: float,
                                      reason_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reason_counts: Dict[str, int] = {}
        for ev in reason_events:
            rc = str(ev.get("reason_code", ""))
            if rc:
                reason_counts[rc] = reason_counts.get(rc, 0) + 1
        top_rc = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_rc_str = ";".join([f"{k}:{v}" for k, v in top_rc]) if top_rc else ""

        rows: List[Dict[str, Any]] = []
        for phase, xs in phase_days.items():
            m = mean(xs) if xs else float("nan")
            p90 = percentile(xs, 0.90) if xs else float("nan")
            share = (sum(xs) / total_days) if total_days > 0 else 0.0
            rows.append({
                "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                "phase": phase, "count": len(xs),
                "mean_days": "" if math.isnan(m) else round(m, 6),
                "p90_days": "" if math.isnan(p90) else round(p90, 6),
                "share_of_total_days": round(share, 6),
                "top_reason_codes": top_rc_str,
            })
        rows.sort(key=lambda r: float(r.get("share_of_total_days", 0.0)), reverse=True)
        return rows

    def _build_reason_breakdown_rows(self, scenario_name: str,
                                     episodes_per_run: int, runs: int,
                                     reason_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total_episodes = max(1, episodes_per_run * runs)
        agg: Dict[str, Dict[str, Any]] = {}
        for ev in reason_events:
            rc = str(ev.get("reason_code", ""))
            if not rc:
                continue
            a = agg.setdefault(rc, {"count": 0, "sb_count": 0, "complexities": [], "doc_burdens": [], "budget_scales": [], "cybers": []})
            a["count"] += 1
            if ev.get("is_small_business") == 1:
                a["sb_count"] += 1
            a["complexities"].append(float(ev.get("complexity", 0.0)))
            a["doc_burdens"].append(float(ev.get("doc_burden", 0.0)))
            a["budget_scales"].append(float(ev.get("budget_scale", 0.0)))
            a["cybers"].append(float(ev.get("cyber", 0.0)))

        rows: List[Dict[str, Any]] = []
        for rc, a in agg.items():
            count = int(a["count"])
            sb_share = (a["sb_count"] / count) if count else 0.0
            rows.append({
                "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                "reason_code": rc, "count": count,
                "rate_per_episode": round(count / total_episodes, 6),
                "sb_share": round(sb_share, 6),
                "avg_complexity": round(mean(a["complexities"]), 6) if a["complexities"] else "",
                "avg_doc_burden": round(mean(a["doc_burdens"]), 6) if a["doc_burdens"] else "",
                "avg_budget_scale": round(mean(a["budget_scales"]), 6) if a["budget_scales"] else "",
                "avg_cyber": round(mean(a["cybers"]), 6) if a["cybers"] else "",
            })
        rows.sort(key=lambda r: int(r.get("count", 0)), reverse=True)
        return rows

    # ---------- Sensitivity perturbations ----------

    def apply_perturbation(self, knobs: Dict[str, float], knob: str, delta: float, direction: int) -> Dict[str, float]:
        out = dict(knobs)
        if knob not in out:
            return out
        if knob == "base_award_days":
            out[knob] = clamp(out[knob] * (1.0 + direction * delta), 5.0, 365.0)
            return out
        out[knob] = clamp(out[knob] + direction * delta, 0.0, 1.0)
        return out

    # ---------- Artifacts ----------

    def artifact_text(self, kind: str, policy: PolicyConfig, episode: Episode, winner_vendor_id: str, outcome: Dict[str, Any]) -> str:
        verbosity = str(self.cfg["outputs"].get("artifact_verbosity", "short")).lower()
        doc = policy.get("documentation_burden", 0.5)
        wT, wC, wR = policy.normalized_weights()

        if kind == "solicitation":
            return (
                f"SOLICITATION: complexity={episode.complexity:.2f}, cyber={episode.cyber_sensitivity:.2f}, "
                f"budget_scale={episode.budget_scale:.2f}, export_cui={int(episode.export_cui_flag)}, "
                f"research_security={int(episode.research_security_flag)}. "
                f"Weights: tech={wT:.2f}, cost={wC:.2f}, risk={wR:.2f}."
            )

        if kind == "decision_memo":
            if outcome.get("reason_code") in (RC_NO_BIDS, RC_NO_AWARD):
                return f"DECISION MEMO: No award ({outcome.get('reason_code')}); doc_burden={doc:.2f}, complexity={episode.complexity:.2f}."
            tech = outcome.get("winner_tech", 0.0)
            cost = outcome.get("winner_cost", 0.0)
            risk = outcome.get("winner_risk", 0.0)
            days = outcome.get("award_days", 0.0)
            dq = outcome.get("dq_count", 0)
            drop = outcome.get("dropouts", 0)
            if verbosity == "medium":
                return (f"DECISION MEMO: Selected {winner_vendor_id} with tech={tech:.2f}, cost={cost:.2f}, risk={risk:.2f}; "
                        f"time≈{days:.1f}d; dropouts={drop}, dq={dq}; doc_burden={doc:.2f}.")
            return (f"DECISION MEMO: Selected {winner_vendor_id} with tech={tech:.2f}, cost={cost:.2f}, risk={risk:.2f}; "
                    f"time≈{days:.0f}d; dropouts={drop}, dq={dq}.")

        if kind == "policy_brief":
            effects = outcome.get("effects_summary", "")
            return f"POLICY BRIEF (synthetic): doc_burden={doc:.2f}; weights tech={wT:.2f}, cost={wC:.2f}, risk={wR:.2f}. {effects}"

        return f"{kind}: (no template)"

    # ---------- Main simulation ----------

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        meta = self.cfg["meta"]
        sim = self.cfg["simulation"]
        outputs = self.cfg["outputs"]

        scenario_name = str(meta.get("scenario_name", "scenario"))
        runs = int(sim.get("runs", 5))
        episodes_per_run = int(sim.get("episodes_per_run", 20))
        vendors_per_episode = sim.get("vendors_per_episode", [6, 20])

        sensitivity = sim.get("sensitivity", {})
        sens_enabled = bool(sensitivity.get("enabled", True))
        sens_delta = float(sensitivity.get("delta", 0.10))
        sens_knobs = list(sensitivity.get("knobs", []))

        policy_runs_rows: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []
        artifact_rows: List[Dict[str, Any]] = []
        run_summary_by_knob_rows: List[Dict[str, Any]] = []
        assumptions_rows: List[Dict[str, Any]] = []

        global_phase_days: Dict[str, List[float]] = {}
        global_reason_events: List[Dict[str, Any]] = []
        global_total_phase_days: float = 0.0

        knob_delta_collect: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

        phases = list(self.regime["phases"])

        for r in range(runs):
            run_id = f"run_{r:04d}"
            base_policy = self.sample_policy()
            assumptions_rows.extend(self._assumption_rows_for_run(scenario_name, run_id, base_policy))

            policy_variants: List[Tuple[str, PolicyConfig]] = [("baseline", base_policy)]
            if sens_enabled and sens_knobs:
                for k in sens_knobs:
                    p_plus = PolicyConfig(
                        policy_config_id=f"{base_policy.policy_config_id}__{k}__plus",
                        knobs=self.apply_perturbation(base_policy.knobs, k, sens_delta, +1),
                    )
                    p_minus = PolicyConfig(
                        policy_config_id=f"{base_policy.policy_config_id}__{k}__minus",
                        knobs=self.apply_perturbation(base_policy.knobs, k, sens_delta, -1),
                    )
                    policy_variants.append((f"{k}_plus", p_plus))
                    policy_variants.append((f"{k}_minus", p_minus))

            variant_summaries: Dict[str, Dict[str, float]] = {}

            for variant_tag, policy in policy_variants:
                total_awards = 0
                total_no_awards = 0
                total_dropouts = 0
                total_dq = 0
                total_bids = 0
                bids_per_episode: List[int] = []
                award_days_list: List[float] = []
                sb_wins = 0
                cost_list: List[float] = []
                risk_list: List[float] = []
                tech_list: List[float] = []
                admin_load_list: List[float] = []

                for e in range(episodes_per_run):
                    ep = self.sample_episode(run_id, e)
                    n_vendors = self.sample_int(vendors_per_episode)
                    vendors = self.sample_vendors(ep.episode_id, n_vendors)

                    step_idx = 0
                    admin_load = 0.0
                    dq_count = 0
                    drop_count = 0

                    expected_competitors = max(1, int(0.6 * n_vendors))
                    bidders: List[Vendor] = []
                    submitted: List[Tuple[Vendor, Budget]] = []
                    screened: List[Tuple[Vendor, Budget]] = []
                    scored: List[Tuple[Vendor, Budget, Dict[str, float]]] = []
                    winner_vendor_id = ""
                    outcome_reason = RC_OK
                    award_days_total = 0.0

                    reviewer_load, reviewer_strict = self.reviewer_state()

                    if bool(outputs.get("write_artifacts", True)):
                        artifact_rows.append({
                            "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                            "run_id": run_id, "episode_id": ep.episode_id,
                            "artifact_type": f"solicitation::{variant_tag}",
                            "text": self.artifact_text("solicitation", policy, ep, "", {}),
                            "ts_utc": now_iso(),
                            "meta_json": jdump({"policy_config_id": policy.policy_config_id, "variant": variant_tag}),
                        })

                    for phase in phases:
                        bids_in_play = max(1, len(submitted) if submitted else len(bidders) if bidders else 1)
                        days_added = self.phase_time_days(policy, ep, phase, bids_in_play=bids_in_play)
                        award_days_total += days_added

                        if variant_tag == "baseline":
                            global_total_phase_days += days_added
                            global_phase_days.setdefault(phase, []).append(days_added)

                        admin_load += days_added * (0.6 + 0.7 * policy.get("documentation_burden", 0.5))

                        trace_rows.append({
                            "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                            "run_id": run_id, "episode_id": ep.episode_id, "step_idx": step_idx,
                            "phase": phase, "actor": "system", "vendor_id": "", "is_small_business": "",
                            "action": "phase_start", "reason_code": RC_OK, "p_value": "",
                            "days_added": round(days_added, 6),
                            "constraints_json": jdump({"policy_config_id": policy.policy_config_id, "variant": variant_tag}),
                            "state_json": "", "score_json": "", "counterfactual_json": "",
                        })
                        step_idx += 1

                        if phase == "BID_DECISION":
                            bidders = []
                            for v in vendors:
                                will_bid, p_bid = self.vendor_bid_decision(policy, ep, v, expected_competitors)
                                if will_bid:
                                    bidders.append(v)
                                else:
                                    drop_count += 1
                                    total_dropouts += 1
                                    if variant_tag == "baseline":
                                        global_reason_events.append(self._reason_event(RC_DROP_OUT, policy, ep, v))
                                trace_rows.append({
                                    "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                                    "run_id": run_id, "episode_id": ep.episode_id, "step_idx": step_idx,
                                    "phase": phase, "actor": "vendor", "vendor_id": v.vendor_id,
                                    "is_small_business": int(v.is_small_business),
                                    "action": "bid" if will_bid else "no_bid",
                                    "reason_code": RC_OK if will_bid else RC_DROP_OUT,
                                    "p_value": round(p_bid, 6), "days_added": "",
                                    "constraints_json": "", "state_json": "", "score_json": "", "counterfactual_json": "",
                                })
                                step_idx += 1
                            if not bidders:
                                outcome_reason = RC_NO_BIDS
                                if variant_tag == "baseline":
                                    global_reason_events.append(self._reason_event(RC_NO_BIDS, policy, ep, None))
                                break

                        elif phase == "SUBMIT":
                            submitted = []
                            for v in bidders:
                                p_drop = self.dropout_probability(policy, ep, v)
                                dropped = (self.rng.random() < p_drop)
                                if dropped:
                                    drop_count += 1
                                    total_dropouts += 1
                                    if variant_tag == "baseline":
                                        global_reason_events.append(self._reason_event(RC_DROP_OUT, policy, ep, v))
                                    continue
                                budget = self.generate_budget(policy, ep, v)
                                submitted.append((v, budget))
                            if not submitted:
                                outcome_reason = RC_NO_BIDS
                                if variant_tag == "baseline":
                                    global_reason_events.append(self._reason_event(RC_NO_BIDS, policy, ep, None))
                                break

                        elif phase == "COMPLIANCE_SCREEN":
                            screened = []
                            for v, b in submitted:
                                passed, rc, _p_fail = self.compliance_screen(policy, ep, v, b)
                                if not passed:
                                    dq_count += 1
                                    total_dq += 1
                                    if variant_tag == "baseline":
                                        global_reason_events.append(self._reason_event(rc, policy, ep, v))
                                    continue
                                screened.append((v, b))
                            if not screened:
                                outcome_reason = RC_NO_AWARD
                                if variant_tag == "baseline":
                                    global_reason_events.append(self._reason_event(RC_NO_AWARD, policy, ep, None))
                                break

                        elif phase in ("TECH_REVIEW","COST_REALISM","RISK_REVIEW","MERIT_REVIEW","PITCH_REVIEW","TRANSITION_REVIEW","BUDGET_CHECK"):
                            if not scored and screened:
                                scored = []
                                for v, b in screened:
                                    s = self.score_vendor(policy, ep, v, b, reviewer_load, reviewer_strict)
                                    scored.append((v, b, s))

                        elif phase == "AWARD":
                            if not scored:
                                outcome_reason = RC_NO_AWARD
                                if variant_tag == "baseline":
                                    global_reason_events.append(self._reason_event(RC_NO_AWARD, policy, ep, None))
                                break
                            scored.sort(key=lambda t: t[2]["total"], reverse=True)
                            winner, _wb, ws = scored[0]
                            winner_vendor_id = winner.vendor_id

                            total_awards += 1
                            total_bids += len(screened)
                            bids_per_episode.append(len(screened))
                            award_days_list.append(award_days_total)
                            cost_list.append(ws["cost"])
                            risk_list.append(ws["risk"])
                            tech_list.append(ws["tech"])
                            admin_load_list.append(admin_load)
                            if winner.is_small_business:
                                sb_wins += 1

                            # episode memo
                            if bool(outputs.get("write_artifacts", True)):
                                artifact_rows.append({
                                    "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                                    "run_id": run_id, "episode_id": ep.episode_id,
                                    "artifact_type": f"decision_memo::{variant_tag}",
                                    "text": self.artifact_text("decision_memo", policy, ep, winner_vendor_id, {
                                        "reason_code": RC_OK,
                                        "award_days": award_days_total,
                                        "dq_count": dq_count,
                                        "dropouts": drop_count,
                                        "winner_tech": ws["tech"],
                                        "winner_cost": ws["cost"],
                                        "winner_risk": ws["risk"],
                                        "winner_total": ws["total"],
                                    }),
                                    "ts_utc": now_iso(),
                                    "meta_json": jdump({"policy_config_id": policy.policy_config_id, "variant": variant_tag, "winner": winner_vendor_id}),
                                })
                            break

                    if outcome_reason in (RC_NO_BIDS, RC_NO_AWARD) and not winner_vendor_id:
                        total_no_awards += 1
                        bids_per_episode.append(0)
                        admin_load_list.append(admin_load)
                        if bool(outputs.get("write_artifacts", True)):
                            artifact_rows.append({
                                "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                                "run_id": run_id, "episode_id": ep.episode_id,
                                "artifact_type": f"decision_memo::{variant_tag}",
                                "text": self.artifact_text("decision_memo", policy, ep, "", {
                                    "reason_code": outcome_reason,
                                    "award_days": award_days_total,
                                    "dq_count": dq_count,
                                    "dropouts": drop_count,
                                }),
                                "ts_utc": now_iso(),
                                "meta_json": jdump({"policy_config_id": policy.policy_config_id, "variant": variant_tag, "winner": ""}),
                            })

                avg_bids_ep = mean(bids_per_episode) if bids_per_episode else 0.0
                avg_bids_award = (total_bids / total_awards) if total_awards else 0.0
                avg_days = mean(award_days_list) if award_days_list else float("nan")

                variant_summaries[variant_tag] = {
                    "awards": float(total_awards),
                    "no_awards": float(total_no_awards),
                    "avg_award_days": float(avg_days) if not math.isnan(avg_days) else float("nan"),
                    "sb_win_rate": (sb_wins / total_awards) if total_awards else 0.0,
                    "avg_cost": mean(cost_list) if cost_list else float("nan"),
                    "avg_risk": mean(risk_list) if risk_list else float("nan"),
                    "avg_tech": mean(tech_list) if tech_list else float("nan"),
                    "avg_admin_load": mean(admin_load_list) if admin_load_list else float("nan"),
                    "dropouts": float(total_dropouts),
                    "dq_count": float(total_dq),
                    "avg_bids_per_episode": float(avg_bids_ep),
                    "avg_bids_per_award": float(avg_bids_award),
                }

                if variant_tag == "baseline":
                    policy_runs_rows.append({
                        "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                        "run_id": run_id, "policy_config_id": policy.policy_config_id, "ts_utc": now_iso(),
                        "episodes": episodes_per_run,
                        "awards": int(total_awards),
                        "no_awards": int(total_no_awards),
                        "avg_bids_per_episode": round(avg_bids_ep, 6),
                        "avg_bids_per_award": round(avg_bids_award, 6),
                        "dropouts": int(total_dropouts),
                        "dq_count": int(total_dq),
                        "avg_award_days": round(avg_days, 6) if not math.isnan(avg_days) else "",
                        "stdev_award_days": round(stdev(award_days_list), 6) if award_days_list else "",
                        "sb_win_rate": round((sb_wins / total_awards), 6) if total_awards else 0.0,
                        "avg_cost_score": round(mean(cost_list), 6) if cost_list else "",
                        "avg_risk_score": round(mean(risk_list), 6) if risk_list else "",
                        "avg_tech_score": round(mean(tech_list), 6) if tech_list else "",
                        "avg_admin_load": round(mean(admin_load_list), 6) if admin_load_list else "",
                        "policy_json": jdump(policy.knobs),
                        "effects_json": "",
                    })

            # Effects vs baseline + run_summary_by_knob
            baseline = variant_summaries.get("baseline", {})
            effects: Dict[str, Any] = {"baseline": baseline, "deltas": {}}

            for tag, summ in variant_summaries.items():
                if tag == "baseline":
                    continue
                deltas: Dict[str, Any] = {}
                for key in ("awards","no_awards","avg_award_days","sb_win_rate","avg_cost","avg_risk","avg_tech","avg_admin_load"):
                    b = baseline.get(key, 0.0)
                    s = summ.get(key, 0.0)
                    if (isinstance(b, float) and math.isnan(b)) or (isinstance(s, float) and math.isnan(s)):
                        deltas[key] = ""
                    else:
                        deltas[key] = s - b
                effects["deltas"][tag] = deltas

            for row in policy_runs_rows:
                if row["run_id"] == run_id:
                    row["effects_json"] = jdump(effects)
                    break

            base_id = base_policy.policy_config_id
            for knob in sens_knobs:
                for direction, tag in (("plus", f"{knob}_plus"), ("minus", f"{knob}_minus")):
                    d = effects["deltas"].get(tag, {})
                    if not d:
                        continue

                    key = (knob, direction)
                    knob_delta_collect.setdefault(key, {}).setdefault("avg_award_days", []).append(float(d.get("avg_award_days", 0.0)) if d.get("avg_award_days","") != "" else 0.0)
                    knob_delta_collect.setdefault(key, {}).setdefault("sb_win_rate", []).append(float(d.get("sb_win_rate", 0.0)) if d.get("sb_win_rate","") != "" else 0.0)

                    run_summary_by_knob_rows.append({
                        "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                        "run_id": run_id, "baseline_policy_config_id": base_id,
                        "knob": knob, "direction": direction, "delta": sens_delta,
                        "delta_awards": d.get("awards",""),
                        "delta_no_awards": d.get("no_awards",""),
                        "delta_avg_award_days": d.get("avg_award_days",""),
                        "delta_sb_win_rate": d.get("sb_win_rate",""),
                        "delta_avg_cost": d.get("avg_cost",""),
                        "delta_avg_risk": d.get("avg_risk",""),
                        "delta_avg_tech": d.get("avg_tech",""),
                        "delta_avg_admin_load": d.get("avg_admin_load",""),
                        "ci95_low_avg_award_days": "",
                        "ci95_high_avg_award_days": "",
                        "ci95_low_sb_win_rate": "",
                        "ci95_high_sb_win_rate": "",
                    })

            # Run-level brief artifact (baseline policy)
            if bool(outputs.get("write_artifacts", True)):
                eff_summary_parts: List[str] = []
                for k in sens_knobs[:3]:
                    plus = effects["deltas"].get(f"{k}_plus", {})
                    if plus and plus.get("sb_win_rate","") != "" and plus.get("avg_award_days","") != "":
                        eff_summary_parts.append(f"{k}+ -> ΔSB={plus['sb_win_rate']:+.3f}, Δdays={plus['avg_award_days']:+.1f}")
                eff_summary = "Sensitivity: " + "; ".join(eff_summary_parts) if eff_summary_parts else ""
                dummy_ep = Episode(run_id=run_id, episode_id="", complexity=0.0, cyber_sensitivity=0.0, budget_scale=0.0,
                                   competition_heat=0.0, export_cui_flag=False, research_security_flag=False)
                artifact_rows.append({
                    "version": VERSION, "scenario": scenario_name, "regime": self.regime_name,
                    "run_id": run_id, "episode_id": "",
                    "artifact_type": "policy_brief::baseline",
                    "text": self.artifact_text("policy_brief", base_policy, dummy_ep, "", {"effects_summary": eff_summary}),
                    "ts_utc": now_iso(),
                    "meta_json": jdump({"policy_config_id": base_policy.policy_config_id}),
                })

        # Fill CI bands across runs into run_summary_by_knob rows
        ci_map: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
        for (knob, direction), series in knob_delta_collect.items():
            ci_days = ci95_normal(series.get("avg_award_days", []))
            ci_sb = ci95_normal(series.get("sb_win_rate", []))
            ci_map[(knob, direction)] = {"avg_award_days": ci_days, "sb_win_rate": ci_sb}

        for row in run_summary_by_knob_rows:
            k = row["knob"]; d = row["direction"]
            cm = ci_map.get((k, d), {})
            if "avg_award_days" in cm:
                lo, hi = cm["avg_award_days"]
                row["ci95_low_avg_award_days"] = "" if math.isnan(lo) else round(lo, 6)
                row["ci95_high_avg_award_days"] = "" if math.isnan(hi) else round(hi, 6)
            if "sb_win_rate" in cm:
                lo, hi = cm["sb_win_rate"]
                row["ci95_low_sb_win_rate"] = "" if math.isnan(lo) else round(lo, 6)
                row["ci95_high_sb_win_rate"] = "" if math.isnan(hi) else round(hi, 6)

        phase_bottlenecks_rows = self._build_phase_bottlenecks_rows(
            scenario_name=scenario_name,
            phase_days=global_phase_days,
            total_days=global_total_phase_days,
            reason_events=global_reason_events,
        )
        reason_breakdown_rows = self._build_reason_breakdown_rows(
            scenario_name=scenario_name,
            episodes_per_run=int(self.cfg["simulation"].get("episodes_per_run", 1)),
            runs=int(self.cfg["simulation"].get("runs", 1)),
            reason_events=global_reason_events,
        )

        return {
            "policy_runs": policy_runs_rows,
            "decision_traces": trace_rows,
            "artifacts": artifact_rows,
            "run_summary_by_knob": run_summary_by_knob_rows,
            "phase_bottlenecks": phase_bottlenecks_rows,
            "reason_code_breakdown": reason_breakdown_rows,
            "assumptions": assumptions_rows,
        }


# =============================
# CSV writing (stable schemas)
# =============================

def write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in header}
            w.writerow(out)


# =============================
# Main (no args required)
# =============================

def main() -> None:
    cfg = build_cfg_from_user_config(CONFIG)

    out_dir = str(cfg["outputs"]["out_dir"])
    ensure_dir(out_dir)

    sim = PolicySimulatorV1_0(cfg)
    results = sim.run()

    write_csv(os.path.join(out_dir, "policy_runs.csv"), results["policy_runs"], POLICY_RUNS_HEADER)
    write_csv(os.path.join(out_dir, "decision_traces.csv"), results["decision_traces"], DECISION_TRACES_HEADER)
    write_csv(os.path.join(out_dir, "artifacts.csv"), results["artifacts"], ARTIFACTS_HEADER)

    write_csv(os.path.join(out_dir, "run_summary_by_knob.csv"), results["run_summary_by_knob"], RUN_SUMMARY_BY_KNOB_HEADER)
    write_csv(os.path.join(out_dir, "phase_bottlenecks.csv"), results["phase_bottlenecks"], PHASE_BOTTLENECKS_HEADER)
    write_csv(os.path.join(out_dir, "reason_code_breakdown.csv"), results["reason_code_breakdown"], REASON_BREAKDOWN_HEADER)
    write_csv(os.path.join(out_dir, "assumptions.csv"), results["assumptions"], ASSUMPTIONS_HEADER)

    # Save the exact effective config
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[OK] v{VERSION} outputs written to ./{out_dir}/")
    print(f"[OK] Regime: {cfg['regime']['name']} | runs={cfg['simulation']['runs']} | episodes/run={cfg['simulation']['episodes_per_run']} | vendors/ep={cfg['simulation']['vendors_per_episode']}")


if __name__ == "__main__":
    main()
