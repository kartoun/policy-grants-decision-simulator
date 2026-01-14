"""
Advanced analysis for Policy / Grants Decision Simulator (v1.0)

Input:  ./output/*.csv produced by the engine
Output: ./figures/*.png and ./figures/REPORT.md

Run:
  python analyze_results.py

Notes
-----
• This script uses ONLY the CSV outputs; no engine changes.
• Dropout (“no-bid”) is split into *interpretable inferred buckets* using the
  logged p_value (= p_bid) from BID_DECISION traces:
    - DROP_OUT_LOW_PBID      : p_bid < 0.20  (very unlikely to bid)
    - DROP_OUT_MID_PBID      : 0.20 ≤ p_bid < 0.50 (borderline)
    - DROP_OUT_HIGH_PBID     : p_bid ≥ 0.50 (surprising no-bid; often capacity/burden/other unmodeled factors)
  This is an inference layer for reporting; it does not claim causal truth.
"""

import os
import math
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "output"
FIG_DIR = "figures"


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def _savefig(name: str) -> str:
    out = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out

def _fmt(x, digits=3):
    if x is None:
        return "NA"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "NA"
    except Exception:
        pass
    return f"{x:.{digits}f}"

def _quantiles(series: pd.Series, qs=(0.1, 0.5, 0.9)) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {q: None for q in qs}
    return {q: float(s.quantile(q)) for q in qs}

def _short_knob(knob: str) -> str:
    return (knob
            .replace("documentation_burden", "doc_burden")
            .replace("oem_direct_adoption", "oem_direct")
            .replace("sb_preference", "sb_pref")
            .replace("export_cui_strictness", "export_cui")
            .replace("research_security_strictness", "research_sec")
            .replace("_", " "))

def _wrap_label(knob: str, direction: str) -> str:
    d = "+" if direction == "plus" else "-"
    return f"{_short_knob(knob)} ({d})"

def _as_num(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Load data
# ----------------------------

def load_all():
    policy_runs = _read_csv(os.path.join(OUTPUT_DIR, "policy_runs.csv"))
    bottlenecks = _read_csv(os.path.join(OUTPUT_DIR, "phase_bottlenecks.csv"))
    run_knobs = _read_csv(os.path.join(OUTPUT_DIR, "run_summary_by_knob.csv"))
    assumptions = _read_csv(os.path.join(OUTPUT_DIR, "assumptions.csv"))
    traces = _read_csv(os.path.join(OUTPUT_DIR, "decision_traces.csv"))
    return policy_runs, bottlenecks, run_knobs, assumptions, traces


# ----------------------------
# Figures: run-level distributions + tradeoffs
# ----------------------------

def fig_overview_distributions(policy_runs: pd.DataFrame):
    paths = []

    policy_runs = _as_num(policy_runs, ["avg_award_days", "sb_win_rate", "avg_admin_load", "awards", "no_awards"])

    plt.figure()
    vals = policy_runs["avg_award_days"].dropna()
    plt.hist(vals, bins=20)
    plt.xlabel("Average Award Days (per run)")
    plt.ylabel("Number of runs")
    plt.title("Distribution: Time-to-Award (run averages)")
    paths.append(_savefig("fig_01_award_days_hist.png"))

    plt.figure()
    vals = policy_runs["sb_win_rate"].dropna()
    plt.hist(vals, bins=20)
    plt.xlabel("Small Business Win Rate (per run)")
    plt.ylabel("Number of runs")
    plt.title("Distribution: Small Business Win Rate (run averages)")
    paths.append(_savefig("fig_02_sb_win_hist.png"))

    if "avg_admin_load" in policy_runs.columns:
        plt.figure()
        vals = policy_runs["avg_admin_load"].dropna()
        plt.hist(vals, bins=20)
        plt.xlabel("Administrative Load (proxy, per run)")
        plt.ylabel("Number of runs")
        plt.title("Distribution: Administrative Load (run averages)")
        paths.append(_savefig("fig_03_admin_load_hist.png"))

    return paths


def fig_tradeoffs(policy_runs: pd.DataFrame):
    paths = []
    policy_runs = _as_num(policy_runs, ["avg_award_days", "sb_win_rate", "avg_admin_load"])

    plt.figure()
    m = policy_runs[["avg_award_days", "sb_win_rate"]].dropna()
    if not m.empty:
        plt.scatter(m["avg_award_days"], m["sb_win_rate"])
    plt.xlabel("Average Award Days")
    plt.ylabel("Small Business Win Rate")
    plt.title("Tradeoff: Time-to-Award vs Small Business Win Rate")
    paths.append(_savefig("fig_04_time_vs_sb.png"))

    if "avg_admin_load" in policy_runs.columns:
        plt.figure()
        m = policy_runs[["avg_award_days", "avg_admin_load"]].dropna()
        if not m.empty:
            plt.scatter(m["avg_award_days"], m["avg_admin_load"])
        plt.xlabel("Average Award Days")
        plt.ylabel("Average Administrative Load (proxy)")
        plt.title("Tradeoff: Time-to-Award vs Administrative Load")
        paths.append(_savefig("fig_05_time_vs_admin.png"))

    return paths


# ----------------------------
# Figures: bottlenecks
# ----------------------------

def fig_bottlenecks(bottlenecks: pd.DataFrame):
    paths = []
    bottlenecks = bottlenecks.copy()
    bottlenecks = _as_num(bottlenecks, ["share_of_total_days", "mean_days", "p90_days"])

    b = bottlenecks.dropna(subset=["share_of_total_days"]).sort_values("share_of_total_days", ascending=True)

    plt.figure()
    plt.barh(b["phase"], b["share_of_total_days"])
    plt.xlabel("Share of Total Time")
    plt.title("Process Bottlenecks: Share of Total Time by Phase")
    paths.append(_savefig("fig_06_phase_time_share.png"))

    b2 = bottlenecks.dropna(subset=["share_of_total_days"]).sort_values("share_of_total_days", ascending=False)
    shares = b2["share_of_total_days"].values
    cum = shares.cumsum()
    plt.figure()
    plt.plot(range(1, len(cum) + 1), cum)
    plt.ylim(0, 1.02)
    plt.xlabel("Top N Phases (sorted by time share)")
    plt.ylabel("Cumulative Share of Total Time")
    plt.title("Pareto View: How Many Phases Explain Most of the Time?")
    paths.append(_savefig("fig_07_phase_pareto.png"))

    return paths


# ----------------------------
# Reason metrics (interpretable denominators + dropout buckets)
# ----------------------------

def compute_reason_metrics_from_traces(traces: pd.DataFrame) -> pd.DataFrame:
    """
    Produces an interpretable reason table with explicit denominators:

    • DROP_OUT_* buckets: per vendor decision (BID_DECISION, actor=vendor)
      - inferred via p_value (= p_bid), see top-of-file comment.

    • DQ_* (if present): per submission attempt (best-effort using SUBMIT rows)
      NOTE: current engine may not log explicit submit actions; this table will
            still produce correct dropout + episode-level rates.

    • NO_BIDS / NO_AWARD: per episode (episode-level outcomes are inferred from traces)
    """
    tr = traces.copy()

    for col in ["phase", "actor", "action", "reason_code", "episode_id", "vendor_id", "is_small_business", "p_value"]:
        if col not in tr.columns:
            tr[col] = ""

    # Episodes
    episodes = tr["episode_id"].dropna().astype(str)
    episodes = episodes[episodes.str.len() > 0].unique().tolist()
    n_episodes = max(1, len(episodes))

    # Vendor bid decisions
    bid_decision_mask = (tr["phase"] == "BID_DECISION") & (tr["actor"] == "vendor")
    bid_decisions = tr[bid_decision_mask].copy()
    bid_decisions["p_value"] = pd.to_numeric(bid_decisions["p_value"], errors="coerce")

    n_vendor_decisions = int(len(bid_decisions))

    # Identify no-bid events
    no_bid_mask = (bid_decisions["action"] == "no_bid") | (bid_decisions["reason_code"] == "DROP_OUT")
    no_bids = bid_decisions[no_bid_mask].copy()

    # Bucket by p_bid
    def bucket(p):
        if pd.isna(p):
            return "DROP_OUT_PBID_UNKNOWN"
        if p < 0.20:
            return "DROP_OUT_LOW_PBID"
        if p < 0.50:
            return "DROP_OUT_MID_PBID"
        return "DROP_OUT_HIGH_PBID"

    no_bids["dropout_bucket"] = no_bids["p_value"].apply(bucket)
    bucket_counts = no_bids["dropout_bucket"].value_counts().to_dict()

    # Submission attempts (best-effort)
    # Current engine may not log vendor submit actions; keep robust:
    submit_attempt_mask = (tr["phase"] == "SUBMIT") & (tr["actor"] == "vendor")
    n_submit_attempts = int(submit_attempt_mask.sum())

    # DQ events (best-effort)
    dq_mask = (tr["action"] == "disqualify") | (tr["reason_code"].astype(str).str.startswith("DQ_"))
    dq_events = tr[dq_mask].copy()
    dq_events["reason_code"] = dq_events["reason_code"].astype(str)
    dq_counts = dq_events["reason_code"].value_counts().to_dict()

    # Episode-level outcomes:
    # If your traces do not have a dedicated END phase, infer:
    # - NO_BIDS: episode had zero "bid" actions
    # - NO_AWARD: episode had bids but no AWARD winner signal
    #
    # Current engine logs vendor BID_DECISION actions (bid/no_bid), and system phase starts.
    # It does not explicitly log "award/no_award" as END rows, so infer from vendor bids only.
    #
    # Conservative:
    # - NO_BIDS: count episodes with 0 bids
    # - NO_AWARD: count episodes with >=1 bid but 0 AWARD phase encountered
    #
    # This is an *inference* suitable for summaries.
    bid_actions = tr[(tr["phase"] == "BID_DECISION") & (tr["actor"] == "vendor")].copy()
    bids = bid_actions[bid_actions["action"] == "bid"]
    bid_counts_by_ep = bids.groupby("episode_id").size().to_dict()

    # AWARD phase encountered (system phase_start)
    award_phase = tr[(tr["phase"] == "AWARD") & (tr["actor"] == "system")]
    award_eps = set(award_phase["episode_id"].astype(str).tolist())

    no_bids_eps = 0
    no_award_eps = 0
    for ep in episodes:
        bcnt = int(bid_counts_by_ep.get(ep, 0))
        if bcnt == 0:
            no_bids_eps += 1
        else:
            # If AWARD phase was reached, treat as "award path exists"
            if ep not in award_eps:
                no_award_eps += 1

    # Build table
    rows = []

    # Dropout buckets (per vendor decision)
    for rc, cnt in sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "reason_code": rc,
            "count": int(cnt),
            "rate": (int(cnt) / n_vendor_decisions) if n_vendor_decisions else float("nan"),
            "denominator": "per_vendor_decision",
            "denom_n": n_vendor_decisions,
        })

    # DQs (per submission attempt, best-effort)
    for rc, cnt in sorted(dq_counts.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "reason_code": rc,
            "count": int(cnt),
            "rate": (int(cnt) / n_submit_attempts) if n_submit_attempts else float("nan"),
            "denominator": "per_submission_attempt",
            "denom_n": n_submit_attempts,
        })

    # Episode-level
    rows.append({
        "reason_code": "NO_BIDS",
        "count": int(no_bids_eps),
        "rate": (int(no_bids_eps) / n_episodes) if n_episodes else float("nan"),
        "denominator": "per_episode",
        "denom_n": n_episodes,
    })
    rows.append({
        "reason_code": "NO_AWARD",
        "count": int(no_award_eps),
        "rate": (int(no_award_eps) / n_episodes) if n_episodes else float("nan"),
        "denominator": "per_episode",
        "denom_n": n_episodes,
    })

    out = pd.DataFrame(rows)
    return out


def fig_reason_codes(traces: pd.DataFrame):
    """
    Final, analyst-grade attrition / disqualification visualization.

    Improvements applied:
    • Reasons are split by denominator (NO mixing on one axis)
    • Zero-rate reasons are removed
    • Counts are annotated explicitly
    • Linear scales only (rates should not be log-scaled)
    """

    paths = []
    reasons = compute_reason_metrics_from_traces(traces)
    reasons = _as_num(reasons, ["count", "rate", "denom_n"])

    # Drop zero / NaN rates
    reasons = reasons[(reasons["rate"] > 0) & (~reasons["rate"].isna())]

    # ---- PANEL 1: per-vendor decision (dropout behavior)
    rv = reasons[reasons["denominator"] == "per_vendor_decision"]
    rv = rv.sort_values("rate", ascending=True)

    if not rv.empty:
        plt.figure(figsize=(9, 4))
        plt.barh(rv["reason_code"], rv["rate"])
        plt.xlabel("Rate per vendor decision")
        plt.title("Vendor Dropout Behavior (by inferred p_bid bucket)")
        plt.xlim(0, min(1.0, rv["rate"].max() * 1.1))

        for i, r in rv.iterrows():
            plt.text(
                r["rate"] + 0.01,
                rv.index.get_loc(i),
                f"n={int(r['count'])}",
                va="center",
                fontsize=9
            )

        paths.append(_savefig("fig_08a_vendor_dropout_rates.png"))

    # ---- PANEL 2: per-episode outcomes
    re = reasons[reasons["denominator"] == "per_episode"]
    re = re.sort_values("rate", ascending=True)

    if not re.empty:
        plt.figure(figsize=(9, 4))
        plt.barh(re["reason_code"], re["rate"])
        plt.xlabel("Rate per episode")
        plt.title("Episode-Level Outcomes")
        plt.xlim(0, min(1.0, re["rate"].max() * 1.2))

        for i, r in re.iterrows():
            plt.text(
                r["rate"] + 0.005,
                re.index.get_loc(i),
                f"n={int(r['count'])}",
                va="center",
                fontsize=9
            )

        paths.append(_savefig("fig_08b_episode_outcome_rates.png"))

    # ---- COUNTS (log scale retained, but now secondary)
    rc = reasons.sort_values("count", ascending=True)

    plt.figure(figsize=(9, 4))
    plt.barh(rc["reason_code"], rc["count"])
    plt.xscale("log")
    plt.xlabel("Count (log scale)")
    plt.title("Attrition / Disqualification Reasons (event counts)")
    paths.append(_savefig("fig_09_reason_counts_log.png"))

    return paths, reasons



# ----------------------------
# Sensitivity figures (clean, grouped, readable)
# ----------------------------

def summarize_sensitivity(run_knobs: pd.DataFrame) -> pd.DataFrame:
    rk = run_knobs.copy()
    rk = _as_num(rk, [
        "delta_avg_award_days", "delta_sb_win_rate",
        "ci95_low_avg_award_days", "ci95_high_avg_award_days",
        "ci95_low_sb_win_rate", "ci95_high_sb_win_rate"
    ])

    grp = rk.groupby(["knob", "direction"], dropna=True).agg(
        mean_delta_days=("delta_avg_award_days", "mean"),
        mean_delta_sb=("delta_sb_win_rate", "mean"),
        ci_lo_days=("ci95_low_avg_award_days", "first"),
        ci_hi_days=("ci95_high_avg_award_days", "first"),
        ci_lo_sb=("ci95_low_sb_win_rate", "first"),
        ci_hi_sb=("ci95_high_sb_win_rate", "first"),
        n=("delta_avg_award_days", "count"),
    ).reset_index()

    grp["label"] = grp.apply(lambda r: _wrap_label(str(r["knob"]), str(r["direction"])), axis=1)
    return grp


def fig_sensitivity(grp: pd.DataFrame):
    paths = []

    # --- Time-to-award deltas
    g = grp.copy()
    g = g.dropna(subset=["mean_delta_days"])
    g = g.sort_values("mean_delta_days", key=lambda s: s.abs(), ascending=True)

    y = g["label"].tolist()
    x = g["mean_delta_days"].tolist()
    lo = g["ci_lo_days"].tolist()
    hi = g["ci_hi_days"].tolist()

    xerr = []
    for i in range(len(x)):
        if pd.isna(lo[i]) or pd.isna(hi[i]) or pd.isna(x[i]):
            xerr.append((0.0, 0.0))
        else:
            xerr.append((abs(x[i] - lo[i]), abs(hi[i] - x[i])))

    plt.figure(figsize=(10, 5))
    plt.barh(y, x)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Δ Avg Award Days (vs baseline)")
    plt.title("Sensitivity: Change in Time-to-Award by Policy Knob (mean ± CI)")
    for i in range(len(x)):
        plt.errorbar(x[i], i, xerr=[[xerr[i][0]], [xerr[i][1]]], fmt="none", capsize=4)
    paths.append(_savefig("fig_10_sensitivity_delta_days_clean.png"))

    # --- SB win rate deltas
    g = grp.copy()
    g = g.dropna(subset=["mean_delta_sb"])
    g = g.sort_values("mean_delta_sb", key=lambda s: s.abs(), ascending=True)

    y = g["label"].tolist()
    x = g["mean_delta_sb"].tolist()
    lo = g["ci_lo_sb"].tolist()
    hi = g["ci_hi_sb"].tolist()

    xerr = []
    for i in range(len(x)):
        if pd.isna(lo[i]) or pd.isna(hi[i]) or pd.isna(x[i]):
            xerr.append((0.0, 0.0))
        else:
            xerr.append((abs(x[i] - lo[i]), abs(hi[i] - x[i])))

    plt.figure(figsize=(10, 5))
    plt.barh(y, x)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Δ Small Business Win Rate (vs baseline)")
    plt.title("Sensitivity: Change in Small Business Win Rate by Policy Knob (mean ± CI)")
    for i in range(len(x)):
        plt.errorbar(x[i], i, xerr=[[xerr[i][0]], [xerr[i][1]]], fmt="none", capsize=4)
    paths.append(_savefig("fig_11_sensitivity_delta_sb_clean.png"))

    return paths


# ----------------------------
# Report
# ----------------------------

def write_report(policy_runs, bottlenecks, reason_metrics, sens_grp, fig_paths):
    policy_runs = _as_num(policy_runs, ["avg_award_days", "sb_win_rate", "avg_admin_load", "awards", "no_awards"])

    q_award = _quantiles(policy_runs.get("avg_award_days", pd.Series(dtype=float)))
    q_sb = _quantiles(policy_runs.get("sb_win_rate", pd.Series(dtype=float)))
    q_admin = _quantiles(policy_runs.get("avg_admin_load", pd.Series(dtype=float))) if "avg_admin_load" in policy_runs.columns else {0.1: None, 0.5: None, 0.9: None}

    b = bottlenecks.copy()
    b = _as_num(b, ["share_of_total_days"])
    b = b.dropna(subset=["share_of_total_days"]).sort_values("share_of_total_days", ascending=False).head(6)

    rm = reason_metrics.copy()
    rm = _as_num(rm, ["rate", "count"])
    rm_top = rm.dropna(subset=["rate"]).sort_values("rate", ascending=False).head(10)

    sg = sens_grp.copy()
    sg = _as_num(sg, ["mean_delta_days", "mean_delta_sb"])
    sg_time = sg.dropna(subset=["mean_delta_days"]).sort_values("mean_delta_days", key=lambda s: s.abs(), ascending=False).head(6)
    sg_sb = sg.dropna(subset=["mean_delta_sb"]).sort_values("mean_delta_sb", key=lambda s: s.abs(), ascending=False).head(6)

    lines = []
    lines.append("# Policy / Grants Decision Simulator — Analyst Summary\n\n")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    lines.append("## Run-level distributions\n")
    lines.append(f"- Avg award days (P10/P50/P90): {_fmt(q_award[0.1],1)} / {_fmt(q_award[0.5],1)} / {_fmt(q_award[0.9],1)}\n")
    lines.append(f"- SB win rate (P10/P50/P90): {_fmt(q_sb[0.1],3)} / {_fmt(q_sb[0.5],3)} / {_fmt(q_sb[0.9],3)}\n")
    if q_admin[0.5] is not None:
        lines.append(f"- Admin load (P10/P50/P90): {_fmt(q_admin[0.1],1)} / {_fmt(q_admin[0.5],1)} / {_fmt(q_admin[0.9],1)}\n")
    lines.append("\n")

    lines.append("## Top process bottlenecks (baseline phase share)\n")
    for _, row in b.iterrows():
        lines.append(f"- {row['phase']}: share={_fmt(row['share_of_total_days'],3)}\n")
    lines.append("\n")

    lines.append("## Top attrition / disqualification reasons (interpretable rates)\n")
    lines.append("Negative deltas are not used here; all rates are non-negative. Denominator is shown per reason.\n")
    for _, row in rm_top.iterrows():
        lines.append(f"- {row['reason_code']}: rate={_fmt(row['rate'],4)} ({row['denominator']}, n={int(row['denom_n'])}), count={int(row['count'])}\n")
    lines.append("\n")

    lines.append("## Sensitivity: biggest levers\n")
    lines.append("Negative Δ days indicates *faster* awards vs baseline.\n\n")
    lines.append("### Time-to-award (|Δ days|)\n")
    for _, row in sg_time.iterrows():
        lines.append(f"- {row['label']}: Δdays={_fmt(row['mean_delta_days'],2)}\n")
    lines.append("\n### Small business win rate (|Δ|)\n")
    for _, row in sg_sb.iterrows():
        lines.append(f"- {row['label']}: ΔSB={_fmt(row['mean_delta_sb'],4)}\n")
    lines.append("\n")

    lines.append("## Figures\n")
    for p in fig_paths:
        lines.append(f"- {os.path.basename(p)}\n")

    report_path = os.path.join(FIG_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return report_path


# ----------------------------
# Main
# ----------------------------

def main():
    _ensure_dir(FIG_DIR)

    policy_runs, bottlenecks, run_knobs, assumptions, traces = load_all()

    fig_paths = []
    fig_paths += fig_overview_distributions(policy_runs)
    fig_paths += fig_tradeoffs(policy_runs)
    fig_paths += fig_bottlenecks(bottlenecks)

    # Reasons (improved: dropout buckets by p_bid)
    reason_figs, reason_metrics = fig_reason_codes(traces)
    fig_paths += reason_figs

    # Sensitivity (clean)
    sens_grp = summarize_sensitivity(run_knobs)
    fig_paths += fig_sensitivity(sens_grp)

    # Report
    report_path = write_report(policy_runs, bottlenecks, reason_metrics, sens_grp, fig_paths)

    print("[OK] Figures written to:", os.path.abspath(FIG_DIR))
    print("[OK] Report written to:", os.path.abspath(report_path))
    print("[NOTE] DROP_OUT is bucketed using p_value (p_bid) from BID_DECISION traces (inference for reporting).")


if __name__ == "__main__":
    main()
