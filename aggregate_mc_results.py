"""
aggregate_mc_results.py
========================
Aggregates the 200 JSON files from the SLURM array MC study into
summary tables for the paper.

Usage:
    python3 aggregate_mc_results.py

Outputs (saved to ./mc_results/):
    scenario_A_summary.txt    Bias/RMSE/coverage table — Scenario A
    scenario_B_summary.txt    Bias/RMSE/coverage table — Scenario B
    scenario_C_summary.txt    Bias/RMSE/coverage table — Scenario C
    ALL_SCENARIOS_summary.txt Combined summary for paper Table 2
    mc_results.csv            Full results for all replications (for R/Python)

Authors: J.P. Arcede (Caraga State University)
Version: 1.0  |  April 2026
"""

import os
import json
import numpy as np
import pandas as pd

RESULTS_DIR = "./mc_results"
SCENARIOS   = ["A", "B", "C"]
SCENARIO_LABELS = {
    "A": "Scenario A: Hinatuan-type (85/15 shallow/non-shallow, T=90d)",
    "B": "Scenario B: Davao-type (75/25 shallow/non-shallow, T=90d)",
    "C": "Scenario C: Balanced (60/40 shallow/non-shallow, T=90d)",
}

PARAMS = ["mu", "beta", "Ksh", "Kns", "ash", "ans"]
PARAM_LABELS = {
    "mu":   "mu (background rate)",
    "beta": "beta (decay rate)",
    "Ksh":  "K_sh (shallow productivity)",
    "Kns":  "K_ns (non-shallow productivity)",
    "ash":  "alpha_sh (shallow mag scaling)",
    "ans":  "alpha_ns (non-shallow mag scaling)",
}


def load_scenario(scenario):
    folder = os.path.join(RESULTS_DIR, f"scenario_{scenario}")
    if not os.path.exists(folder):
        print(f"  WARNING: {folder} not found")
        return []
    records = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname)) as f:
                records.append(json.load(f))
    return records


def compute_summary(records, scenario):
    ok = [r for r in records if r.get("status") == "ok"]
    n_total = len(records)
    n_ok    = len(ok)

    if n_ok == 0:
        return None

    # Event counts
    ns      = np.array([r["n"]    for r in ok])
    n_sh    = np.array([r["n_sh"] for r in ok])
    n_ns    = np.array([r["n_ns"] for r in ok])

    # True parameters
    true_vals = {
        "mu":   ok[0]["true_mu"],
        "beta": ok[0]["true_beta"],
        "Ksh":  ok[0]["true_Ksh"],
        "Kns":  ok[0]["true_Kns"],
        "ash":  ok[0]["true_ash"],
        "ans":  ok[0]["true_ans"],
    }
    true_R = ok[0]["true_R"]

    # Parameter estimates
    est = {
        "mu":   np.array([r["mu_hat"]   for r in ok]),
        "beta": np.array([r["beta_hat"] for r in ok]),
        "Ksh":  np.array([r["Ksh_hat"]  for r in ok]),
        "Kns":  np.array([r["Kns_hat"]  for r in ok]),
        "ash":  np.array([r["ash_hat"]  for r in ok]),
        "ans":  np.array([r["ans_hat"]  for r in ok]),
    }

    # R estimates
    R_vals = np.array([r["R_hat"] for r in ok if r.get("R_hat") is not None])
    R_covered = np.array([r["R_covered"] for r in ok])

    # LRT
    lrt_ps  = np.array([r["lrt_p"] for r in ok if r.get("lrt_p") is not None])
    lrt_power = float((lrt_ps < 0.05).mean()) if len(lrt_ps) > 0 else np.nan

    # AIC comparison
    aic2 = np.array([r["aic2"] for r in ok if r.get("aic2") is not None])
    aic1 = np.array([r["aic1"] for r in ok if r.get("aic1") is not None])
    if len(aic2) > 0 and len(aic1) > 0:
        min_len = min(len(aic2), len(aic1))
        aic_2zone_wins = float((aic2[:min_len] < aic1[:min_len]).mean())
    else:
        aic_2zone_wins = np.nan

    # KS pass rate
    ks_ps     = np.array([r["ks_p"] for r in ok if r.get("ks_p") is not None])
    ks_pass   = float((ks_ps > 0.05).mean()) if len(ks_ps) > 0 else np.nan

    lines = []
    lines.append("=" * 72)
    lines.append(SCENARIO_LABELS[scenario])
    lines.append("=" * 72)
    lines.append(f"Successful fits: {n_ok}/{n_total}")
    lines.append(f"Average events:  total={ns.mean():.1f}  "
                 f"shallow={n_sh.mean():.1f}  non-shallow={n_ns.mean():.1f}")
    lines.append(f"True R = K_sh/K_ns = {true_R:.3f}")
    lines.append("")
    lines.append("DMDHP-2zone: BIAS / RMSE / COVERAGE")
    lines.append(f"{'Param':<12} {'True':>10} {'MeanHat':>10} "
                 f"{'Bias':>10} {'RMSE':>10} {'Cov(95%)':>10}")
    lines.append("-" * 65)
    for p in PARAMS:
        tv   = true_vals[p]
        hats = est[p]
        bias = float(np.mean(hats - tv))
        rmse = float(np.sqrt(np.mean((hats - tv) ** 2)))
        lines.append(f"{p:<12} {tv:>10.4f} {np.mean(hats):>10.4f} "
                     f"{bias:>10.4f} {rmse:>10.4f}  {'—':>9}")
    lines.append("")

    # R summary
    if len(R_vals) > 0:
        lines.append(f"R = K_sh/K_ns:  true={true_R:.3f}  "
                     f"mean={R_vals.mean():.3f}  "
                     f"median={np.median(R_vals):.3f}  "
                     f"coverage={R_covered.mean():.3f}")
    lines.append("")
    lines.append(f"LRT power (p<0.05):        {lrt_power:.3f}")
    lines.append(f"AIC win rate (2-zone < MDHP): {aic_2zone_wins:.3f}")
    lines.append(f"KS pass rate (p>0.05):     {ks_pass:.3f}")
    lines.append(f"Bootstrap refits (avg):    "
                 f"{np.mean([r['n_boot_ok'] for r in ok]):.1f}/{50}")
    lines.append("=" * 72)

    return "\n".join(lines), {
        "scenario": scenario,
        "n_ok": n_ok, "n_total": n_total,
        "avg_n": float(ns.mean()),
        "avg_n_sh": float(n_sh.mean()),
        "avg_n_ns": float(n_ns.mean()),
        "true_R": true_R,
        "R_mean": float(R_vals.mean()) if len(R_vals) > 0 else np.nan,
        "R_coverage": float(R_covered.mean()),
        "lrt_power": lrt_power,
        "aic_2zone_wins": aic_2zone_wins,
        "ks_pass": ks_pass,
    }


def write_combined_summary(summaries):
    lines = []
    lines.append("=" * 80)
    lines.append("ALL SCENARIOS SUMMARY — DMDHP-2ZONE MC STUDY")
    lines.append("For: Arcede et al. (in prep.) — Table 2")
    lines.append("=" * 80)
    lines.append("")
    hdr = (f"{'Scenario':<12} {'N_ok':>6} {'Avg_N':>7} "
           f"{'N_sh':>6} {'N_ns':>6} {'True_R':>7} "
           f"{'R_mean':>7} {'R_cov':>7} "
           f"{'LRT_pwr':>8} {'AIC_win':>8} {'KS_pass':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in summaries:
        if s is None:
            continue
        lines.append(
            f"{s['scenario']:<12} {s['n_ok']:>6} {s['avg_n']:>7.1f} "
            f"{s['avg_n_sh']:>6.1f} {s['avg_n_ns']:>6.1f} "
            f"{s['true_R']:>7.3f} {s['R_mean']:>7.3f} {s['R_coverage']:>7.3f} "
            f"{s['lrt_power']:>8.3f} {s['aic_2zone_wins']:>8.3f} "
            f"{s['ks_pass']:>8.3f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    print("=== DMDHP-2ZONE MC AGGREGATION ===\n")
    all_stats = []

    for scenario in SCENARIOS:
        print(f"Loading Scenario {scenario}...")
        records = load_scenario(scenario)
        print(f"  Found {len(records)} result files")

        if len(records) == 0:
            print(f"  No results yet — skipping")
            all_stats.append(None)
            continue

        result = compute_summary(records, scenario)
        if result is None:
            all_stats.append(None)
            continue

        text, stats = result
        print(text)
        all_stats.append(stats)

        path = os.path.join(RESULTS_DIR, f"scenario_{scenario}_summary.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"  Saved: {path}\n")

    # Combined summary
    combined = write_combined_summary([s for s in all_stats if s is not None])
    path = os.path.join(RESULTS_DIR, "ALL_SCENARIOS_summary.txt")
    with open(path, "w") as f:
        f.write(combined)
    print("\n" + combined)
    print(f"\nSaved: {path}")

    # Save full CSV
    all_records = []
    for scenario in SCENARIOS:
        all_records.extend(load_scenario(scenario))
    if all_records:
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(RESULTS_DIR, "mc_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Full results CSV: {csv_path}")


if __name__ == "__main__":
    main()
