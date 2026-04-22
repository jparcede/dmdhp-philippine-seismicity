"""
dmdhp_2zone_mc_single.py
=========================
Single-replication worker for the DMDHP-2zone Monte Carlo simulation study.
Designed to run as one job in a SLURM array — each job handles exactly
one replication and saves its result to a shared output directory.

Usage (SLURM array job):
    sbatch --array=1-200 run_slurm_array.sh

Usage (single test run):
    python3 dmdhp_2zone_mc_single.py --rep 1 --scenario A --outdir ./mc_results

True parameters are calibrated to match fitted values from the
2023 Hinatuan sequence (SEQ1), the most data-rich Philippine sequence.

Three scenarios:
    A: Hinatuan-type   — 85/15 shallow/non-shallow, T=90 days, n~200-400 events
    B: Davao-type      — 75/25 shallow/non-shallow, T=90 days, n~150-250 events
    C: Balanced        — 60/40 shallow/non-shallow, T=90 days, n~100-200 events

Authors: J.P. Arcede (Caraga State University)
Version: 1.0  |  April 2026
"""

import os
import sys
import math
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, chi2
from numpy.random import default_rng

# ── True parameters — calibrated to Hinatuan fitted values ───────────────────
# K_sh=0.305, K_ns=0.037 from SEQ1 fit
# alpha_sh=1.47, alpha_ns=0.16 from SEQ1 fit
# mu=1.13, beta=4.00 from SEQ1 fit (scaled down for smaller catalog targets)

TRUE_PARAMS = {
    "A": dict(
        mu=2.50, beta=4.00,
        K_sh=0.15, K_ns=0.020,
        a_sh=1.20, a_ns=0.20,
        p_sh=0.85,   # 85% shallow — matches Hinatuan (84.2%)
        T=90.0, m0=4.0,
        label="Hinatuan-type (85/15, T=90d)",
        notes="R=K_sh/K_ns=7.5, expected N~300, calibrated to SEQ1",
    ),
    "B": dict(
        mu=1.00, beta=4.00,
        K_sh=0.15, K_ns=0.060,
        a_sh=1.00, a_ns=0.50,
        p_sh=0.75,   # 75% shallow — matches Davao Oriental (74.8%)
        T=90.0, m0=4.0,
        label="Davao-type (75/25, T=90d)",
        notes="R=K_sh/K_ns=2.5, expected N~115, calibrated to SEQ2",
    ),
    "C": dict(
        mu=0.80, beta=4.00,
        K_sh=0.12, K_ns=0.080,
        a_sh=0.80, a_ns=0.70,
        p_sh=0.60,   # 60% shallow — balanced test case
        T=90.0, m0=4.0,
        label="Balanced (60/40, T=90d)",
        notes="R=K_sh/K_ns=1.5, expected N~85, tests weak depth signal",
    ),
}

D1      = 70.0
ALPHA_UPPER = 2.25
N_STARTS    = 15
MAXITER     = 1500
N_BOOT      = 50       # bootstrap samples per replication
ALPHA_LEVEL = 0.05


# ── Depth zone ────────────────────────────────────────────────────────────────

def depth_zone_2(d):
    return 0 if d < D1 else 1


# ── Simulate 2-zone DMDHP ─────────────────────────────────────────────────────

def simulate_2zone(params, rng, max_events=20000):
    mu   = params["mu"]
    beta = params["beta"]
    K    = np.array([params["K_sh"], params["K_ns"]])
    alph = np.array([params["a_sh"], params["a_ns"]])
    p_sh = params["p_sh"]
    T    = params["T"]
    m0   = params["m0"]
    b    = 1.0 * math.log(10)   # GR b-value decay

    # Background events
    n0  = rng.poisson(mu * T)
    t0s = np.sort(rng.uniform(0, T, n0))
    z0s = (rng.uniform(size=n0) > p_sh).astype(int)
    m0s = m0 + rng.exponential(1.0 / b, n0)
    m0s = np.clip(m0s, m0, 7.5)

    times  = list(t0s)
    mags   = list(m0s)
    depths = list(z0s * 100.0 + 10.0)

    q = 0
    while q < len(times):
        if len(times) > max_events:
            return None  # supercritical — discard
        tp, mp, dp = times[q], mags[q], depths[q]
        zp  = depth_zone_2(dp)
        kp  = K[zp] * math.exp(alph[zp] * (mp - m0))
        noff = rng.poisson(kp)
        for _ in range(noff):
            tc = tp + rng.exponential(1.0 / beta)
            if tc > T:
                continue
            zc = int(rng.uniform() > p_sh)
            mc = m0 + rng.exponential(1.0 / b)
            mc = min(mc, 7.5)
            dc = zc * 100.0 + 10.0
            times.append(tc)
            mags.append(mc)
            depths.append(dc)
        q += 1

    order  = np.argsort(times)
    return (np.array(times)[order],
            np.array(mags)[order],
            np.array(depths)[order])


# ── Log-likelihoods ───────────────────────────────────────────────────────────

def loglik_2zone(theta, times, mags, depths, T, m0):
    if np.any(theta <= 0) or not np.all(np.isfinite(theta)):
        return -np.inf
    mu, beta, Ksh, Kns, ash, ans = theta
    K = np.array([Ksh, Kns])
    a = np.array([ash, ans])
    n = len(times)
    ll, R, t_prev = 0.0, 0.0, 0.0
    ks = np.empty(n)
    for i in range(n):
        dt = times[i] - t_prev
        R  *= math.exp(-beta * dt)
        lam = mu + beta * R
        if lam <= 0 or not math.isfinite(lam):
            return -np.inf
        ll += math.log(lam)
        j   = depth_zone_2(depths[i])
        ki  = K[j] * math.exp(a[j] * (mags[i] - m0))
        ks[i] = ki
        R  += ki
        t_prev = times[i]
    comp = mu * T + np.sum(ks * (1.0 - np.exp(-beta * (T - times))))
    return ll - comp


def loglik_mdhp(theta, times, mags, T, m0):
    if np.any(theta <= 0) or not np.all(np.isfinite(theta)):
        return -np.inf
    mu, beta, K, alpha = theta
    n = len(times)
    ll, R, t_prev = 0.0, 0.0, 0.0
    ks = np.empty(n)
    for i in range(n):
        dt = times[i] - t_prev
        R  *= math.exp(-beta * dt)
        lam = mu + beta * R
        if lam <= 0 or not math.isfinite(lam):
            return -np.inf
        ll += math.log(lam)
        ki  = K * math.exp(alpha * (mags[i] - m0))
        ks[i] = ki
        R  += ki
        t_prev = times[i]
    comp = mu * T + np.sum(ks * (1.0 - np.exp(-beta * (T - times))))
    return ll - comp


# ── Multi-start fitting ───────────────────────────────────────────────────────

def fit_2zone(times, mags, depths, T, m0, seed=0):
    rng  = default_rng(seed)
    n    = len(times)
    mu0  = max(1e-3, 0.2 * n / max(T, 1))
    phi0 = np.log(np.array([mu0, 3.0, 0.15, 0.03, 1.2, 0.3]))
    lb   = np.array([math.log(1e-6), math.log(1e-4), math.log(1e-8),
                     math.log(1e-8), math.log(1e-6), math.log(1e-6)])
    ub   = np.array([math.log(5.0),  math.log(10.0), math.log(10.0),
                     math.log(10.0), math.log(ALPHA_UPPER), math.log(ALPHA_UPPER)])
    bounds = list(zip(lb, ub))

    best, best_val = None, np.inf
    for s in range(N_STARTS):
        start = np.clip(phi0 + (rng.normal(0, 0.5, 6) if s > 0 else 0), lb, ub)
        try:
            res = minimize(
                lambda p: (lambda ll: np.inf if not math.isfinite(ll) else -ll)(
                    loglik_2zone(np.exp(np.clip(p, -40, 40)),
                                 times, mags, depths, T, m0)),
                start, method="L-BFGS-B", bounds=bounds,
                options=dict(maxiter=MAXITER))
            if math.isfinite(res.fun) and res.fun < best_val:
                best_val, best = res.fun, res
        except Exception:
            continue
    if best is None:
        return None, np.nan, False
    th = np.exp(np.clip(best.x, -40, 40))
    return th, -best.fun, bool(best.success)


def fit_mdhp(times, mags, T, m0, seed=1):
    rng  = default_rng(seed)
    n    = len(times)
    mu0  = max(1e-3, 0.2 * n / max(T, 1))
    phi0 = np.log(np.array([mu0, 3.0, 0.15, 1.0]))
    lb   = np.array([math.log(1e-6), math.log(1e-4),
                     math.log(1e-8), math.log(1e-6)])
    ub   = np.array([math.log(5.0),  math.log(10.0),
                     math.log(10.0), math.log(ALPHA_UPPER)])
    bounds = list(zip(lb, ub))

    best, best_val = None, np.inf
    for s in range(N_STARTS):
        start = np.clip(phi0 + (rng.normal(0, 0.5, 4) if s > 0 else 0), lb, ub)
        try:
            res = minimize(
                lambda p: (lambda ll: np.inf if not math.isfinite(ll) else -ll)(
                    loglik_mdhp(np.exp(np.clip(p, -40, 40)),
                                times, mags, T, m0)),
                start, method="L-BFGS-B", bounds=bounds,
                options=dict(maxiter=MAXITER))
            if math.isfinite(res.fun) and res.fun < best_val:
                best_val, best = res.fun, res
        except Exception:
            continue
    if best is None:
        return None, np.nan, False
    th = np.exp(np.clip(best.x, -40, 40))
    return th, -best.fun, bool(best.success)


# ── Time-rescaling residuals ──────────────────────────────────────────────────

def pit_ks_2zone(theta, times, mags, depths, m0):
    mu, beta, Ksh, Kns, ash, ans = theta
    K = np.array([Ksh, Kns])
    a = np.array([ash, ans])
    n = len(times)
    taus = np.empty(n)
    R, t_prev = 0.0, 0.0
    for i in range(n):
        dt = times[i] - t_prev
        taus[i] = mu * dt + R * (1.0 - math.exp(-beta * dt))
        j  = depth_zone_2(depths[i])
        ki = K[j] * math.exp(a[j] * (mags[i] - m0))
        R  = R * math.exp(-beta * dt) + ki
        t_prev = times[i]
    U  = 1.0 - np.exp(-taus)
    return kstest(U, "uniform").pvalue


# ── Bootstrap CI for R ────────────────────────────────────────────────────────

def bootstrap_R(theta, params, n_boot, rng):
    boot_R = []
    attempts = 0
    while len(boot_R) < n_boot and attempts < 3 * n_boot:
        attempts += 1
        result = simulate_2zone(params, rng, max_events=10000)
        if result is None:
            continue
        t_b, m_b, d_b = result
        if len(t_b) < 10:
            continue
        th_b, ll_b, ok_b = fit_2zone(t_b, m_b, d_b, params["T"],
                                      params["m0"],
                                      seed=int(rng.integers(0, 2**31)))
        if ok_b and th_b is not None and np.all(np.isfinite(th_b)):
            Ksh_b, Kns_b = th_b[2], th_b[3]
            if Kns_b > 1e-8:
                boot_R.append(Ksh_b / Kns_b)
    if len(boot_R) < 5:
        return np.nan, np.nan, len(boot_R)
    boot_R = np.array(boot_R)
    boot_R = boot_R[boot_R < 1e4]  # drop exploded
    if len(boot_R) < 5:
        return np.nan, np.nan, len(boot_R)
    return (float(np.quantile(boot_R, ALPHA_LEVEL / 2)),
            float(np.quantile(boot_R, 1 - ALPHA_LEVEL / 2)),
            len(boot_R))


# ── Main replication ──────────────────────────────────────────────────────────

def run_replication(rep_id, scenario, outdir, seed_base=42):
    params = TRUE_PARAMS[scenario]
    rng    = default_rng(seed_base + rep_id * 997)

    # 1 — Simulate catalog
    result = None
    for attempt in range(20):
        result = simulate_2zone(params, rng)
        if result is not None and len(result[0]) >= 20:
            break
    if result is None:
        return {"rep": rep_id, "scenario": scenario, "status": "sim_failed"}

    times, mags, depths = result
    n      = len(times)
    T      = params["T"]
    m0     = params["m0"]
    n_sh   = int((depths < D1).sum())
    n_ns   = int((depths >= D1).sum())

    # 2 — Fit 2-zone DMDHP
    th2, ll2, ok2 = fit_2zone(times, mags, depths, T, m0,
                               seed=int(rng.integers(0, 2**31)))
    if not ok2 or th2 is None:
        return {"rep": rep_id, "scenario": scenario, "status": "fit_failed",
                "n": n, "n_sh": n_sh, "n_ns": n_ns}

    mu_hat, beta_hat, Ksh_hat, Kns_hat, ash_hat, ans_hat = th2
    R_hat = Ksh_hat / Kns_hat if Kns_hat > 1e-10 else np.nan
    aic2  = -2.0 * ll2 + 2.0 * 6

    # 3 — Fit MDHP baseline
    th1, ll1, ok1 = fit_mdhp(times, mags, T, m0,
                              seed=int(rng.integers(0, 2**31)))
    aic1 = -2.0 * ll1 + 2.0 * 4 if ok1 and np.isfinite(ll1) else np.nan

    # 4 — LRT
    if ok1 and np.isfinite(ll1) and np.isfinite(ll2):
        LR    = 2.0 * (ll2 - ll1)
        lrt_p = 1.0 - chi2.cdf(max(LR, 0.0), df=2)
    else:
        LR, lrt_p = np.nan, np.nan

    # 5 — Bootstrap CI for R
    R_lo, R_hi, n_boot_ok = bootstrap_R(th2, params, N_BOOT, rng)

    # 6 — PIT KS test
    ks_p = pit_ks_2zone(th2, times, mags, depths, m0)

    # 7 — Coverage check (is true R in CI?)
    true_R = params["K_sh"] / params["K_ns"]
    R_covered = (np.isfinite(R_lo) and np.isfinite(R_hi) and
                 R_lo <= true_R <= R_hi)

    result_dict = {
        "rep":        rep_id,
        "scenario":   scenario,
        "status":     "ok",
        "n":          n,
        "n_sh":       n_sh,
        "n_ns":       n_ns,
        # true params
        "true_mu":    params["mu"],
        "true_beta":  params["beta"],
        "true_Ksh":   params["K_sh"],
        "true_Kns":   params["K_ns"],
        "true_ash":   params["a_sh"],
        "true_ans":   params["a_ns"],
        "true_R":     true_R,
        # estimates
        "mu_hat":     float(mu_hat),
        "beta_hat":   float(beta_hat),
        "Ksh_hat":    float(Ksh_hat),
        "Kns_hat":    float(Kns_hat),
        "ash_hat":    float(ash_hat),
        "ans_hat":    float(ans_hat),
        "R_hat":      float(R_hat) if np.isfinite(R_hat) else None,
        # model comparison
        "ll2":        float(ll2),
        "aic2":       float(aic2),
        "ll1":        float(ll1) if np.isfinite(ll1) else None,
        "aic1":       float(aic1) if np.isfinite(aic1) else None,
        "LR":         float(LR) if np.isfinite(LR) else None,
        "lrt_p":      float(lrt_p) if np.isfinite(lrt_p) else None,
        # bootstrap R CI
        "R_lo":       float(R_lo) if np.isfinite(R_lo) else None,
        "R_hi":       float(R_hi) if np.isfinite(R_hi) else None,
        "R_covered":  bool(R_covered),
        "n_boot_ok":  n_boot_ok,
        # diagnostics
        "ks_p":       float(ks_p),
    }

    # Save individual result as JSON
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"rep_{rep_id:04d}_scen{scenario}.json")
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    return result_dict


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DMDHP-2zone MC single replication")
    parser.add_argument("--rep",      type=int,   required=True,
                        help="Replication index (1-based)")
    parser.add_argument("--scenario", type=str,   required=True,
                        choices=["A", "B", "C"],
                        help="Scenario: A=Hinatuan-type, B=Davao-type, C=Balanced")
    parser.add_argument("--outdir",   type=str,   default="./mc_results",
                        help="Output directory for JSON results")
    parser.add_argument("--seed",     type=int,   default=20260420,
                        help="Base random seed")
    args = parser.parse_args()

    print(f"Rep {args.rep} | Scenario {args.scenario} | "
          f"Outdir: {args.outdir}", flush=True)

    result = run_replication(args.rep, args.scenario, args.outdir, args.seed)
    print(f"Done: status={result.get('status')} "
          f"n={result.get('n')} "
          f"R_hat={result.get('R_hat')} "
          f"lrt_p={result.get('lrt_p')} "
          f"R_covered={result.get('R_covered')}", flush=True)


if __name__ == "__main__":
    main()
