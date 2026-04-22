"""
dmdhp_2zone_ph_application.py
==============================
Fits the 2-zone DMDHP (shallow vs non-shallow) to multiple Philippine
earthquake sequences and compares depth-dependent productivity across
tectonic settings.

Primary model:   DMDHP-2zone  (6 params: mu, beta, K_sh, K_ns, a_sh, a_ns)
Baseline model:  MDHP         (4 params: mu, beta, K, alpha)

Sequences fitted:
    SEQ1  2023 Hinatuan, Surigao del Sur    Mw 7.4  Philippine Trench INT
    SEQ2  2025 Davao Oriental               Mw 7.4  Philippine Trench INT
    SEQ3  2019 Davao del Sur                Mw 6.9  Cotabato Trench INT
    SEQ4  2019 Cotabato (Oct 29)            Mw 6.6  Cotabato Fault CSS
    SEQ5  2019 Cotabato (Oct 16)            Mw 6.3  Cotabato Fault CSS
    SEQ8  2013 Bohol                        Mw 7.2  N. Bohol Fault CRV

Usage
-----
    python dmdhp_2zone_ph_application.py

Outputs (saved to ./ph_results/)
----------------------------------
    productivity_ratio_table.txt    Key comparison table (Table 1 of paper)
    parameter_estimates_all.txt     Full parameter table for all sequences
    model_comparison_all.txt        AIC / LRT table
    productivity_ratio_plot.png     Figure 1 — R = K_sh/K_ns by sequence
    pit_diagnostics/                PIT plots per sequence
    {seq_id}_results.npz            Numerical results per sequence

Authors: J.P. Arcede (Caraga State University)
Version: 1.0  |  April 2026
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kstest, chi2
from numpy.random import default_rng

# ── output directories ────────────────────────────────────────────────────────
OUTDIR     = "ph_results"
PIT_DIR    = os.path.join(OUTDIR, "pit_diagnostics")
os.makedirs(OUTDIR,  exist_ok=True)
os.makedirs(PIT_DIR, exist_ok=True)

CATALOG_DIR_PH     = "ph_catalogs"
CATALOG_DIR_GLOBAL = "global_catalogs"

# ── model settings ────────────────────────────────────────────────────────────
D1          = 70.0      # shallow / non-shallow boundary (km)
ALPHA_UPPER = 2.25
N_STARTS    = 20
MAXITER     = 2000
N_BOOT      = 200
ALPHA_LEVEL = 0.05
SEED        = 20260419

# ── sequences to fit ──────────────────────────────────────────────────────────
SEQUENCES = [
    dict(seq_id="SEQ1_Hinatuan2023",     name="2023 Hinatuan",
         mw=7.4, mechanism="INT", trench="Philippine Trench",    min_events=50),
    dict(seq_id="SEQ2_DavaoOriental2025",name="2025 Davao Oriental",
         mw=7.4, mechanism="INT", trench="Philippine Trench",    min_events=50),
    dict(seq_id="SEQ3_DavaoDeSur2019",   name="2019 Davao del Sur",
         mw=6.9, mechanism="INT", trench="Cotabato Trench",      min_events=50),
    dict(seq_id="SEQ4_Cotabato2019Oct29",name="2019 Cotabato (Oct 29)",
         mw=6.6, mechanism="CSS", trench="Cotabato Fault",       min_events=30),
    dict(seq_id="SEQ5_Cotabato2019Oct16",name="2019 Cotabato (Oct 16)",
         mw=6.3, mechanism="CSS", trench="Cotabato Fault",       min_events=30),
    dict(seq_id="SEQ8_Bohol2013",        name="2013 Bohol",
         mw=7.2, mechanism="CRV", trench="N. Bohol Fault",       min_events=30),
    # ── Global 2-zone identifiable sequences ─────────────────────────────────
    dict(seq_id="G01_Tohoku2011",         name="2011 Tohoku, Japan",
         mw=9.1, mechanism="INT", trench="Japan Trench",          min_events=50),
    dict(seq_id="G10_Mexico2017",         name="2017 Chiapas, Mexico",
         mw=8.2, mechanism="INT", trench="Middle America Trench", min_events=50),
]

# ── depth zone helper ─────────────────────────────────────────────────────────

def depth_zone_2(d):
    return 0 if d < D1 else 1


# ── 2-zone DMDHP likelihood ───────────────────────────────────────────────────

def loglik_dmdhp_2zone(theta, times, mags, depths, T, m0=4.0):
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0.0) or not np.all(np.isfinite(theta)):
        return -np.inf
    mu, beta, Ksh, Kns, ash, ans = theta
    K     = np.array([Ksh, Kns])
    alpha = np.array([ash, ans])
    n = len(times)
    if n == 0:
        return -mu * T
    ll, R, t_prev = 0.0, 0.0, 0.0
    ks = np.empty(n, dtype=float)
    for i in range(n):
        dt = times[i] - t_prev
        if dt < 0:
            return -np.inf
        R  *= math.exp(-beta * dt)
        lam = mu + beta * R
        if lam <= 0 or not math.isfinite(lam):
            return -np.inf
        ll += math.log(lam)
        j   = depth_zone_2(depths[i])
        ki  = K[j] * math.exp(alpha[j] * (mags[i] - m0))
        ks[i] = ki
        R  += ki
        t_prev = times[i]
    comp = mu * T + np.sum(ks * (1.0 - np.exp(-beta * (T - times))))
    return ll - comp


def negloglik_2zone(phi, times, mags, depths, T, m0):
    theta = np.exp(np.clip(phi, -40.0, 40.0))
    ll = loglik_dmdhp_2zone(theta, times, mags, depths, T, m0)
    return np.inf if not math.isfinite(ll) else -ll


# ── MDHP baseline likelihood ──────────────────────────────────────────────────

def loglik_mdhp(theta, times, mags, T, m0=4.0):
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0.0) or not np.all(np.isfinite(theta)):
        return -np.inf
    mu, beta, K, alpha = theta
    n = len(times)
    if n == 0:
        return -mu * T
    ll, R, t_prev = 0.0, 0.0, 0.0
    ks = np.empty(n, dtype=float)
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


def negloglik_mdhp(phi, times, mags, T, m0):
    theta = np.exp(np.clip(phi, -40.0, 40.0))
    ll = loglik_mdhp(theta, times, mags, T, m0)
    return np.inf if not math.isfinite(ll) else -ll


# ── parameter bounds ──────────────────────────────────────────────────────────

def bounds_2zone(au=ALPHA_UPPER):
    return [
        (math.log(1e-6), math.log(5.0)),    # mu
        (math.log(1e-4), math.log(10.0)),   # beta
        (math.log(1e-8), math.log(10.0)),   # K_sh
        (math.log(1e-8), math.log(10.0)),   # K_ns
        (math.log(1e-6), math.log(au)),     # a_sh
        (math.log(1e-6), math.log(au)),     # a_ns
    ]


def bounds_mdhp(au=ALPHA_UPPER):
    return [
        (math.log(1e-6), math.log(5.0)),
        (math.log(1e-4), math.log(10.0)),
        (math.log(1e-8), math.log(10.0)),
        (math.log(1e-6), math.log(au)),
    ]


# ── multi-start fitting ───────────────────────────────────────────────────────

def multi_start(negfun, phi0, bounds, args, n_starts=N_STARTS, seed=0):
    rng  = default_rng(seed)
    best, best_val = None, np.inf
    lb   = np.array([b[0] for b in bounds])
    ub   = np.array([b[1] for b in bounds])
    for s in range(n_starts):
        start = np.clip(phi0 + (rng.normal(0, 0.6, phi0.size) if s > 0 else 0), lb, ub)
        try:
            res = minimize(negfun, start, args=args, method="L-BFGS-B",
                           bounds=bounds, options=dict(maxiter=MAXITER))
        except Exception:
            continue
        if math.isfinite(res.fun) and res.fun < best_val:
            best_val, best = res.fun, res
    if best is None:
        return None, np.nan, False
    return np.exp(np.clip(best.x, -40.0, 40.0)), -best.fun, bool(best.success)


def fit_2zone(times, mags, depths, T, m0, seed=0):
    mu0   = max(1e-3, 0.3 * len(times) / max(T, 1))
    phi0  = np.log(np.array([mu0, 2.0, 0.3, 0.1, 1.0, 0.5]))
    return multi_start(negloglik_2zone, phi0, bounds_2zone(),
                       args=(times, mags, depths, T, m0), seed=seed)


def fit_mdhp(times, mags, T, m0, seed=1):
    mu0   = max(1e-3, 0.3 * len(times) / max(T, 1))
    phi0  = np.log(np.array([mu0, 2.0, 0.3, 1.0]))
    return multi_start(negloglik_mdhp, phi0, bounds_mdhp(),
                       args=(times, mags, T, m0), seed=seed)


# ── time-rescaling residuals ──────────────────────────────────────────────────

def time_rescaling_2zone(theta, times, mags, depths, m0):
    mu, beta, Ksh, Kns, ash, ans = theta
    K     = np.array([Ksh, Kns])
    alpha = np.array([ash, ans])
    n     = len(times)
    taus  = np.empty(n)
    R, t_prev = 0.0, 0.0
    for i in range(n):
        dt      = times[i] - t_prev
        taus[i] = mu * dt + R * (1.0 - math.exp(-beta * dt))
        j       = depth_zone_2(depths[i])
        ki      = K[j] * math.exp(alpha[j] * (mags[i] - m0))
        R       = R * math.exp(-beta * dt) + ki
        t_prev  = times[i]
    U = 1.0 - np.exp(-taus)
    return taus, U


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def simulate_2zone(theta, T, m0, rng, max_events=50000):
    mu, beta, Ksh, Kns, ash, ans = theta
    K     = np.array([Ksh, Kns])
    alpha = np.array([ash, ans])
    # immigrant depth probabilities — use empirical shallow fraction
    p_sh  = 0.85

    n0   = rng.poisson(mu * T)
    t0s  = np.sort(rng.uniform(0, T, n0))
    z0s  = (rng.uniform(size=n0) > p_sh).astype(int)
    m0s  = m0 + rng.exponential(1.0 / (1.0 * math.log(10)), n0)
    m0s  = np.clip(m0s, m0, 7.5)

    times, mags, depths = list(t0s), list(m0s), list(z0s * 100.0 + 10.0)
    q = 0
    while q < len(times):
        if len(times) > max_events:
            return None
        tp, mp, dp = times[q], mags[q], depths[q]
        zp  = depth_zone_2(dp)
        kp  = K[zp] * math.exp(alpha[zp] * (mp - m0))
        noff = rng.poisson(kp)
        for _ in range(noff):
            tc = tp + rng.exponential(1.0 / beta)
            if tc > T:
                continue
            zc = int(rng.uniform() > p_sh)
            mc = m0 + rng.exponential(1.0 / (1.0 * math.log(10)))
            mc = min(mc, 7.5)
            dc = zc * 100.0 + 10.0
            times.append(tc); mags.append(mc); depths.append(dc)
        q += 1
    order  = np.argsort(times)
    return (np.array(times)[order],
            np.array(mags)[order],
            np.array(depths)[order])


def bootstrap_ci_2zone(theta, T, m0, n_boot=N_BOOT, seed=42):
    rng   = default_rng(seed)
    boot  = []
    attempts = 0
    while len(boot) < n_boot and attempts < 3 * n_boot:
        attempts += 1
        result = simulate_2zone(theta, T, m0, rng)
        if result is None:
            continue
        times_b, mags_b, depths_b = result
        if len(times_b) > 10000:
            continue
        th_b, ll_b, ok_b = fit_2zone(times_b, mags_b, depths_b, T, m0,
                                      seed=int(rng.integers(0, 2**31)))
        if ok_b and th_b is not None and np.all(np.isfinite(th_b)):
            boot.append(th_b)
    if len(boot) < max(10, n_boot // 4):
        return np.full(6, np.nan), np.full(6, np.nan), np.empty((0,6)), len(boot)
    boot_arr = np.array(boot)
    lo = np.quantile(boot_arr, ALPHA_LEVEL / 2, axis=0)
    hi = np.quantile(boot_arr, 1 - ALPHA_LEVEL / 2, axis=0)
    return lo, hi, boot_arr, len(boot)


# ── PIT diagnostics ───────────────────────────────────────────────────────────

def plot_pit(U, taus, seq_id, name, ks_p):
    n = len(U)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Histogram
    ax = axes[0]
    ax.hist(U, bins=12, density=True, color="#2C3E50", edgecolor="white", lw=0.5)
    ax.axhline(1.0, linestyle="--", color="#E74C3C", lw=1.5)
    ax.set_xlabel("PIT residual U")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT histogram  (KS p = {ks_p:.3f})")

    # Q-Q
    ax = axes[1]
    U_s = np.sort(U)
    theo = (np.arange(1, n + 1) - 0.5) / n
    ax.plot(theo, U_s, "o", ms=3, color="#2C3E50", alpha=0.7)
    ax.plot([0, 1], [0, 1], "--", color="#E74C3C", lw=1.5)
    ax.set_xlabel("Theoretical Uniform")
    ax.set_ylabel("Empirical PIT")
    ax.set_title("Q-Q plot")

    # Cumulative rescaled times
    ax = axes[2]
    cum = np.cumsum(taus)
    idx = np.arange(1, n + 1)
    ax.plot(idx, cum, color="#2C3E50", lw=1.5, label="Observed")
    ax.plot(idx, idx, "--", color="#E74C3C", lw=1.5, label="Expected")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Cumulative rescaled time")
    ax.set_title("Cumulative rescaled times")
    ax.legend(fontsize=8)

    fig.suptitle(f"{name} — DMDHP-2zone residual diagnostics", fontsize=11)
    fig.tight_layout()
    path = os.path.join(PIT_DIR, f"{seq_id}_pit.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ── productivity ratio plot ───────────────────────────────────────────────────

def plot_productivity_ratios(results):
    int_res = [r for r in results if r["mechanism"] == "INT" and r["R"] is not None]
    css_res = [r for r in results if r["mechanism"] in ("CSS","CRV") and r["R"] is not None]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"INT": "#E74C3C", "CSS": "#2E86C1", "CRV": "#F39C12"}
    markers = {"INT": "o", "CSS": "s", "CRV": "^"}
    labels_done = set()

    all_res = results
    for i, r in enumerate(all_res):
        if r["R"] is None:
            continue
        mech  = r["mechanism"]
        color = colors.get(mech, "#7F8C8D")
        mark  = markers.get(mech, "o")
        label = {"INT": "Interface thrust (INT)",
                 "CSS": "Crustal strike-slip (CSS)",
                 "CRV": "Crustal reverse (CRV)"}.get(mech, mech)
        lbl   = label if label not in labels_done else ""
        labels_done.add(label)

        ax.errorbar(i, r["R"],
                    yerr=[[r["R"] - r["R_lo"]], [r["R_hi"] - r["R"]]],
                    fmt=mark, color=color, markersize=10, capsize=5,
                    elinewidth=1.5, label=lbl)
        ax.text(i, r["R"] + 0.05, r["short_name"], ha="center",
                fontsize=8, color=color)

    ax.axhline(1.0, linestyle="--", color="#7F8C8D", lw=1.2,
               label="R = 1 (no depth effect)")
    ax.set_xticks([])
    ax.set_ylabel("Productivity ratio  R = K$_{sh}$ / K$_{ns}$", fontsize=11)
    ax.set_title("Depth-dependent productivity ratio by tectonic mechanism\n"
                 "Philippine earthquake sequences", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "productivity_ratio_plot.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== DMDHP-2ZONE: PHILIPPINE MULTI-SEQUENCE APPLICATION ===\n")
    rng_master = default_rng(SEED)
    all_results = []

    for seq in SEQUENCES:
        seq_id = seq["seq_id"]
        # Check ph_catalogs first, then global_catalogs
        catalog_path = os.path.join(CATALOG_DIR_PH, f"{seq_id}_catalog.csv")
        if not os.path.exists(catalog_path):
            catalog_path = os.path.join(CATALOG_DIR_GLOBAL, f"{seq_id}_catalog.csv")

        print(f"\n{'='*60}")
        print(f"  {seq_id}")
        print(f"  {seq['name']}  (Mw {seq['mw']}, {seq['mechanism']})")
        print(f"{'='*60}")

        if not os.path.exists(catalog_path):
            print(f"  WARNING: Catalog not found: {catalog_path} — skipping.")
            all_results.append({**seq, "status": "missing", "R": None})
            continue

        df = pd.read_csv(catalog_path)
        print(f"  Events loaded: {len(df)}")

        if len(df) < seq["min_events"]:
            print(f"  SKIP: too few events ({len(df)} < {seq['min_events']})")
            all_results.append({**seq, "status": "insufficient", "R": None})
            continue

        # Determine m0 from catalog
        m0 = float(df["mag"].min())
        m0 = max(round(m0 * 2) / 2, 3.0)  # round to nearest 0.5
        print(f"  m0 = {m0:.1f}")

        times  = df["t_days"].values.astype(float)
        mags   = df["mag"].values.astype(float)
        depths = df["depth"].values.astype(float)
        T      = float(times.max())

        n_sh  = int((depths <  D1).sum())
        n_ns  = int((depths >= D1).sum())
        print(f"  Shallow: {n_sh}  |  Non-shallow: {n_ns}  |  T = {T:.2f} days")

        # ── Fit 2-zone DMDHP ──────────────────────────────────────────────────
        print(f"  Fitting DMDHP-2zone ...")
        seed = int(rng_master.integers(0, 2**31))
        th2, ll2, ok2 = fit_2zone(times, mags, depths, T, m0, seed=seed)

        if not ok2 or th2 is None:
            print(f"  WARNING: DMDHP-2zone did not converge.")
            all_results.append({**seq, "status": "no_converge", "R": None})
            continue

        aic2 = -2.0 * ll2 + 2.0 * 6
        mu, beta, Ksh, Kns, ash, ans = th2
        R = Ksh / Kns if Kns > 0 else np.nan
        print(f"  converged: logL={ll2:.3f}  AIC={aic2:.3f}")
        print(f"  mu={mu:.4f}  beta={beta:.4f}")
        print(f"  K_sh={Ksh:.4f}  K_ns={Kns:.4f}  R=K_sh/K_ns={R:.3f}")
        print(f"  a_sh={ash:.4f}  a_ns={ans:.4f}")

        # ── Bootstrap CI ──────────────────────────────────────────────────────
        print(f"  Bootstrap CI ({N_BOOT} samples) ...")
        lo, hi, boot_arr, n_boot_ok = bootstrap_ci_2zone(th2, T, m0, n_boot=N_BOOT,
                                                seed=seed + 111)
        print(f"  Successful bootstrap refits: {n_boot_ok}/{N_BOOT}")
        # Compute R CI directly from bootstrap distribution of R = K_sh/K_ns
        if len(boot_arr) >= 10:
            boot_R = boot_arr[:, 2] / np.where(boot_arr[:, 3] > 1e-10, boot_arr[:, 3], np.nan)
            boot_R = boot_R[np.isfinite(boot_R) & (boot_R < 1e6)]
            if len(boot_R) >= 5:
                R_lo = float(np.quantile(boot_R, ALPHA_LEVEL / 2))
                R_hi = float(np.quantile(boot_R, 1 - ALPHA_LEVEL / 2))
            else:
                R_lo, R_hi = np.nan, np.nan
        else:
            R_lo, R_hi = np.nan, np.nan

        # ── Fit MDHP baseline ─────────────────────────────────────────────────
        print(f"  Fitting MDHP baseline ...")
        th1, ll1, ok1 = fit_mdhp(times, mags, T, m0, seed=seed + 1)
        aic1 = -2.0 * ll1 + 2.0 * 4 if ok1 and ll1 is not None else np.nan

        # ── LRT ───────────────────────────────────────────────────────────────
        if ok1 and np.isfinite(ll1) and np.isfinite(ll2):
            LR    = 2.0 * (ll2 - ll1)
            lrt_p = 1.0 - chi2.cdf(max(LR, 0.0), df=2)
        else:
            LR, lrt_p = np.nan, np.nan

        print(f"  LRT (2-zone vs MDHP): LR={LR:.3f}  p={lrt_p:.4f}")

        # ── PIT diagnostics ───────────────────────────────────────────────────
        taus, U = time_rescaling_2zone(th2, times, mags, depths, m0)
        ks_p    = kstest(U, "uniform").pvalue
        plot_pit(U, taus, seq_id, seq["name"], ks_p)
        print(f"  KS p-value: {ks_p:.4f}")

        # ── Store results ─────────────────────────────────────────────────────
        result = {
            **seq,
            "status":    "ok",
            "n_total":   len(df),
            "n_sh":      n_sh,
            "n_ns":      n_ns,
            "T":         T,
            "m0":        m0,
            "mu":        mu,   "mu_lo":  lo[0], "mu_hi":  hi[0],
            "beta":      beta, "beta_lo":lo[1], "beta_hi":hi[1],
            "K_sh":      Ksh,  "K_sh_lo":lo[2], "K_sh_hi":hi[2],
            "K_ns":      Kns,  "K_ns_lo":lo[3], "K_ns_hi":hi[3],
            "a_sh":      ash,  "a_sh_lo":lo[4], "a_sh_hi":hi[4],
            "a_ns":      ans,  "a_ns_lo":lo[5], "a_ns_hi":hi[5],
            "ll_2zone":  ll2,  "aic_2zone": aic2,
            "ll_mdhp":   ll1,  "aic_mdhp":  aic1,
            "LR":        LR,   "lrt_p":     lrt_p,
            "R":         R,    "R_lo":      R_lo, "R_hi": R_hi,
            "ks_p":      ks_p,
            "n_boot_ok": n_boot_ok,
            "short_name": seq["name"].replace("2023 ", "").replace(
                          "2025 ", "").replace("2019 ", "").replace("2013 ", ""),
        }
        all_results.append(result)

        np.savez(os.path.join(OUTDIR, f"{seq_id}_results.npz"), **{
            k: v for k, v in result.items()
            if isinstance(v, (int, float, np.ndarray)) or v is None
        })

    # ── Write tables ──────────────────────────────────────────────────────────
    ok_results = [r for r in all_results if r.get("status") == "ok"]

    # Table 1 — productivity ratio
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 1 — PRODUCTIVITY RATIO R = K_sh / K_ns BY SEQUENCE")
    lines.append("DMDHP-2zone (shallow vs non-shallow, d1=70 km)")
    lines.append("=" * 90)
    hdr = (f"{'Sequence':<28s}  {'Mw':>4}  {'Mech':>4}  "
           f"{'K_sh':>8}  {'K_ns':>8}  {'R':>7}  "
           f"{'R 95CI Lo':>10}  {'R 95CI Hi':>10}  "
           f"{'LRT p':>8}  {'KS p':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in ok_results:
        lrt_s = f"{r['lrt_p']:.4f}" if np.isfinite(r['lrt_p']) else "   —"
        sig   = "***" if r['lrt_p'] < 0.001 else ("**" if r['lrt_p'] < 0.01
                else ("*" if r['lrt_p'] < 0.05 else ""))
        R_lo_s = f"{r['R_lo']:.3f}" if np.isfinite(r.get('R_lo', np.nan)) else "  —"
        R_hi_s = f"{r['R_hi']:.3f}" if np.isfinite(r.get('R_hi', np.nan)) else "  —"
        lines.append(
            f"{r['name']:<28s}  {r['mw']:>4.1f}  {r['mechanism']:>4s}  "
            f"{r['K_sh']:>8.4f}  {r['K_ns']:>8.4f}  {r['R']:>7.3f}  "
            f"{R_lo_s:>10s}  {R_hi_s:>10s}  "
            f"{lrt_s:>8s}{sig:3s}  {r['ks_p']:>7.3f}"
        )
    lines.append("")
    lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05")
    lines.append("R > 1: shallow earthquakes more productive than non-shallow")
    lines.append("R ≈ 1: no depth-dependent productivity")
    lines.append("KS p: p-value of Kolmogorov-Smirnov test for PIT uniformity")
    lines.append("=" * 90)

    ratio_text = "\n".join(lines)
    with open(os.path.join(OUTDIR, "productivity_ratio_table.txt"), "w") as f:
        f.write(ratio_text)
    print("\n" + ratio_text)

    # Table 2 — full parameters
    param_lines = []
    param_lines.append("=" * 95)
    param_lines.append("TABLE 2 — FULL PARAMETER ESTIMATES (DMDHP-2zone)")
    param_lines.append("=" * 95)
    for r in ok_results:
        param_lines.append(f"\n{r['name']}  (Mw {r['mw']}, {r['mechanism']})")
        param_lines.append(f"  n={r['n_total']}  n_sh={r['n_sh']}  n_ns={r['n_ns']}"
                           f"  T={r['T']:.1f}d  m0={r['m0']:.1f}")
        param_lines.append(f"  {'Param':>8}  {'Estimate':>10}  {'95CI Lo':>10}  {'95CI Hi':>10}")
        param_lines.append(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
        for name, key in [("mu","mu"),("beta","beta"),("K_sh","K_sh"),
                          ("K_ns","K_ns"),("a_sh","a_sh"),("a_ns","a_ns")]:
            lo_s = f"{r[key+'_lo']:.5f}" if np.isfinite(r.get(key+'_lo', np.nan)) else "    —"
            hi_s = f"{r[key+'_hi']:.5f}" if np.isfinite(r.get(key+'_hi', np.nan)) else "    —"
            param_lines.append(f"  {name:>8}  {r[key]:>10.5f}  {lo_s:>10}  {hi_s:>10}")
        param_lines.append(f"  logL={r['ll_2zone']:.3f}  AIC={r['aic_2zone']:.3f}"
                           f"  |  MDHP logL={r['ll_mdhp']:.3f}  AIC={r['aic_mdhp']:.3f}"
                           f"  |  LR={r['LR']:.3f}  p={r['lrt_p']:.4f}")
    param_lines.append("\n" + "=" * 95)

    with open(os.path.join(OUTDIR, "parameter_estimates_all.txt"), "w") as f:
        f.write("\n".join(param_lines))

    # ── Productivity ratio plot ───────────────────────────────────────────────
    plot_productivity_ratios(ok_results)

    print(f"\nAll results saved to: {os.path.abspath(OUTDIR)}/")
    print("Key output: ph_results/productivity_ratio_table.txt")
    print("Key figure: ph_results/productivity_ratio_plot.png")


if __name__ == "__main__":
    main()
