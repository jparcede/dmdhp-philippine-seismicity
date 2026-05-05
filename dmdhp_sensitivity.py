"""
dmdhp_sensitivity.py
====================
Sensitivity analysis for the DMDHP paper.

Tests two dimensions of sensitivity:
  1. Depth threshold d1 = 50, 60, 70, 80, 90 km  (main test)
  2. Completeness magnitude m0 = 4.0, 4.5          (secondary test)

Applied to: 2023 Mw 7.4 Hinatuan sequence (the primary result)
Also applied to: 2025 Mw 7.4 Davao Oriental (borderline significant)

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    python3 dmdhp_sensitivity.py

Output:
  sensitivity_results/threshold_sensitivity.txt   — Table for paper
  sensitivity_results/completeness_sensitivity.txt
  sensitivity_results/sensitivity_summary.csv     — Full results

Author: J.P. Arcede (Caraga State University)
"""

import os, sys, csv, math, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kstest

OUTDIR = "sensitivity_results"
os.makedirs(OUTDIR, exist_ok=True)

# ── Core DMDHP functions ──────────────────────────────────────────────────────

def compute_loglik_dmdhp(params, times, mags, zones, T, m0):
    """Log-likelihood of 2-zone DMDHP."""
    mu, beta, Ksh, Kns, ash, ans = params
    if mu <= 0 or beta <= 0 or Ksh <= 0 or Kns <= 0 or ash <= 0 or ans <= 0:
        return -1e10
    K  = np.where(zones == 0, Ksh, Kns)
    a  = np.where(zones == 0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    # Conditional intensity at each event time
    lam = np.zeros(N)
    lam[:] = mu
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += beta * np.sum(ki[:i] * np.exp(-beta * dt))
    if np.any(lam <= 0):
        return -1e10
    ll = np.sum(np.log(lam))
    # Compensator
    comp = mu * T + np.sum(ki * (1.0 - np.exp(-beta * (T - times))))
    return ll - comp


def compute_loglik_mdhp(params, times, mags, T, m0):
    """Log-likelihood of baseline MDHP (uniform K)."""
    mu, beta, K, a = params
    if mu <= 0 or beta <= 0 or K <= 0 or a <= 0:
        return -1e10
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    lam = np.zeros(N)
    lam[:] = mu
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += beta * np.sum(ki[:i] * np.exp(-beta * dt))
    if np.any(lam <= 0):
        return -1e10
    ll = np.sum(np.log(lam))
    comp = mu * T + np.sum(ki * (1.0 - np.exp(-beta * (T - times))))
    return ll - comp


def fit_dmdhp(times, mags, zones, T, m0, n_starts=15):
    """Fit DMDHP by MLE with multi-start."""
    best_ll = -1e15
    best_params = None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [
            np.random.uniform(0.5, 3.0),   # mu
            np.random.uniform(2.0, 6.0),   # beta
            np.random.uniform(0.05, 0.5),  # Ksh
            np.random.uniform(0.01, 0.2),  # Kns
            np.random.uniform(0.5, 2.0),   # ash
            np.random.uniform(0.1, 1.5),   # ans
        ]
        bounds = [(1e-4,None),(1e-4,None),(1e-4,None),(1e-4,None),
                  (1e-4,2.25),(1e-4,2.25)]
        res = minimize(
            lambda p: -compute_loglik_dmdhp(p, times, mags, zones, T, m0),
            p0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_params = res.x
    return best_params, best_ll


def fit_mdhp(times, mags, T, m0, n_starts=15):
    """Fit baseline MDHP."""
    best_ll = -1e15
    best_params = None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [
            np.random.uniform(0.5, 3.0),
            np.random.uniform(2.0, 6.0),
            np.random.uniform(0.05, 0.5),
            np.random.uniform(0.5, 2.0),
        ]
        bounds = [(1e-4,None),(1e-4,None),(1e-4,None),(1e-4,2.25)]
        res = minimize(
            lambda p: -compute_loglik_mdhp(p, times, mags, T, m0),
            p0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_params = res.x
    return best_params, best_ll


def pit_ks(params, times, mags, zones, T, m0):
    """KS test on PIT residuals."""
    mu, beta, Ksh, Kns, ash, ans = params
    K  = np.where(zones == 0, Ksh, Kns)
    a  = np.where(zones == 0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    taus = np.zeros(N)
    R_acc, t_prev = 0.0, 0.0
    for i in range(N):
        dt = times[i] - t_prev
        taus[i] = mu * dt + R_acc * (1.0 - math.exp(-beta * dt))
        R_acc   = R_acc * math.exp(-beta * dt) + ki[i]
        t_prev  = times[i]
    U = 1.0 - np.exp(-taus)
    _, ks_p = kstest(U, 'uniform')
    return ks_p


def run_one(times, mags, T, m0, d1):
    """Run DMDHP + MDHP for one (d1, m0) combination."""
    # Apply completeness filter
    mask  = mags >= m0
    t     = times[mask]
    m     = mags[mask]
    if len(t) < 30:
        return None

    zones = (np.array([d for d in depths])[mask] >= d1).astype(int)
    Nsh   = int(np.sum(zones == 0))
    Nns   = int(np.sum(zones == 1))

    # Fit both models
    p_dm, ll_dm = fit_dmdhp(t, m, zones, T, m0)
    p_md, ll_md = fit_mdhp(t, m, T, m0)

    if p_dm is None or p_md is None:
        return None

    mu, beta, Ksh, Kns, ash, ans = p_dm
    R     = Ksh / Kns if Kns > 1e-8 else float('inf')
    lrt   = 2.0 * (ll_dm - ll_md)
    lrt_p = 1.0 - chi2.cdf(lrt, df=2)
    ks_p  = pit_ks(p_dm, t, m, zones, T, m0)

    aic_dm = -2 * ll_dm + 2 * 6
    aic_md = -2 * ll_md + 2 * 4

    return {
        'd1': d1, 'm0': m0,
        'N': len(t), 'Nsh': Nsh, 'Nns': Nns,
        'Ksh': Ksh, 'Kns': Kns, 'R': R,
        'lrt_p': lrt_p, 'ks_p': ks_p,
        'AIC_DMDHP': aic_dm, 'AIC_MDHP': aic_md,
        'ΔAIC': aic_dm - aic_md,
    }


# ── Load catalogs ─────────────────────────────────────────────────────────────

def load_catalog(filepath, m0_min=3.5):
    """Load CSV catalog. Returns times, mags, depths arrays."""
    import csv
    times_out, mags_out, depths_out = [], [], []
    with open(filepath, newline='') as f:
        sample = f.read(2048); f.seek(0)
        import csv as _csv
        dialect = _csv.Sniffer().sniff(sample)
        reader  = _csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames

        def col(names):
            for n in names:
                for h in headers:
                    if n.lower() in h.lower():
                        return h
            return None

        t_col = col(['t_day','time_day','days','time'])
        m_col = col(['mag','magnitude','mw','ml'])
        d_col = col(['depth','dep'])

        for row in reader:
            try:
                t = float(row[t_col])
                m = float(row[m_col])
                d = float(row[d_col])
                if m >= m0_min:
                    times_out.append(t)
                    mags_out.append(m)
                    depths_out.append(d)
            except (ValueError, TypeError):
                pass

    idx = np.argsort(times_out)
    return (np.array(times_out)[idx],
            np.array(mags_out)[idx],
            np.array(depths_out)[idx])


# ── Main ──────────────────────────────────────────────────────────────────────

sequences = [
    ("Hinatuan 2023",     "ph_catalogs/SEQ1_Hinatuan2023_catalog.csv"),
    ("Davao Oriental 2025","ph_catalogs/SEQ2_DavaoOriental2025_catalog.csv"),
]

depth_thresholds   = [50, 60, 70, 80, 90]
completeness_mags  = [4.0, 4.5]

all_results = []

for seq_name, cat_path in sequences:
    if not os.path.exists(cat_path):
        print(f"  SKIP {seq_name}: catalog not found at {cat_path}")
        continue

    print(f"\n{'='*60}")
    print(f"Sequence: {seq_name}")
    print(f"{'='*60}")

    times, mags, depths = load_catalog(cat_path)
    T = times.max()
    print(f"  Loaded {len(times)} events, T={T:.1f} days")

    # ── Test 1: Depth threshold sensitivity (m0=4.0 fixed) ───────────────────
    print(f"\n  Depth threshold sensitivity (m0=4.0):")
    print(f"  {'d1':>4}  {'N':>5}  {'Nsh':>5}  {'Nns':>4}  "
          f"{'Ksh':>7}  {'Kns':>7}  {'R':>8}  {'LRT p':>8}  {'KS p':>6}")
    print(f"  {'-'*72}")

    for d1 in depth_thresholds:
        r = run_one(times, mags, T, 4.0, d1)
        if r is None:
            print(f"  {d1:>4}: FAILED")
            continue
        r['sequence'] = seq_name
        r['test'] = 'threshold'
        all_results.append(r)

        sig = "***" if r['lrt_p']<0.001 else ("*" if r['lrt_p']<0.05 else "")
        print(f"  {d1:>4}  {r['N']:>5}  {r['Nsh']:>5}  {r['Nns']:>4}  "
              f"{r['Ksh']:>7.4f}  {r['Kns']:>7.4f}  {r['R']:>8.3f}  "
              f"{r['lrt_p']:>8.4f}{sig:3}  {r['ks_p']:>6.3f}")

    # ── Test 2: Completeness sensitivity (d1=70 fixed) ───────────────────────
    print(f"\n  Completeness sensitivity (d1=70 km):")
    print(f"  {'m0':>4}  {'N':>5}  {'Nsh':>5}  {'Nns':>4}  "
          f"{'Ksh':>7}  {'Kns':>7}  {'R':>8}  {'LRT p':>8}  {'KS p':>6}")
    print(f"  {'-'*72}")

    for m0 in completeness_mags:
        r = run_one(times, mags, T, m0, 70)
        if r is None:
            print(f"  m0={m0}: FAILED")
            continue
        r['sequence'] = seq_name
        r['test'] = 'completeness'
        all_results.append(r)

        sig = "***" if r['lrt_p']<0.001 else ("*" if r['lrt_p']<0.05 else "")
        print(f"  {m0:>4}  {r['N']:>5}  {r['Nsh']:>5}  {r['Nns']:>4}  "
              f"{r['Ksh']:>7.4f}  {r['Kns']:>7.4f}  {r['R']:>8.3f}  "
              f"{r['lrt_p']:>8.4f}{sig:3}  {r['ks_p']:>6.3f}")

# ── Write outputs ─────────────────────────────────────────────────────────────

# CSV
csv_path = os.path.join(OUTDIR, "sensitivity_summary.csv")
if all_results:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

# Formatted table for paper
table_path = os.path.join(OUTDIR, "sensitivity_table_for_paper.txt")
with open(table_path, 'w') as f:
    f.write("SENSITIVITY ANALYSIS — DMDHP RESULTS\n")
    f.write("="*80 + "\n\n")

    for seq_name in [r[0] for r in sequences]:
        seq_results = [r for r in all_results if r.get('sequence') == seq_name]
        if not seq_results:
            continue

        f.write(f"Sequence: {seq_name}\n")
        f.write("-"*80 + "\n\n")

        # Threshold table
        thresh = [r for r in seq_results if r['test'] == 'threshold']
        if thresh:
            f.write("Panel A: Depth threshold sensitivity (m0 = 4.0 fixed)\n\n")
            f.write(f"{'d1 (km)':>8}  {'N':>5}  {'Nsh':>5}  {'Nns':>4}  "
                    f"{'K_sh':>7}  {'K_ns':>7}  {'R':>8}  {'LRT p':>10}  {'KS p':>6}\n")
            f.write("-"*75 + "\n")
            for r in thresh:
                sig = " ***" if r['lrt_p']<0.001 else (" *" if r['lrt_p']<0.05 else "")
                marker = " ← baseline" if r['d1'] == 70 else ""
                f.write(f"{r['d1']:>8}  {r['N']:>5}  {r['Nsh']:>5}  {r['Nns']:>4}  "
                        f"{r['Ksh']:>7.4f}  {r['Kns']:>7.4f}  {r['R']:>8.3f}  "
                        f"{r['lrt_p']:>8.4f}{sig:4}  {r['ks_p']:>6.3f}{marker}\n")
            f.write("\n")

        # Completeness table
        comp = [r for r in seq_results if r['test'] == 'completeness']
        if comp:
            f.write("Panel B: Completeness magnitude sensitivity (d1 = 70 km fixed)\n\n")
            f.write(f"{'m0':>4}  {'N':>5}  {'Nsh':>5}  {'Nns':>4}  "
                    f"{'K_sh':>7}  {'K_ns':>7}  {'R':>8}  {'LRT p':>10}  {'KS p':>6}\n")
            f.write("-"*70 + "\n")
            for r in comp:
                sig = " ***" if r['lrt_p']<0.001 else (" *" if r['lrt_p']<0.05 else "")
                marker = " ← baseline" if r['m0'] == 4.0 else ""
                f.write(f"{r['m0']:>4}  {r['N']:>5}  {r['Nsh']:>5}  {r['Nns']:>4}  "
                        f"{r['Ksh']:>7.4f}  {r['Kns']:>7.4f}  {r['R']:>8.3f}  "
                        f"{r['lrt_p']:>8.4f}{sig:4}  {r['ks_p']:>6.3f}{marker}\n")
            f.write("\n")
        f.write("="*80 + "\n\n")

print(f"Saved: {table_path}")
print("\nDone. Upload sensitivity_results/ to Google Drive and paste results here.")
print("Total time: ~10-20 minutes depending on catalog size.")
