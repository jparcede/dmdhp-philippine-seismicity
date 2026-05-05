"""
dmdhp_omori_mc_v2.py
=====================
Fixed Monte Carlo simulation — Omori-Utsu kernel DMDHP.
Cleaner simulation engine that correctly generates catalogs.

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    nohup python3 dmdhp_omori_mc_v2.py > mc_v2.log 2>&1 &

Output: omori_results/mc_omori_summary.txt
"""

import os, math, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

OUTDIR = "omori_results"
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def loglik_dmdhp(params, times, mags, zones, T, m0):
    mu, c, p, Ksh, Kns, ash, ans = params
    if any(x <= 0 for x in [mu,c,p,Ksh,Kns,ash,ans]): return -1e10
    if p > 3.0 or c > 10.0: return -1e10
    K  = np.where(zones==0, Ksh, Kns)
    a  = np.where(zones==0, ash, ans)
    ki = K * np.exp(np.clip(a*(mags-m0), -20, 20))
    N  = len(times)
    lam = np.full(N, mu, dtype=float)
    for i in range(N):
        dt = times[i] - times[:i]
        contrib = ki[:i] * p * c**p / (dt + c)**(p+1)
        lam[i] += np.sum(contrib)
    if np.any(lam <= 0): return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu*T + np.sum(ki * (1.0 - (c/(dt_r+c))**p))
    if not np.isfinite(ll - comp): return -1e10
    return ll - comp


def loglik_mdhp(params, times, mags, T, m0):
    mu, c, p, K, a = params
    if any(x <= 0 for x in [mu,c,p,K,a]): return -1e10
    if p > 3.0 or c > 10.0: return -1e10
    ki  = K * np.exp(np.clip(a*(mags-m0), -20, 20))
    N   = len(times)
    lam = np.full(N, mu, dtype=float)
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += np.sum(ki[:i] * p * c**p / (dt+c)**(p+1))
    if np.any(lam <= 0): return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu*T + np.sum(ki * (1.0 - (c/(dt_r+c))**p))
    if not np.isfinite(ll - comp): return -1e10
    return ll - comp


def fit_dmdhp(times, mags, zones, T, m0, n_starts=10):
    best_ll, best_p = -1e15, None
    rng = np.random.default_rng(int(time.time()*1e6) % 2**32)
    for _ in range(n_starts):
        p0 = [rng.uniform(0.1, 2.0),
              rng.uniform(0.005, 0.2),
              rng.uniform(0.7, 1.5),
              rng.uniform(0.02, 0.5),
              rng.uniform(0.005, 0.3),
              rng.uniform(0.3, 2.2),
              rng.uniform(0.0001, 0.5)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3.0),
                (1e-4,None),(1e-4,None),(1e-4,2.25),(1e-4,2.25)]
        try:
            res = minimize(
                lambda x: -loglik_dmdhp(x, times, mags, zones, T, m0),
                p0, method='L-BFGS-B', bounds=bnds,
                options={'maxiter':1500, 'ftol':1e-11})
            if np.isfinite(-res.fun) and -res.fun > best_ll:
                best_ll = -res.fun
                best_p  = res.x
        except Exception:
            pass
    return best_p, best_ll


def fit_mdhp(times, mags, T, m0, n_starts=10):
    best_ll, best_p = -1e15, None
    rng = np.random.default_rng(42)
    for _ in range(n_starts):
        p0 = [rng.uniform(0.1, 2.0),
              rng.uniform(0.005, 0.2),
              rng.uniform(0.7, 1.5),
              rng.uniform(0.02, 0.5),
              rng.uniform(0.3, 2.2)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3.0),(1e-4,None),(1e-4,2.25)]
        try:
            res = minimize(
                lambda x: -loglik_mdhp(x, times, mags, T, m0),
                p0, method='L-BFGS-B', bounds=bnds,
                options={'maxiter':1500, 'ftol':1e-11})
            if np.isfinite(-res.fun) and -res.fun > best_ll:
                best_ll = -res.fun
                best_p  = res.x
        except Exception:
            pass
    return best_p, best_ll


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_catalog(mu, c, p, Ksh, Kns, ash, ans, m0, T,
                     p_sh=0.842, seed=None):
    """
    Simulate Omori-Utsu DMDHP using modified thinning algorithm.
    Returns (times, mags, zones) arrays sorted by time, or None if empty.
    """
    rng = np.random.default_rng(seed)

    # Store events as list of dicts for clarity
    events = []  # each: {'t': float, 'mag': float, 'zone': int, 'ki': float}

    def current_intensity(t):
        lam = mu
        for ev in events:
            if ev['t'] >= t:
                break
            dt = t - ev['t']
            lam += ev['ki'] * p * c**p / (dt + c)**(p+1)
        return lam

    t = 0.0
    max_events = 5000  # safety cap

    while t < T and len(events) < max_events:
        # Compute upper bound on intensity from t
        lam_ub = mu
        for ev in events:
            dt = t - ev['t']
            # Omori-Utsu is decreasing so current value is upper bound
            lam_ub += ev['ki'] * p * c**p / (dt + c)**(p+1)
        lam_ub = max(lam_ub * 2.0, mu * 2.0)  # safety factor

        # Sample next candidate time
        dt_next = rng.exponential(1.0 / lam_ub)
        t_cand  = t + dt_next

        if t_cand > T:
            break

        # Acceptance step
        lam_cand = current_intensity(t_cand)
        if rng.uniform() <= lam_cand / lam_ub:
            # Accept — generate event attributes
            mag  = m0 + rng.exponential(1.0 / math.log(10))
            mag  = min(mag, 7.5)
            zone = 0 if rng.uniform() < p_sh else 1
            K_z  = Ksh if zone == 0 else Kns
            a_z  = ash if zone == 0 else ans
            ki   = K_z * math.exp(min(a_z * (mag - m0), 20))
            events.append({'t': t_cand, 'mag': mag, 'zone': zone, 'ki': ki})

        t = t_cand

    if len(events) < 5:
        return None

    times = np.array([e['t']    for e in events])
    mags  = np.array([e['mag']  for e in events])
    zones = np.array([e['zone'] for e in events], dtype=int)

    return times, mags, zones


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SCENARIOS
# Calibrated to Hinatuan Omori-Utsu fitted params:
# mu=0.475, c=0.0871, p=0.473, ash=1.748, ans=0.0001, p_sh=0.842
# ══════════════════════════════════════════════════════════════════════════════

scenarios = [
    {'name': 'A (R=2.0)', 'true_R': 2.0,
     'mu': 0.475, 'c': 0.0871, 'p': 0.473,
     'Ksh': 0.231, 'Kns': 0.1155,
     'ash': 1.748, 'ans': 0.0001,
     'm0': 4.0,    'T': 88.2,    'p_sh': 0.842},
    {'name': 'B (R=3.0)', 'true_R': 3.0,
     'mu': 0.475, 'c': 0.0871, 'p': 0.473,
     'Ksh': 0.231, 'Kns': 0.077,
     'ash': 1.748, 'ans': 0.0001,
     'm0': 4.0,    'T': 88.2,    'p_sh': 0.842},
    {'name': 'C (R=5.0)', 'true_R': 5.0,
     'mu': 0.475, 'c': 0.0871, 'p': 0.473,
     'Ksh': 0.231, 'Kns': 0.0462,
     'ash': 1.748, 'ans': 0.0001,
     'm0': 4.0,    'T': 88.2,    'p_sh': 0.842},
]

B = 200
results = []

print("=" * 65)
print("Monte Carlo — Omori-Utsu DMDHP (v2 fixed simulation)")
print("=" * 65)

# Quick sanity check: simulate one catalog and print stats
print("\nSanity check — simulating one catalog from Scenario A...")
test = simulate_catalog(
    0.475, 0.0871, 0.473, 0.231, 0.1155, 1.748, 0.0001,
    4.0, 88.2, p_sh=0.842, seed=0
)
if test is not None:
    t_t, m_t, z_t = test
    print(f"  N={len(t_t)}, Nsh={np.sum(z_t==0)}, Nns={np.sum(z_t==1)}, "
          f"T_max={t_t.max():.1f}")
    print("  Sanity check PASSED ✅")
else:
    print("  Sanity check FAILED — no events generated ❌")
    print("  Check parameters. Exiting.")
    exit(1)

print()

for sc in scenarios:
    print(f"\nScenario {sc['name']} (B={B})...")
    t0      = time.time()
    n_ok    = 0
    n_sig   = 0
    R_vals  = []
    Nns_vals = []

    for b in range(B):
        if b > 0 and b % 25 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / b * (B - b)
            print(f"  Rep {b:3d}/{B} | {elapsed:5.0f}s elapsed | "
                  f"ETA {eta:5.0f}s | power so far: "
                  f"{n_sig}/{n_ok}={n_sig/max(n_ok,1):.3f}")

        cat = simulate_catalog(
            sc['mu'], sc['c'], sc['p'],
            sc['Ksh'], sc['Kns'], sc['ash'], sc['ans'],
            sc['m0'], sc['T'], p_sh=sc['p_sh'], seed=b * 31 + 7
        )
        if cat is None:
            continue

        times, mags, zones = cat
        Nns = int(np.sum(zones == 1))

        if len(times) < 15 or Nns < 3:
            continue

        try:
            p_dm, ll_dm = fit_dmdhp(times, mags, zones, sc['T'], sc['m0'])
            p_md, ll_md = fit_mdhp(times, mags,          sc['T'], sc['m0'])

            if p_dm is None or p_md is None:
                continue
            if not (np.isfinite(ll_dm) and np.isfinite(ll_md)):
                continue

            lrt   = max(2.0 * (ll_dm - ll_md), 0.0)
            lrt_p = 1.0 - chi2.cdf(lrt, df=2)
            Ksh_h, Kns_h = p_dm[3], p_dm[4]
            R_hat = Ksh_h / Kns_h if Kns_h > 1e-8 else None

            n_ok += 1
            Nns_vals.append(Nns)
            if lrt_p < 0.05:
                n_sig += 1
            if R_hat is not None and 0 < R_hat < 1e4:
                R_vals.append(R_hat)

        except Exception:
            pass

    power     = n_sig / n_ok if n_ok > 0 else 0.0
    med_R     = float(np.median(R_vals))   if R_vals   else None
    mean_Nns  = float(np.mean(Nns_vals))   if Nns_vals else None

    med_R_s  = f"{med_R:.3f}"    if med_R    is not None else "N/A"
    nns_s    = f"{mean_Nns:.1f}" if mean_Nns is not None else "N/A"

    print(f"\n  DONE: n_ok={n_ok}/{B}, power={power:.3f}, "
          f"med_R={med_R_s}, mean_Nns={nns_s}")

    results.append({
        'scenario': sc['name'], 'true_R': sc['true_R'],
        'n_ok': n_ok, 'n_sig': n_sig,
        'power': power, 'med_R': med_R, 'mean_Nns': mean_Nns,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(OUTDIR, "mc_omori_summary.txt")
with open(out, 'w') as f:
    f.write("Monte Carlo Summary — Omori-Utsu kernel DMDHP (v2)\n")
    f.write("Calibrated to Hinatuan: mu=0.475, c=0.0871, p=0.473\n")
    f.write("=" * 65 + "\n")
    f.write(f"{'Scenario':<22} {'True R':>7} {'n_ok':>5} "
            f"{'Power':>7} {'Med R':>8} {'Nns':>7}\n")
    f.write("-" * 65 + "\n")
    for r in results:
        ms  = f"{r['med_R']:.3f}"    if r['med_R']    is not None else "N/A"
        ns  = f"{r['mean_Nns']:.1f}" if r['mean_Nns'] is not None else "N/A"
        f.write(f"{r['scenario']:<22} {r['true_R']:>7.1f} {r['n_ok']:>5} "
                f"{r['power']:>7.3f} {ms:>8} {ns:>7}\n")

print(f"\nSaved: {out}")
print("\n" + "=" * 65)
print("ALL DONE. Upload omori_results/mc_omori_summary.txt to Google Drive.")
print("=" * 65)
