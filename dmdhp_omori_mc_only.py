"""
dmdhp_omori_mc_only.py
=======================
Runs ONLY Part 3 (Monte Carlo) of the Omori-Utsu analysis.
Parts 1 and 2 already completed — results saved in omori_results/

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    python3 dmdhp_omori_mc_only.py

Output: omori_results/mc_omori_summary.txt
"""

import os, math, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

OUTDIR = "omori_results"
os.makedirs(OUTDIR, exist_ok=True)

# ── Core model functions ──────────────────────────────────────────────────────

def loglik_dmdhp_omori(params, times, mags, zones, T, m0):
    mu, c, p, Ksh, Kns, ash, ans = params
    if any(x <= 0 for x in [mu,c,p,Ksh,Kns,ash,ans]): return -1e10
    if p > 3.0 or c > 10.0: return -1e10
    K  = np.where(zones==0, Ksh, Kns)
    a  = np.where(zones==0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    lam = np.full(N, mu)
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += np.sum(ki[:i] * p * c**p / (dt+c)**(p+1))
    if np.any(lam <= 0): return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu*T + np.sum(ki * (1.0 - (c/(dt_r+c))**p))
    return ll - comp


def loglik_mdhp_omori(params, times, mags, T, m0):
    mu, c, p, K, a = params
    if any(x <= 0 for x in [mu,c,p,K,a]): return -1e10
    if p > 3.0 or c > 10.0: return -1e10
    ki  = K * np.exp(a * (mags - m0))
    N   = len(times)
    lam = np.full(N, mu)
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += np.sum(ki[:i] * p * c**p / (dt+c)**(p+1))
    if np.any(lam <= 0): return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu*T + np.sum(ki * (1.0 - (c/(dt_r+c))**p))
    return ll - comp


def fit_dmdhp(times, mags, zones, T, m0, n_starts=10):
    best_ll, best_p = -1e15, None
    rng = np.random.default_rng(int(time.time()*1000) % 2**31)
    for _ in range(n_starts):
        p0 = [rng.uniform(0.3,3.0), rng.uniform(0.001,0.1),
              rng.uniform(0.8,1.3),  rng.uniform(0.05,0.5),
              rng.uniform(0.005,0.15), rng.uniform(0.5,2.2),
              rng.uniform(0.01,1.0)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),
                (1e-4,None),(1e-4,2.25),(1e-4,2.25)]
        try:
            res = minimize(
                lambda x: -loglik_dmdhp_omori(x,times,mags,zones,T,m0),
                p0, method='L-BFGS-B', bounds=bnds,
                options={'maxiter':2000,'ftol':1e-12})
            if -res.fun > best_ll:
                best_ll = -res.fun; best_p = res.x
        except: pass
    return best_p, best_ll


def fit_mdhp(times, mags, T, m0, n_starts=10):
    best_ll, best_p = -1e15, None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [np.random.uniform(0.3,3.0), np.random.uniform(0.001,0.1),
              np.random.uniform(0.8,1.3),  np.random.uniform(0.05,0.5),
              np.random.uniform(0.5,2.2)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),(1e-4,2.25)]
        try:
            res = minimize(
                lambda x: -loglik_mdhp_omori(x,times,mags,T,m0),
                p0, method='L-BFGS-B', bounds=bnds,
                options={'maxiter':2000,'ftol':1e-12})
            if -res.fun > best_ll:
                best_ll = -res.fun; best_p = res.x
        except: pass
    return best_p, best_ll


def simulate_omori(mu, c, p, Ksh, Kns, ash, ans, m0, T,
                   p_sh=0.842, seed=None):
    rng = np.random.default_rng(seed)
    events = []

    def intensity(t):
        lam = mu
        for te, ki in events:
            if te >= t: break
            dt = t - te
            lam += ki * p * c**p / (dt+c)**(p+1)
        return lam

    t = 0.0
    while t < T:
        lam_bar = mu
        for te, ki in events:
            dt = t - te
            lam_bar += ki * p * c**p / (dt+c)**(p+1)
        lam_bar = max(lam_bar * 1.5, mu)

        dt_prop = rng.exponential(1.0/lam_bar)
        t += dt_prop
        if t > T: break

        lam_t = intensity(t)
        if rng.uniform() < lam_t / lam_bar:
            mag  = m0 + rng.exponential(1.0/math.log(10))
            mag  = min(mag, 7.5)
            zone = 0 if rng.uniform() < p_sh else 1
            K    = Ksh if zone==0 else Kns
            a    = ash if zone==0 else ans
            ki   = K * math.exp(a * (mag - m0))
            events.append((t, ki))

    if not events:
        return None

    # Rebuild with zone info
    rng2 = np.random.default_rng(seed)
    times_out, mags_out, zones_out = [], [], []
    t2 = 0.0
    ev_idx = 0
    ki_list = []

    # Re-simulate to get zone assignments
    rng3 = np.random.default_rng((seed or 0) + 99999)
    times_f, mags_f, zones_f = [], [], []
    for ev_t, ev_ki in events:
        times_f.append(ev_t)
        # Recover magnitude and zone from ki
        # Just assign zone probabilistically
        z = 0 if rng3.uniform() < p_sh else 1
        zones_f.append(z)
        # Recover approximate mag
        K = Ksh if z==0 else Kns
        a = ash if z==0 else ans
        if K > 0 and a > 0:
            logki = math.log(ev_ki/K) / a + m0 if ev_ki > 0 else m0
        else:
            logki = m0
        mags_f.append(max(m0, logki))

    if not times_f:
        return None

    times_a  = np.array(times_f)
    mags_a   = np.array(mags_f)
    zones_a  = np.array(zones_f)
    idx      = np.argsort(times_a)
    return times_a[idx], mags_a[idx], zones_a[idx]


# ── Monte Carlo scenarios ─────────────────────────────────────────────────────
# Calibrated to Hinatuan Omori-Utsu fitted params:
# mu=0.475, c=0.0871, p=0.473, Ksh=0.231, ash=1.748, ans=0.0001
# p_sh = 705/837 = 0.842

mc_scenarios = [
    {
        'name': 'A (Hinatuan-type, R=2.0)',
        'mu': 0.475, 'c': 0.0871, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.1155,   # R=2.0
        'ash': 1.748, 'ans': 0.0001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 2.0,
    },
    {
        'name': 'B (Hinatuan-type, R=3.0)',
        'mu': 0.475, 'c': 0.0871, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.077,    # R=3.0
        'ash': 1.748, 'ans': 0.0001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 3.0,
    },
    {
        'name': 'C (Hinatuan-type, R=5.0)',
        'mu': 0.475, 'c': 0.0871, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.0462,   # R=5.0
        'ash': 1.748, 'ans': 0.0001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 5.0,
    },
]

B_mc = 200
mc_results = []

print("="*65)
print("PART 3: Monte Carlo simulation — Omori-Utsu kernel")
print("="*65)

for sc in mc_scenarios:
    print(f"\n  Scenario {sc['name']} (B={B_mc})...")
    t_start = time.time()
    n_sig, n_ok = 0, 0
    R_vals, Nns_vals = [], []

    for b in range(B_mc):
        if b % 25 == 0:
            elapsed = time.time() - t_start
            print(f"    Rep {b}/{B_mc}... ({elapsed:.0f}s elapsed)")

        result = simulate_omori(
            sc['mu'], sc['c'], sc['p'],
            sc['Ksh'], sc['Kns'], sc['ash'], sc['ans'],
            sc['m0'], sc['T'], p_sh=sc['p_sh'], seed=b*7+13
        )
        if result is None: continue
        t_arr, m_arr, z_arr = result
        Nns = int(np.sum(z_arr == 1))
        if len(t_arr) < 20 or Nns < 5: continue

        try:
            p_dm, ll_dm = fit_dmdhp(t_arr, m_arr, z_arr, sc['T'], sc['m0'])
            p_md, ll_md = fit_mdhp(t_arr, m_arr, sc['T'], sc['m0'])
            if p_dm is None or p_md is None: continue

            lrt   = 2*(ll_dm - ll_md)
            lrt_p = 1 - chi2.cdf(max(lrt, 0), df=2)
            R_hat = p_dm[3]/p_dm[4] if p_dm[4] > 1e-8 else None

            n_ok += 1
            Nns_vals.append(Nns)
            if lrt_p < 0.05:
                n_sig += 1
            if R_hat is not None and 0 < R_hat < 1000:
                R_vals.append(R_hat)
        except:
            pass

    power   = n_sig / n_ok if n_ok > 0 else 0.0
    med_R   = float(np.median(R_vals)) if R_vals else None
    mean_Nns = float(np.mean(Nns_vals)) if Nns_vals else None

    # Fixed f-string — no conditional inside format spec
    med_R_str   = f"{med_R:.3f}"   if med_R   is not None else "N/A"
    nns_str     = f"{mean_Nns:.1f}" if mean_Nns is not None else "N/A"

    print(f"    DONE: n_ok={n_ok}, power={power:.3f}, "
          f"med_R={med_R_str}, mean_Nns={nns_str}")

    mc_results.append({
        'scenario':  sc['name'],
        'true_R':    sc['true_R'],
        'n_ok':      n_ok,
        'n_sig':     n_sig,
        'power':     power,
        'med_R':     med_R,
        'mean_Nns':  mean_Nns,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(OUTDIR, "mc_omori_summary.txt")
with open(out, 'w') as f:
    f.write("Monte Carlo Summary — Omori-Utsu kernel DMDHP\n")
    f.write("Calibrated to Hinatuan fitted params: "
            "mu=0.475, c=0.0871, p=0.473\n")
    f.write("="*65 + "\n")
    f.write(f"{'Scenario':<32} {'True R':>7} {'n_ok':>5} "
            f"{'Power':>7} {'Med R':>7} {'Nns':>6}\n")
    f.write("-"*65 + "\n")
    for r in mc_results:
        med_str = f"{r['med_R']:.3f}" if r['med_R'] is not None else "N/A"
        nns_str = f"{r['mean_Nns']:.1f}" if r['mean_Nns'] is not None else "N/A"
        f.write(f"{r['scenario']:<32} {r['true_R']:>7.1f} {r['n_ok']:>5} "
                f"{r['power']:>7.3f} {med_str:>7} {nns_str:>6}\n")

print(f"\nSaved: {out}")
print("\n" + "="*65)
print("ALL DONE. Upload omori_results/mc_omori_summary.txt to Google Drive.")
print("="*65)
