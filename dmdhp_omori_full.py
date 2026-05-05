"""
dmdhp_omori_full.py
====================
Full Omori-Utsu kernel DMDHP analysis:
  1. All Philippine + global sequences (primary results)
  2. Bootstrap CIs for R (B=200, Hinatuan only — others too sparse)
  3. Monte Carlo simulation: 3 scenarios with Omori-Utsu kernel

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    python3 dmdhp_omori_full.py

Outputs in: omori_results/
  primary_results.txt     — main table for paper
  bootstrap_hinatuan.txt  — 95% CI for R under Omori-Utsu
  mc_omori_summary.txt    — MC power under Omori-Utsu
  primary_results.csv     — CSV version

Author: J.P. Arcede (Caraga State University)
"""

import os, math, json, csv, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kstest
from multiprocessing import Pool, cpu_count

OUTDIR = "omori_results"
os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CORE MODEL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


def fit_dmdhp(times, mags, zones, T, m0, n_starts=20):
    best_ll, best_p = -1e15, None
    np.random.seed(int(time.time()*1000) % 2**31)
    for _ in range(n_starts):
        p0 = [np.random.uniform(0.3,3.0),
              np.random.uniform(0.001,0.1),
              np.random.uniform(0.8,1.3),
              np.random.uniform(0.05,0.5),
              np.random.uniform(0.005,0.15),
              np.random.uniform(0.5,2.2),
              np.random.uniform(0.01,1.0)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),
                (1e-4,None),(1e-4,2.25),(1e-4,2.25)]
        try:
            res = minimize(lambda x: -loglik_dmdhp_omori(x,times,mags,zones,T,m0),
                           p0, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter':3000,'ftol':1e-13})
            if -res.fun > best_ll:
                best_ll = -res.fun; best_p = res.x
        except: pass
    return best_p, best_ll


def fit_mdhp(times, mags, T, m0, n_starts=20):
    best_ll, best_p = -1e15, None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [np.random.uniform(0.3,3.0),
              np.random.uniform(0.001,0.1),
              np.random.uniform(0.8,1.3),
              np.random.uniform(0.05,0.5),
              np.random.uniform(0.5,2.2)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),(1e-4,2.25)]
        try:
            res = minimize(lambda x: -loglik_mdhp_omori(x,times,mags,T,m0),
                           p0, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter':3000,'ftol':1e-13})
            if -res.fun > best_ll:
                best_ll = -res.fun; best_p = res.x
        except: pass
    return best_p, best_ll


def pit_ks(params, times, mags, zones, T, m0):
    mu, c, p, Ksh, Kns, ash, ans = params
    K  = np.where(zones==0, Ksh, Kns)
    a  = np.where(zones==0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    taus = np.zeros(N)
    for i in range(N):
        t0 = times[i-1] if i>0 else 0.0
        t1 = times[i]
        taus[i] = mu*(t1-t0)
        for j in range(i):
            dt0 = t0 - times[j]
            dt1 = t1 - times[j]
            taus[i] += ki[j] * ((c/(dt0+c))**p - (c/(dt1+c))**p)
    U = 1.0 - np.exp(-taus)
    _, ks_p = kstest(U, 'uniform')
    return ks_p


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION (Omori-Utsu thinning)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_omori(mu, c, p, Ksh, Kns, ash, ans, m0, T,
                   p_sh=0.85, d_sh=35.0, d_ns=150.0, seed=None):
    """Simulate DMDHP with Omori-Utsu kernel using Ogata thinning."""
    rng = np.random.default_rng(seed)
    events = []  # (time, mag, zone, ki)

    def intensity(t):
        lam = mu
        for te, me, ze, ki in events:
            if te >= t: break
            dt = t - te
            lam += ki * p * c**p / (dt+c)**(p+1)
        return lam

    t = 0.0
    while t < T:
        # Upper bound on intensity
        lam_bar = mu
        for te, me, ze, ki in events:
            dt = t - te
            lam_bar += ki * p * c**p / (dt+c)**(p+1)
        lam_bar *= 1.5  # safety factor

        if lam_bar <= 0: break
        dt_prop = rng.exponential(1.0/lam_bar)
        t += dt_prop
        if t > T: break

        lam_t = intensity(t)
        if rng.uniform() < lam_t / lam_bar:
            # Accept event
            mag = m0 + rng.exponential(1.0/math.log(10))
            mag = min(mag, 7.5)
            zone = 0 if rng.uniform() < p_sh else 1
            K  = Ksh if zone==0 else Kns
            a  = ash if zone==0 else ans
            ki = K * math.exp(a * (mag - m0))
            events.append((t, mag, zone, ki))

    if not events:
        return None
    times  = np.array([e[0] for e in events])
    mags   = np.array([e[1] for e in events])
    zones  = np.array([e[2] for e in events])
    depths = np.where(zones==0, d_sh, d_ns)
    return times, mags, zones, depths


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CATALOG
# ══════════════════════════════════════════════════════════════════════════════

def load_catalog(filepath, m0_filter=4.0):
    import csv as _csv
    times_out, mags_out, depths_out = [], [], []
    with open(filepath, newline='') as f:
        sample = f.read(2048); f.seek(0)
        dialect = _csv.Sniffer().sniff(sample)
        reader  = _csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames
        def col(names):
            for n in names:
                for h in headers:
                    if n.lower() in h.lower(): return h
            return None
        tc = col(['t_day','time_day','days','time'])
        mc = col(['mag','magnitude','mw','ml'])
        dc = col(['depth','dep'])
        for row in reader:
            try:
                t,m,d = float(row[tc]),float(row[mc]),float(row[dc])
                if m >= m0_filter:
                    times_out.append(t); mags_out.append(m); depths_out.append(d)
            except: pass
    idx = np.argsort(times_out)
    return (np.array(times_out)[idx],
            np.array(mags_out)[idx],
            np.array(depths_out)[idx])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: PRIMARY RESULTS — ALL SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

sequences = [
    ("Hinatuan 2023",      "ph_catalogs/SEQ1_Hinatuan2023_catalog.csv",       4.0, 70),
    ("Davao Oriental 2025","ph_catalogs/SEQ2_DavaoOriental2025_catalog.csv",   4.0, 70),
    ("Davao del Sur 2019", "ph_catalogs/SEQ3_DavaoDeSur2019_catalog.csv",      4.0, 70),
    ("Cotabato Oct29 2019","ph_catalogs/SEQ4_Cotabato2019Oct29_catalog.csv",   4.0, 70),
    ("Cotabato Oct16 2019","ph_catalogs/SEQ5_Cotabato2019Oct16_catalog.csv",   4.0, 70),
    ("Bohol 2013",         "ph_catalogs/SEQ8_Bohol2013_catalog.csv",           4.0, 70),
    ("Tohoku 2011",        "global_catalogs/G01_Tohoku2011_catalog.csv",       4.5, 70),
    ("Chiapas 2017",       "global_catalogs/G10_Mexico2017_catalog.csv",       4.5, 70),
]

print("="*70)
print("PART 1: Primary results — Omori-Utsu kernel DMDHP")
print("="*70)

primary_results = []

for seq_name, cat_path, m0, d1 in sequences:
    if not os.path.exists(cat_path):
        print(f"  SKIP {seq_name}: {cat_path} not found")
        continue

    times, mags, depths = load_catalog(cat_path, m0)
    zones = (depths >= d1).astype(int)
    T     = times.max()
    N     = len(times)
    Nsh   = int(np.sum(zones==0))
    Nns   = int(np.sum(zones==1))

    print(f"\n  {seq_name} (N={N}, Nns={Nns})...")

    if Nns < 10:
        print(f"    → 1-zone only (Nns={Nns})")
        # Fit MDHP only
        p_md, ll_md = fit_mdhp(times, mags, T, m0)
        mu_m,c_m,p_m,K_m,a_m = p_md
        primary_results.append({
            'sequence': seq_name, 'Mw': '', 'mech': '',
            'N': N, 'Nsh': Nsh, 'Nns': Nns,
            'Ksh': K_m, 'Kns': None, 'R': None,
            'c': c_m, 'p_omori': p_m,
            'lrt_p': None, 'ks_p': None,
            'zone': '1Z'
        })
        continue

    p_dm, ll_dm = fit_dmdhp(times, mags, zones, T, m0)
    p_md, ll_md = fit_mdhp(times, mags, T, m0)

    if p_dm is None:
        print(f"    → fit failed")
        continue

    mu_d,c_d,p_d,Ksh,Kns,ash,ans = p_dm
    R     = Ksh/Kns if Kns > 1e-8 else float('inf')
    lrt   = 2*(ll_dm - ll_md)
    lrt_p = 1 - chi2.cdf(max(lrt,0), df=2)
    ks_p  = pit_ks(p_dm, times, mags, zones, T, m0)

    aic_dm = -2*ll_dm + 2*7
    aic_md = -2*ll_md + 2*5

    sig = "***" if lrt_p<0.001 else ("**" if lrt_p<0.01 else
          ("*" if lrt_p<0.05 else ""))
    print(f"    R={R:.3f}, LRT p={lrt_p:.4f}{sig}, KS p={ks_p:.3f}, "
          f"c={c_d:.4f}, p={p_d:.3f}")

    primary_results.append({
        'sequence': seq_name, 'N': N, 'Nsh': Nsh, 'Nns': Nns,
        'Ksh': Ksh, 'Kns': Kns, 'R': R,
        'ash': ash, 'ans': ans, 'c': c_d, 'p_omori': p_d,
        'lrt_p': lrt_p, 'ks_p': ks_p,
        'AIC_DMDHP': aic_dm, 'AIC_MDHP': aic_md,
        'zone': '2Z' if Nns >= 20 else '1Z'
    })

# Save primary results
out1 = os.path.join(OUTDIR, "primary_results.txt")
with open(out1, 'w') as f:
    f.write("PRIMARY RESULTS — Omori-Utsu kernel DMDHP\n")
    f.write("="*75 + "\n")
    f.write(f"{'Sequence':<24} {'N':>5} {'Nsh':>5} {'Nns':>4} "
            f"{'Ksh':>7} {'Kns':>7} {'R':>7} {'LRT p':>10} "
            f"{'KS p':>6} {'c':>7} {'p':>5}\n")
    f.write("-"*75 + "\n")
    for r in primary_results:
        if r.get('zone') == '1Z':
            f.write(f"{r['sequence']:<24} {r['N']:>5} {r['Nsh']:>5} "
                    f"{r['Nns']:>4} {r['Ksh']:>7.4f}  1-zone only\n")
        else:
            sig = ("***" if r['lrt_p']<0.001 else ("**" if r['lrt_p']<0.01
                   else ("*" if r['lrt_p']<0.05 else "")))
            f.write(f"{r['sequence']:<24} {r['N']:>5} {r['Nsh']:>5} "
                    f"{r['Nns']:>4} {r['Ksh']:>7.4f} {r['Kns']:>7.4f} "
                    f"{r['R']:>7.3f} {r['lrt_p']:>8.4f}{sig:4} "
                    f"{r['ks_p']:>6.3f} {r['c']:>7.5f} {r['p_omori']:>5.3f}\n")
print(f"\nSaved: {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: BOOTSTRAP CI FOR HINATUAN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PART 2: Bootstrap CIs for Hinatuan (B=200)")
print("="*70)

hinatuan_result = next((r for r in primary_results
                        if 'Hinatuan' in r['sequence']), None)

if hinatuan_result and hinatuan_result.get('zone') == '2Z':
    cat_path = "ph_catalogs/SEQ1_Hinatuan2023_catalog.csv"
    times, mags, depths = load_catalog(cat_path, 4.0)
    zones = (depths >= 70).astype(int)
    T     = times.max()

    # Get fitted params
    r = hinatuan_result
    fitted_params = None
    # Re-fit to get full params
    p_dm, ll_dm = fit_dmdhp(times, mags, zones, T, 4.0, n_starts=25)
    mu_d,c_d,p_d,Ksh,Kns,ash,ans = p_dm

    print(f"  Fitted: R={Ksh/Kns:.3f}, c={c_d:.5f}, p={p_d:.3f}")
    print(f"  Running B=200 bootstrap replications...")

    B = 200
    R_boots = []
    p_sh = np.sum(zones==0) / len(zones)

    for b in range(B):
        if b % 20 == 0:
            print(f"    Bootstrap {b}/{B}...")
        result = simulate_omori(mu_d, c_d, p_d, Ksh, Kns, ash, ans,
                                4.0, T, p_sh=p_sh, seed=b)
        if result is None: continue
        bt, bm, bz, bd = result
        if len(bt) < 30 or np.sum(bz==1) < 5: continue
        try:
            bp, bll = fit_dmdhp(bt, bm, bz, T, 4.0, n_starts=10)
            if bp is not None:
                bR = bp[3]/bp[4] if bp[4] > 1e-8 else None
                if bR is not None and bR < 1e5:
                    R_boots.append(bR)
        except: pass

    R_boots = np.array(R_boots)
    ci_lo = np.percentile(R_boots, 2.5)
    ci_hi = np.percentile(R_boots, 97.5)
    med_R = np.median(R_boots)

    print(f"\n  Bootstrap results (n={len(R_boots)} valid):")
    print(f"  R_hat = {Ksh/Kns:.3f}")
    print(f"  Median bootstrap R = {med_R:.3f}")
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    out2 = os.path.join(OUTDIR, "bootstrap_hinatuan.txt")
    with open(out2, 'w') as f:
        f.write("Bootstrap CIs — Omori-Utsu DMDHP, Hinatuan 2023\n")
        f.write("="*50 + "\n")
        f.write(f"R_hat = {Ksh/Kns:.4f}\n")
        f.write(f"Median bootstrap R = {med_R:.4f}\n")
        f.write(f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"n_valid bootstrap = {len(R_boots)}\n")
        f.write(f"c = {c_d:.5f}, p = {p_d:.4f}\n")
    print(f"\nSaved: {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: MONTE CARLO — Omori-Utsu kernel
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PART 3: Monte Carlo simulation — Omori-Utsu kernel")
print("="*70)

# Calibrate to Hinatuan fitted params
# From omori_lrt_results: Hinatuan c=0.08707, p=0.473
# Use these as true parameters for Scenario A
mc_scenarios = [
    {
        'name': 'A (Hinatuan-type, R=2.0)',
        'mu': 2.0, 'c': 0.087, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.116,   # R=2.0
        'ash': 1.748, 'ans': 0.001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 2.0,
    },
    {
        'name': 'B (Hinatuan-type, R=3.0)',
        'mu': 2.0, 'c': 0.087, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.077,   # R=3.0
        'ash': 1.748, 'ans': 0.001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 3.0,
    },
    {
        'name': 'C (Hinatuan-type, R=5.0)',
        'mu': 2.0, 'c': 0.087, 'p': 0.473,
        'Ksh': 0.231, 'Kns': 0.046,   # R=5.0
        'ash': 1.748, 'ans': 0.001,
        'm0': 4.0, 'T': 88.2, 'p_sh': 0.842,
        'true_R': 5.0,
    },
]

B_mc = 200
mc_results = []

for sc in mc_scenarios:
    print(f"\n  Scenario {sc['name']} (B={B_mc})...")
    n_sig = 0
    n_ok  = 0
    R_vals = []

    for b in range(B_mc):
        result = simulate_omori(
            sc['mu'], sc['c'], sc['p'],
            sc['Ksh'], sc['Kns'], sc['ash'], sc['ans'],
            sc['m0'], sc['T'], p_sh=sc['p_sh'], seed=b*7+13
        )
        if result is None: continue
        t, m, z, d = result
        if len(t) < 20 or np.sum(z==1) < 5: continue

        try:
            p_dm, ll_dm = fit_dmdhp(t, m, z, sc['T'], sc['m0'], n_starts=10)
            p_md, ll_md = fit_mdhp(t, m, sc['T'], sc['m0'], n_starts=10)
            if p_dm is None or p_md is None: continue

            lrt   = 2*(ll_dm - ll_md)
            lrt_p = 1 - chi2.cdf(max(lrt,0), df=2)
            R_hat = p_dm[3]/p_dm[4] if p_dm[4] > 1e-8 else None

            n_ok += 1
            if lrt_p < 0.05: n_sig += 1
            if R_hat is not None and R_hat < 1000:
                R_vals.append(R_hat)
        except: pass

    power = n_sig/n_ok if n_ok > 0 else 0
    med_R = np.median(R_vals) if R_vals else None
    print(f"    n_ok={n_ok}, power={power:.3f}, med_R={med_R:.3f if med_R else 'N/A'}")

    mc_results.append({
        'scenario': sc['name'],
        'true_R': sc['true_R'],
        'n_ok': n_ok,
        'power': power,
        'med_R': med_R,
    })

out3 = os.path.join(OUTDIR, "mc_omori_summary.txt")
with open(out3, 'w') as f:
    f.write("Monte Carlo Summary — Omori-Utsu kernel DMDHP\n")
    f.write("="*60 + "\n")
    f.write(f"{'Scenario':<30} {'True R':>7} {'n_ok':>5} "
            f"{'Power':>7} {'Med R':>7}\n")
    f.write("-"*60 + "\n")
    for r in mc_results:
        f.write(f"{r['scenario']:<30} {r['true_R']:>7.1f} {r['n_ok']:>5} "
                f"{r['power']:>7.3f} "
                f"{r['med_R']:>7.3f if r['med_R'] else 'N/A':>7}\n")

print(f"\nSaved: {out3}")
print("\n" + "="*70)
print("ALL DONE. Upload omori_results/ to Google Drive.")
print("="*70)
