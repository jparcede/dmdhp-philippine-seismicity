"""
dmdhp_omori_lrt.py
==================
Fits both 2-zone DMDHP and baseline MDHP with Omori-Utsu kernel
and computes the LRT p-value for the Hinatuan sequence.

Also fits the Omori-Utsu DMDHP to all Philippine sequences.

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    python3 dmdhp_omori_lrt.py
"""

import os, math, csv
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kstest

OUTDIR = "sensitivity_results"
os.makedirs(OUTDIR, exist_ok=True)

# ── Omori-Utsu kernel functions ───────────────────────────────────────────────

def loglik_dmdhp_omori(params, times, mags, zones, T, m0):
    mu, c, p, Ksh, Kns, ash, ans = params
    if any(x <= 0 for x in [mu, c, p, Ksh, Kns, ash, ans]):
        return -1e10
    if p > 3.0 or c > 10.0:
        return -1e10
    K  = np.where(zones == 0, Ksh, Kns)
    a  = np.where(zones == 0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    lam = np.full(N, mu)
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += np.sum(ki[:i] * p * c**p / (dt + c)**(p+1))
    if np.any(lam <= 0):
        return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu * T + np.sum(ki * (1.0 - (c / (dt_r + c))**p))
    return ll - comp


def loglik_mdhp_omori(params, times, mags, T, m0):
    mu, c, p, K, a = params
    if any(x <= 0 for x in [mu, c, p, K, a]):
        return -1e10
    if p > 3.0 or c > 10.0:
        return -1e10
    ki  = K * np.exp(a * (mags - m0))
    N   = len(times)
    lam = np.full(N, mu)
    for i in range(N):
        dt = times[i] - times[:i]
        lam[i] += np.sum(ki[:i] * p * c**p / (dt + c)**(p+1))
    if np.any(lam <= 0):
        return -1e10
    ll   = np.sum(np.log(lam))
    dt_r = T - times
    comp = mu * T + np.sum(ki * (1.0 - (c / (dt_r + c))**p))
    return ll - comp


def fit_dmdhp_omori(times, mags, zones, T, m0, n_starts=20):
    best_ll, best_p = -1e15, None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [np.random.uniform(0.5,3.0),   # mu
              np.random.uniform(0.001,0.05), # c
              np.random.uniform(0.9,1.3),    # p
              np.random.uniform(0.05,0.5),   # Ksh
              np.random.uniform(0.01,0.2),   # Kns
              np.random.uniform(0.5,2.0),    # ash
              np.random.uniform(0.1,1.5)]    # ans
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),
                (1e-4,None),(1e-4,2.25),(1e-4,2.25)]
        res = minimize(lambda x: -loglik_dmdhp_omori(x,times,mags,zones,T,m0),
                       p0, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter':3000,'ftol':1e-13})
        if -res.fun > best_ll:
            best_ll = -res.fun
            best_p  = res.x
    return best_p, best_ll


def fit_mdhp_omori(times, mags, T, m0, n_starts=20):
    best_ll, best_p = -1e15, None
    np.random.seed(42)
    for _ in range(n_starts):
        p0 = [np.random.uniform(0.5,3.0),
              np.random.uniform(0.001,0.05),
              np.random.uniform(0.9,1.3),
              np.random.uniform(0.05,0.5),
              np.random.uniform(0.5,2.0)]
        bnds = [(1e-4,None),(1e-5,10),(0.1,3),(1e-4,None),(1e-4,2.25)]
        res = minimize(lambda x: -loglik_mdhp_omori(x,times,mags,T,m0),
                       p0, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter':3000,'ftol':1e-13})
        if -res.fun > best_ll:
            best_ll = -res.fun
            best_p  = res.x
    return best_p, best_ll


def pit_ks_omori(params, times, mags, zones, T, m0):
    mu, c, p, Ksh, Kns, ash, ans = params
    K  = np.where(zones == 0, Ksh, Kns)
    a  = np.where(zones == 0, ash, ans)
    ki = K * np.exp(a * (mags - m0))
    N  = len(times)
    # numerical integration via cumulative sum
    taus = np.zeros(N)
    for i in range(N):
        t0 = times[i-1] if i > 0 else 0.0
        t1 = times[i]
        # background
        taus[i] = mu * (t1 - t0)
        # triggering
        for j in range(i):
            dt0 = t0 - times[j]
            dt1 = t1 - times[j]
            # integral of p*c^p/(t+c)^(p+1) from dt0 to dt1
            # = (c/(dt0+c))^p - (c/(dt1+c))^p
            contrib = (c/(dt0+c))**p - (c/(dt1+c))**p
            taus[i] += ki[j] * contrib
    U = 1.0 - np.exp(-taus)
    _, ks_p = kstest(U, 'uniform')
    return ks_p


# ── Load catalog ──────────────────────────────────────────────────────────────

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


# ── Sequences ─────────────────────────────────────────────────────────────────

sequences = [
    ("Hinatuan 2023",      "ph_catalogs/SEQ1_Hinatuan2023_catalog.csv",      70, 4.0),
    ("Davao Oriental 2025","ph_catalogs/SEQ2_DavaoOriental2025_catalog.csv",  70, 4.0),
    ("Davao del Sur 2019", "ph_catalogs/SEQ3_DavaoDeSur2019_catalog.csv",     70, 4.0),
]

results = []
print("="*65)
print("Omori-Utsu kernel DMDHP — LRT results")
print("="*65)
print(f"{'Sequence':<24} {'Nns':>4} {'Ksh':>7} {'Kns':>7} "
      f"{'R':>7} {'LRT p':>9} {'KS p':>6}")
print("-"*65)

for seq_name, cat_path, d1, m0 in sequences:
    if not os.path.exists(cat_path):
        print(f"{seq_name:<24}: catalog not found")
        continue

    times, mags, depths = load_catalog(cat_path, m0)
    zones = (depths >= d1).astype(int)
    T     = times.max()
    Nns   = int(np.sum(zones == 1))

    if Nns < 20:
        print(f"{seq_name:<24} {Nns:>4}  --- 1-zone only ---")
        continue

    print(f"  Fitting {seq_name} (N={len(times)}, Nns={Nns})...")

    p_dm, ll_dm = fit_dmdhp_omori(times, mags, zones, T, m0)
    p_md, ll_md = fit_mdhp_omori(times, mags, T, m0)

    mu_d,c_d,p_d,Ksh,Kns,ash,ans = p_dm
    R     = Ksh/Kns if Kns > 1e-8 else float('inf')
    lrt   = 2*(ll_dm - ll_md)
    lrt_p = 1 - chi2.cdf(lrt, df=2)
    ks_p  = pit_ks_omori(p_dm, times, mags, zones, T, m0)

    aic_dm = -2*ll_dm + 2*7
    aic_md = -2*ll_md + 2*5

    sig = "***" if lrt_p<0.001 else ("**" if lrt_p<0.01 else
          ("*" if lrt_p<0.05 else ""))

    print(f"{seq_name:<24} {Nns:>4} {Ksh:>7.4f} {Kns:>7.4f} "
          f"{R:>7.3f} {lrt_p:>8.4f}{sig:3} {ks_p:>6.3f}")

    results.append({
        'sequence': seq_name, 'kernel': 'Omori-Utsu',
        'd1': d1, 'm0': m0, 'Nns': Nns,
        'Ksh': Ksh, 'Kns': Kns, 'R': R,
        'c': c_d, 'p_omori': p_d,
        'lrt_p': lrt_p, 'ks_p': ks_p,
        'AIC_DMDHP': aic_dm, 'AIC_MDHP': aic_md,
    })

print("-"*65)

# Save
out = os.path.join(OUTDIR, "omori_lrt_results.txt")
with open(out, 'w') as f:
    f.write("Omori-Utsu kernel DMDHP — LRT results\n")
    f.write("="*65 + "\n")
    f.write(f"{'Sequence':<24} {'Nns':>4} {'Ksh':>7} {'Kns':>7} "
            f"{'R':>7} {'LRT p':>9} {'KS p':>6} {'c':>7} {'p':>5}\n")
    f.write("-"*65 + "\n")
    for r in results:
        sig = ("***" if r['lrt_p']<0.001 else ("**" if r['lrt_p']<0.01
               else ("*" if r['lrt_p']<0.05 else "")))
        f.write(f"{r['sequence']:<24} {r['Nns']:>4} {r['Ksh']:>7.4f} "
                f"{r['Kns']:>7.4f} {r['R']:>7.3f} "
                f"{r['lrt_p']:>8.4f}{sig:3} {r['ks_p']:>6.3f} "
                f"{r['c']:>7.5f} {r['p_omori']:>5.3f}\n")
print(f"\nSaved: {out}")
print("Done.")
