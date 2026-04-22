"""
fix_fig2_pit.py
================
Fixes Fig2 by finding the actual Hinatuan catalog file and
regenerating the PIT diagnostics from real data.

Run from: ~/Documents/arcede_dmdhp_paper
Usage:    python3 fix_fig2_pit.py
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest

OUTDIR = "./paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Step 1: Find the actual catalog file ─────────────────────────────────────
print("Searching for Hinatuan catalog...")

search_dirs = [
    "ph_catalogs", "catalogs", "data", ".", "ph_results",
    os.path.expanduser("~/Documents/arcede_dmdhp_paper/ph_catalogs"),
]

cat_file = None
for d in search_dirs:
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        flow = f.lower()
        if ("hinatuan" in flow or "seq1" in flow or "7.4" in flow) \
           and f.endswith(".csv"):
            cat_file = os.path.join(d, f)
            print(f"  Found: {cat_file}")
            break
    if cat_file:
        break

# Also check pit_diagnostics for saved U arrays
pit_file = None
pit_dir = "ph_results/pit_diagnostics"
if os.path.exists(pit_dir):
    print(f"\nFiles in {pit_dir}:")
    for f in sorted(os.listdir(pit_dir)):
        print(f"  {f}")
        flow = f.lower()
        if ("hinatuan" in flow or "seq1" in flow or "seq_1" in flow) \
           and (f.endswith(".npy") or f.endswith(".txt") or f.endswith(".csv")):
            pit_file = os.path.join(pit_dir, f)

# ── Step 2: Load catalog ─────────────────────────────────────────────────────
U = None

if pit_file:
    print(f"\nLoading PIT residuals from: {pit_file}")
    if pit_file.endswith(".npy"):
        U = np.load(pit_file)
    else:
        U = np.loadtxt(pit_file)
    print(f"  Loaded {len(U)} PIT residuals")

elif cat_file:
    print(f"\nLoading catalog from: {cat_file}")
    import csv
    times, mags, depths = [], [], []

    with open(cat_file, newline="") as f:
        # Detect delimiter
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader  = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames
        print(f"  Columns: {headers}")

        # Map column names flexibly
        def col(names, headers):
            for n in names:
                for h in headers:
                    if n.lower() in h.lower():
                        return h
            return None

        t_col = col(["time_day","days","t_day","time"], headers)
        m_col = col(["mag","magnitude","mw","ml"], headers)
        d_col = col(["depth","dep"], headers)
        print(f"  Using: time={t_col}, mag={m_col}, depth={d_col}")

        for row in reader:
            try:
                times.append(float(row[t_col]))
                mags.append(float(row[m_col]))
                depths.append(float(row[d_col]))
            except (ValueError, TypeError):
                pass

    times  = np.array(times)
    mags   = np.array(mags)
    depths = np.array(depths)

    # Sort by time
    idx    = np.argsort(times)
    times  = times[idx]
    mags   = mags[idx]
    depths = depths[idx]
    T      = times.max()
    m0     = 4.0

    print(f"  {len(times)} events, T={T:.1f} days")
    print(f"  Shallow: {(depths < 70).sum()}, Non-shallow: {(depths >= 70).sum()}")

    # Fitted DMDHP parameters for Hinatuan
    mu,  beta = 1.1312, 4.0008
    Ksh, Kns  = 0.3053, 0.0368
    ash, ans  = 1.4658, 0.1569

    def zone(d): return 0 if d < 70 else 1
    K_arr = [Ksh, Kns]
    a_arr = [ash, ans]

    # Compute compensator increments (PIT residuals)
    taus   = np.empty(len(times))
    R_acc  = 0.0
    t_prev = 0.0

    for i in range(len(times)):
        dt       = times[i] - t_prev
        # Integral of lambda*(s) from t_{i-1} to t_i
        taus[i]  = mu * dt + R_acc * (1.0 - math.exp(-beta * dt))
        # Update running sum
        z        = zone(depths[i])
        ki       = K_arr[z] * math.exp(a_arr[z] * (mags[i] - m0))
        R_acc    = R_acc * math.exp(-beta * dt) + ki
        t_prev   = times[i]

    U = 1.0 - np.exp(-taus)
    print(f"  Computed {len(U)} PIT residuals")
    print(f"  U range: [{U.min():.4f}, {U.max():.4f}]")

else:
    print("\nERROR: Could not find catalog or PIT file.")
    print("Please check ph_catalogs/ directory and re-run.")
    print("\nListing current directory contents:")
    for item in sorted(os.listdir(".")):
        if os.path.isdir(item):
            print(f"  {item}/")
            for sub in sorted(os.listdir(item))[:8]:
                print(f"    {sub}")
        else:
            print(f"  {item}")
    sys.exit(1)

# ── Step 3: Generate Fig2 ─────────────────────────────────────────────────────
n    = len(U)
ks_stat, ks_p = kstest(U, "uniform")
print(f"\nKS test: stat={ks_stat:.4f}, p={ks_p:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
fig.suptitle("PIT Diagnostics — DMDHP 2-Zone\n"
             "2023 Mw 7.4 Hinatuan Sequence  "
             f"($N = {n}$ events, KS $p = {ks_p:.3f}$)",
             fontsize=13, fontweight="bold")

# (a) Histogram
ax = axes[0, 0]
ax.hist(U, bins=14, density=True, color="#2166AC",
        edgecolor="white", lw=0.6, alpha=0.82)
ax.axhline(1.0, color="#D6604D", lw=1.8, ls="--", label="Uniform(0,1)")
ax.set_xlabel("PIT residual $U_i$")
ax.set_ylabel("Density")
ax.set_title(f"(a) Histogram  (KS $p = {ks_p:.3f}$)")
ax.set_xlim(0, 1)
ax.legend()

# (b) Q-Q plot
ax = axes[0, 1]
U_s  = np.sort(U)
theo = (np.arange(1, n + 1) - 0.5) / n
ax.scatter(theo, U_s, s=6, color="#2166AC", alpha=0.5, zorder=3)
ax.plot([0, 1], [0, 1], "--", color="#D6604D", lw=1.8, label="1:1 line")
# 95% KS band
ks_band = 1.36 / math.sqrt(n)
ax.fill_between([0, 1],
                [max(0, i - ks_band) for i in [0, 1]],
                [min(1, i + ks_band) for i in [0, 1]],
                color="#D6604D", alpha=0.12, label="95% KS band")
ax.set_xlabel("Theoretical Uniform quantile")
ax.set_ylabel("Empirical PIT quantile")
ax.set_title("(b) Q-Q plot")
ax.legend(fontsize=8)

# (c) ECDF
ax = axes[1, 0]
ecdf = np.arange(1, n + 1) / n
ax.step(U_s, ecdf, color="#2166AC", lw=1.8, label="Empirical CDF", where="post")
ax.plot([0, 1], [0, 1], "--", color="#D6604D", lw=1.8, label="Uniform CDF")
ax.fill_between(U_s,
                np.maximum(ecdf - ks_band, 0),
                np.minimum(ecdf + ks_band, 1),
                color="#D6604D", alpha=0.10, label="95% KS band")
ax.set_xlabel("$u$")
ax.set_ylabel("$F_n(u)$")
ax.set_title("(c) Empirical CDF")
ax.legend(fontsize=8)

# (d) ACF
ax = axes[1, 1]
max_lag = min(25, n // 5)
lags    = np.arange(1, max_lag + 1)
acf_vals = np.array([np.corrcoef(U[:-lag], U[lag:])[0, 1] for lag in lags])
conf     = 1.96 / math.sqrt(n)
ax.bar(lags, acf_vals, color=np.where(np.abs(acf_vals) > conf, "#D6604D", "#2166AC"),
       alpha=0.75, width=0.7)
ax.axhline( conf, color="black", ls="--", lw=1.2, label=f"±1.96/$\\sqrt{{N}}$")
ax.axhline(-conf, color="black", ls="--", lw=1.2)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Lag $k$")
ax.set_ylabel("ACF$(U_i, U_{i+k})$")
ax.set_title("(d) ACF of PIT residuals\n(red = significant at 5%)")
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = f"{OUTDIR}/Fig2_hinatuan_pit.png"
plt.savefig(fig_path)
plt.close()
print(f"\nSaved: {fig_path}")
print("Fig2 is now based on REAL catalog data.")
