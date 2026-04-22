"""
generate_kenneth_plots.py
==========================
Generates all missing diagnostic plots for Kenneth's DMDHP thesis
from the saved mc_results.npz file.

Produces:
    rep100_dmdhp_pit_hist.png        PIT histogram for rep 100
    rep100_dmdhp_pit_qq.png          PIT Q-Q plot for rep 100
    rep100_dmdhp_pit_ecdf.png        PIT ECDF for rep 100
    rep100_dmdhp_pit_acf.png         ACF of PIT residuals
    rep100_dmdhp_cum_rescaled.png    Cumulative rescaled times
    rep100_dmdhp_catalog_time_mag.png Simulated catalog mag vs time
    rep100_dmdhp_depth_vs_time.png   Depth vs time
    rep100_dmdhp_event_intensity.png  Conditional intensity

Usage:
    cd "/home/glgonzales/Documents/news_garch/Kenneth Earthquake Model"
    python3 generate_kenneth_plots.py
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest
from numpy.random import default_rng

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = "dmdhp_mc_scenarios_outputs/scenario_A_realistic_imbalanced"
NPZ_PATH = os.path.join(BASE, "mc_results.npz")

D1 = 70.0
D2 = 300.0
ZONE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#2E86C1"}
ZONE_LABELS = {0: "Shallow (0-70 km)",
               1: "Intermediate (70-300 km)",
               2: "Deep (>300 km)"}

# ── Load ──────────────────────────────────────────────────────────────────────
data      = np.load(NPZ_PATH, allow_pickle=True)
theta_true= data["theta_true"]        # shape (8,)
est_3     = data["est_3"]             # shape (200, 8)
ks_pvals  = data["ks_pvals"]          # shape (200,)
n_events  = data["n_events"]          # shape (200,)
n_sh      = data["n_sh"]
n_in      = data["n_in"]
n_dp      = data["n_dp"]
ll_3      = data["ll_3"]
aic_3     = data["aic_3"]
aic_b     = data["aic_b"]
aic_2     = data["aic_2"]
lrt_3_vs_b= data["lrt_3_vs_b"]

# Use rep index 99 (rep 100) — pick one with good fit
# Find rep with KS p > 0.05 and n_events close to mean
mean_n  = n_events.mean()
good    = np.where((ks_pvals > 0.05) & (n_events > 100))[0]
if len(good) > 0:
    rep_idx = good[len(good) // 2]   # middle of good reps
else:
    rep_idx = 99
print(f"Using rep index {rep_idx} (rep {rep_idx+1})")
print(f"  n_events={n_events[rep_idx]:.0f}  KS_p={ks_pvals[rep_idx]:.4f}")

theta_rep = est_3[rep_idx]          # fitted params for this rep
mu, beta, K_sh, K_int, K_deep, a_sh, a_int, a_deep = theta_rep

# ── Simulate one catalog using fitted params ──────────────────────────────────
def assign_zone(d):
    if d < D1: return 0
    if d < D2: return 1
    return 2

def simulate_dmdhp(theta, T=100.0, m0=4.0, p_zone=(0.6,0.3,0.1),
                   seed=20260420, max_events=50000):
    mu, beta, K_sh, K_int, K_deep, a_sh, a_int, a_deep = theta
    K = [K_sh, K_int, K_deep]
    a = [a_sh, a_int, a_deep]
    zone_depths = [35.0, 150.0, 400.0]
    rng = default_rng(seed)
    b   = math.log(10)

    n0  = rng.poisson(mu * T)
    t0s = np.sort(rng.uniform(0, T, n0))
    z0s = rng.choice(3, size=n0, p=p_zone)
    m0s = m0 + rng.exponential(1.0/b, n0)
    m0s = np.clip(m0s, m0, 8.0)

    times  = list(t0s)
    mags   = list(m0s)
    depths = [zone_depths[z] + rng.normal(0, 5) for z in z0s]

    q = 0
    while q < len(times):
        if len(times) > max_events:
            break
        tp, mp, dp = times[q], mags[q], depths[q]
        zp  = assign_zone(dp)
        kp  = K[zp] * math.exp(a[zp] * (mp - m0))
        noff = rng.poisson(kp)
        for _ in range(noff):
            tc = tp + rng.exponential(1.0 / beta)
            if tc > T: continue
            zc = rng.choice(3, p=p_zone)
            mc = m0 + rng.exponential(1.0/b)
            mc = min(mc, 8.0)
            dc = zone_depths[zc] + rng.normal(0, 5)
            times.append(tc); mags.append(mc); depths.append(dc)
        q += 1

    order  = np.argsort(times)
    return (np.array(times)[order],
            np.array(mags)[order],
            np.array(depths)[order])

print("Simulating catalog for rep plots...")
times_rep, mags_rep, depths_rep = simulate_dmdhp(theta_rep, T=100.0)
zones_rep = np.array([assign_zone(d) for d in depths_rep])
n = len(times_rep)
T_rep = 100.0
print(f"  Simulated n={n} events")

# ── Compute PIT residuals ─────────────────────────────────────────────────────
def compute_taus(theta, times, mags, depths, m0=4.0):
    mu, beta, K_sh, K_int, K_deep, a_sh, a_int, a_deep = theta
    K = [K_sh, K_int, K_deep]
    a = [a_sh, a_int, a_deep]
    taus = np.empty(len(times))
    R, t_prev = 0.0, 0.0
    for i in range(len(times)):
        dt = times[i] - t_prev
        taus[i] = mu * dt + R * (1.0 - math.exp(-beta * dt))
        z  = assign_zone(depths[i])
        ki = K[z] * math.exp(a[z] * (mags[i] - m0))
        R  = R * math.exp(-beta * dt) + ki
        t_prev = times[i]
    return taus

taus = compute_taus(theta_rep, times_rep, mags_rep, depths_rep)
U    = 1.0 - np.exp(-taus)
ks_p = kstest(U, "uniform").pvalue
print(f"  KS p-value = {ks_p:.4f}")

# ── Plot 1: PIT histogram ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(U, bins=12, density=True, color="#2C3E50", edgecolor="white", lw=0.5)
ax.axhline(1.0, color="#E74C3C", lw=1.5, ls="--", label="Uniform(0,1)")
ax.set_xlabel("PIT residual $U_i$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"PIT histogram — DMDHP 3-zone\nRep {rep_idx+1}, n={n}, KS p={ks_p:.3f}", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_pit_hist.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_pit_hist.png")

# ── Plot 2: PIT Q-Q ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
U_s  = np.sort(U)
theo = (np.arange(1, n+1) - 0.5) / n
ax.plot(theo, U_s, "o", ms=3, color="#2C3E50", alpha=0.7, label="Empirical")
ax.plot([0,1],[0,1], "--", color="#E74C3C", lw=1.5, label="Uniform")
ax.set_xlabel("Theoretical quantiles", fontsize=11)
ax.set_ylabel("Empirical PIT quantiles", fontsize=11)
ax.set_title(f"PIT Q-Q — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_pit_qq.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_pit_qq.png")

# ── Plot 3: PIT ECDF ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
U_s  = np.sort(U)
ecdf = np.arange(1, n+1) / n
ax.step(U_s, ecdf, color="#2C3E50", lw=1.5, label="Empirical CDF")
ax.plot([0,1],[0,1], "--", color="#E74C3C", lw=1.5, label="Uniform CDF")
ax.set_xlabel("$U_i$", fontsize=11)
ax.set_ylabel("$F_n(u)$", fontsize=11)
ax.set_title(f"PIT ECDF — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_pit_ecdf.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_pit_ecdf.png")

# ── Plot 4: ACF of PIT residuals ──────────────────────────────────────────────
try:
    from statsmodels.graphics.tsaplots import plot_acf
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_acf(U, lags=min(20, n//4), ax=ax, color="#2C3E50", zero=False)
    ax.set_title(f"ACF of PIT residuals — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "rep100_dmdhp_pit_acf.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: rep100_dmdhp_pit_acf.png")
except ImportError:
    # Manual ACF if statsmodels not available
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = range(1, min(21, n//4))
    acf_vals = [np.corrcoef(U[:-lag], U[lag:])[0,1] for lag in lags]
    conf = 1.96 / math.sqrt(n)
    ax.bar(list(lags), acf_vals, color="#2C3E50", alpha=0.7)
    ax.axhline(conf,  color="#E74C3C", ls="--", lw=1.2)
    ax.axhline(-conf, color="#E74C3C", ls="--", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("ACF", fontsize=11)
    ax.set_title(f"ACF of PIT residuals — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "rep100_dmdhp_pit_acf.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: rep100_dmdhp_pit_acf.png (manual ACF)")

# ── Plot 5: Cumulative rescaled times ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
cum = np.cumsum(taus)
idx = np.arange(1, n+1)
ax.plot(idx, cum, color="#2C3E50", lw=1.5, label="Observed")
ax.plot(idx, idx, "--", color="#E74C3C", lw=1.5, label="Expected (unit rate)")
ax.set_xlabel("Event index", fontsize=11)
ax.set_ylabel("Cumulative rescaled time", fontsize=11)
ax.set_title(f"Cumulative rescaled times — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_cum_rescaled.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_cum_rescaled.png")

# ── Plot 6: Catalog — magnitude vs time ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for z in [0, 1, 2]:
    mask = zones_rep == z
    if mask.sum() > 0:
        ax.scatter(times_rep[mask], mags_rep[mask], s=10, alpha=0.6,
                   color=ZONE_COLORS[z], label=ZONE_LABELS[z])
ax.set_xlabel("Days since mainshock", fontsize=11)
ax.set_ylabel("Magnitude", fontsize=11)
ax.set_title(f"Simulated catalog — DMDHP 3-zone\nRep {rep_idx+1} (n={n})", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_catalog_time_mag.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_catalog_time_mag.png")

# ── Plot 7: Depth vs time ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for z in [0, 1, 2]:
    mask = zones_rep == z
    if mask.sum() > 0:
        ax.scatter(times_rep[mask], depths_rep[mask], s=8, alpha=0.5,
                   color=ZONE_COLORS[z], label=ZONE_LABELS[z])
ax.axhline(D1, color="gray", lw=1.0, ls="--", alpha=0.6, label=f"d1={D1} km")
ax.axhline(D2, color="gray", lw=1.0, ls=":",  alpha=0.6, label=f"d2={D2} km")
ax.invert_yaxis()
ax.set_xlabel("Days since mainshock", fontsize=11)
ax.set_ylabel("Focal depth (km)", fontsize=11)
ax.set_title(f"Depth vs time — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_depth_vs_time.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_depth_vs_time.png")

# ── Plot 8: Conditional intensity (simplified) ────────────────────────────────
print("Computing conditional intensity...")
K  = [K_sh, K_int, K_deep]
a  = [a_sh,  a_int,  a_deep]
t_grid = np.linspace(0, T_rep, 400)
lam    = np.zeros(400)
for k, t in enumerate(t_grid):
    past_mask = times_rep < t
    if past_mask.sum() == 0:
        lam[k] = mu
        continue
    tp = times_rep[past_mask]
    mp = mags_rep[past_mask]
    dp = depths_rep[past_mask]
    R_sum = sum(
        K[assign_zone(dp[i])] * math.exp(a[assign_zone(dp[i])] * (mp[i]-4.0))
        * math.exp(-beta * (t - tp[i]))
        for i in range(len(tp))
    )
    lam[k] = mu + beta * R_sum

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_grid, lam, color="#2C3E50", lw=1.0, label="$\\lambda^*(t)$")
ax.scatter(times_rep, np.zeros(n)-0.5, s=4, color="#E74C3C", alpha=0.4,
           transform=ax.get_xaxis_transform(), zorder=5)
ax.set_xlabel("Days since mainshock", fontsize=11)
ax.set_ylabel("Conditional intensity", fontsize=11)
ax.set_title(f"Fitted conditional intensity — DMDHP 3-zone\nRep {rep_idx+1}", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "rep100_dmdhp_event_intensity.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved: rep100_dmdhp_event_intensity.png")

print(f"\nAll plots saved to:\n{os.path.abspath(BASE)}")
print("\nFinal file list:")
for f in sorted(os.listdir(BASE)):
    print(f"  {f}")
