"""
download_ph_catalogs.py
=======================
Downloads aftershock catalogs for multiple Philippine earthquake sequences
from USGS ComCat. Designed for the multi-sequence DMDHP publication study.

Usage
-----
    python download_ph_catalogs.py

Outputs (saved to ./ph_catalogs/)
-----------------------------------
    {seq_id}_raw.csv           Raw USGS download
    {seq_id}_catalog.csv       Clean catalog ready for DMDHP fitting
    {seq_id}_summary.txt       Descriptive statistics
    {seq_id}_map.png           Epicentral map
    {seq_id}_depth_hist.png    Depth distribution by zone
    ph_catalog_overview.txt    Cross-sequence comparison table

Philippine Sequences Included
------------------------------
    SEQ1  2023 Hinatuan (Surigao del Sur)    Mw 7.4  shallow interface thrust
    SEQ2  2025 Davao Oriental                Mw 7.4  shallow interface thrust
    SEQ3  2019 Davao del Sur                 Mw 6.9  shallow interface
    SEQ4  2019 Cotabato (Oct 29)             Mw 6.6  crustal strike-slip
    SEQ5  2019 Cotabato (Oct 16)             Mw 6.3  crustal strike-slip
    SEQ6  2017 Surigao del Norte             Mw 6.7  shallow strike-slip
    SEQ7  2017 Leyte                         Mw 6.5  crustal fault
    SEQ8  2013 Bohol                         Mw 7.2  reverse fault

Authors: J.P. Arcede (Caraga State University)
Version: 1.0  |  April 2026
"""

import os
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── output directory ──────────────────────────────────────────────────────────
OUTDIR = "ph_catalogs"
os.makedirs(OUTDIR, exist_ok=True)

# ── depth zone boundaries ─────────────────────────────────────────────────────
D1 = 70.0    # shallow / intermediate boundary (km)
D2 = 300.0   # intermediate / deep boundary (km)

ZONE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#2E86C1"}
ZONE_LABELS = {0: f"Shallow (0–{D1:.0f} km)",
               1: f"Intermediate ({D1:.0f}–{D2:.0f} km)",
               2: f"Deep (>{D2:.0f} km)"}

# ── Philippine sequences database ────────────────────────────────────────────
#
# Each entry defines one mainshock and its aftershock search parameters.
# Mechanism codes:
#   INT  = subduction interface thrust
#   INSL = intraslab (within subducting plate)
#   CSS  = crustal strike-slip
#   CRV  = crustal reverse / thrust
#   CNR  = crustal normal
#
SEQUENCES = [
    dict(
        seq_id      = "SEQ1_Hinatuan2023",
        name        = "2023 Hinatuan, Surigao del Sur",
        mw          = 7.4,
        lat         = 8.46,
        lon         = 127.70,
        depth_km    = 25.0,
        origin_utc  = "2023-12-02T14:37:00",
        end_utc     = "2024-02-29T00:00:00",
        radius_km   = 200.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        tectonic    = "Philippine Trench interface thrust",
        region      = "Caraga / Eastern Mindanao",
    ),
    dict(
        seq_id      = "SEQ2_DavaoOriental2025",
        name        = "2025 Davao Oriental",
        mw          = 7.4,
        lat         = 7.26,
        lon         = 126.76,
        depth_km    = 23.0,
        origin_utc  = "2025-10-10T03:12:00",
        end_utc     = "2026-01-10T00:00:00",
        radius_km   = 200.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        tectonic    = "Philippine Trench oblique reverse",
        region      = "Davao Oriental / Eastern Mindanao",
    ),
    dict(
        seq_id      = "SEQ3_DavaoDeSur2019",
        name        = "2019 Davao del Sur",
        mw          = 6.9,
        lat         = 6.22,
        lon         = 125.60,
        depth_km    = 60.0,
        origin_utc  = "2019-12-15T06:11:00",
        end_utc     = "2020-03-15T00:00:00",
        radius_km   = 150.0,
        min_mag     = 3.0,
        mechanism   = "INT",
        tectonic    = "Cotabato Trench interface",
        region      = "Davao del Sur / Southern Mindanao",
    ),
    dict(
        seq_id      = "SEQ4_Cotabato2019Oct29",
        name        = "2019 Cotabato (Oct 29, Mw 6.6)",
        mw          = 6.6,
        lat         = 6.62,
        lon         = 124.83,
        depth_km    = 8.0,
        origin_utc  = "2019-10-29T15:03:00",
        end_utc     = "2020-01-29T00:00:00",
        radius_km   = 120.0,
        min_mag     = 3.0,
        mechanism   = "CSS",
        tectonic    = "Cotabato Fault System strike-slip",
        region      = "North Cotabato / Central Mindanao",
    ),
    dict(
        seq_id      = "SEQ5_Cotabato2019Oct16",
        name        = "2019 Cotabato (Oct 16, Mw 6.3)",
        mw          = 6.3,
        lat         = 6.86,
        lon         = 124.92,
        depth_km    = 7.0,
        origin_utc  = "2019-10-16T03:37:00",
        end_utc     = "2020-01-16T00:00:00",
        radius_km   = 120.0,
        min_mag     = 3.0,
        mechanism   = "CSS",
        tectonic    = "Cotabato Fault System strike-slip",
        region      = "North Cotabato / Central Mindanao",
    ),
    dict(
        seq_id      = "SEQ6_Surigao2017",
        name        = "2017 Surigao del Norte",
        mw          = 6.7,
        lat         = 9.81,
        lon         = 126.12,
        depth_km    = 10.0,
        origin_utc  = "2017-02-10T05:03:00",
        end_utc     = "2017-05-10T00:00:00",
        radius_km   = 150.0,
        min_mag     = 3.0,
        mechanism   = "CSS",
        tectonic    = "Philippine Fault strike-slip",
        region      = "Surigao del Norte / NE Mindanao",
    ),
    dict(
        seq_id      = "SEQ7_Leyte2017",
        name        = "2017 Leyte",
        mw          = 6.5,
        lat         = 10.99,
        lon         = 124.84,
        depth_km    = 10.0,
        origin_utc  = "2017-07-06T04:03:00",
        end_utc     = "2017-10-06T00:00:00",
        radius_km   = 120.0,
        min_mag     = 3.0,
        mechanism   = "CSS",
        tectonic    = "Philippine Fault strike-slip",
        region      = "Leyte / Eastern Visayas",
    ),
    dict(
        seq_id      = "SEQ8_Bohol2013",
        name        = "2013 Bohol",
        mw          = 7.2,
        lat         = 9.86,
        lon         = 124.07,
        depth_km    = 20.0,
        origin_utc  = "2013-10-15T00:12:00",
        end_utc     = "2014-01-15T00:00:00",
        radius_km   = 150.0,
        min_mag     = 3.0,
        mechanism   = "CRV",
        tectonic    = "North Bohol Fault reverse",
        region      = "Bohol / Central Visayas",
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def build_usgs_url(seq):
    base = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = (
        f"?format=csv"
        f"&starttime={seq['origin_utc']}"
        f"&endtime={seq['end_utc']}"
        f"&latitude={seq['lat']}"
        f"&longitude={seq['lon']}"
        f"&maxradiuskm={seq['radius_km']}"
        f"&minmagnitude={seq['min_mag']}"
        f"&orderby=time-asc"
    )
    return base + params


def download_catalog(url, raw_path, seq_id, max_retries=3, wait=10):
    if os.path.exists(raw_path):
        print(f"    Raw catalog exists, skipping download: {raw_path}")
        return True
    print(f"    Downloading from USGS ComCat...")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url,
                headers={"User-Agent": f"Arcede-DMDHP-Study/1.0 (carsu.edu.ph)"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            with open(raw_path, "wb") as f:
                f.write(data)
            print(f"    Saved {len(data):,} bytes")
            return True
        except urllib.error.URLError as e:
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(wait)
    return False


def latlon_to_km(lat, lon, lat0, lon0):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat0))
    x = (np.asarray(lon) - lon0) * km_per_deg_lon
    y = (np.asarray(lat) - lat0) * km_per_deg_lat
    return x, y


def assign_depth_zone(depth):
    if depth < D1:
        return 0
    elif depth < D2:
        return 1
    return 2


def preprocess(raw_path, seq):
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"    Raw rows: {len(df)}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    t0 = pd.Timestamp(seq["origin_utc"], tz="UTC")

    # Keep only aftershocks
    df = df[df["time"] > t0].copy()
    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
    df = df[df["mag"] >= seq["min_mag"]].copy()

    # Derived columns
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    df["x_km"], df["y_km"] = latlon_to_km(
        df["latitude"].values, df["longitude"].values,
        seq["lat"], seq["lon"])
    df["zone"] = df["depth"].apply(assign_depth_zone)

    df = df.sort_values("t_days").reset_index(drop=True)
    print(f"    Clean aftershocks (m≥{seq['min_mag']}): {len(df)}")
    return df, t0


def write_summary(df, seq, t0, path):
    n_by_zone = df["zone"].value_counts().sort_index()
    T = float(df["t_days"].max()) if len(df) > 0 else 0.0

    lines = []
    lines.append("=" * 65)
    lines.append(f"SEQUENCE: {seq['name']}")
    lines.append("=" * 65)
    lines.append(f"ID:               {seq['seq_id']}")
    lines.append(f"Mainshock:        Mw {seq['mw']}  |  {seq['origin_utc']} UTC")
    lines.append(f"Epicenter:        {seq['lat']}N  {seq['lon']}E  |  depth {seq['depth_km']} km")
    lines.append(f"Mechanism:        {seq['mechanism']}  —  {seq['tectonic']}")
    lines.append(f"Region:           {seq['region']}")
    lines.append(f"Window:           {seq['origin_utc']} to {seq['end_utc']} UTC")
    lines.append(f"Search radius:    {seq['radius_km']} km")
    lines.append(f"Min magnitude:    m0 = {seq['min_mag']}")
    lines.append("")
    lines.append(f"Total aftershocks: {len(df)}")
    lines.append(f"Time span:         {T:.2f} days")
    if len(df) > 0:
        lines.append(f"Magnitude range:   {df['mag'].min():.1f} – {df['mag'].max():.1f}")
        lines.append(f"Depth range:       {df['depth'].min():.1f} – {df['depth'].max():.1f} km")
    lines.append("")
    lines.append("Events by depth zone:")
    for z, label in ZONE_LABELS.items():
        n = int(n_by_zone.get(z, 0))
        pct = 100.0 * n / max(len(df), 1)
        lines.append(f"  Zone {z} {label:<35s}: {n:5d}  ({pct:.1f}%)")
    lines.append("")
    lines.append("Magnitude distribution:")
    for lo in [3.0, 4.0, 5.0, 6.0]:
        hi = lo + 1.0
        n = int(((df["mag"] >= lo) & (df["mag"] < hi)).sum()) if len(df) > 0 else 0
        lines.append(f"  {lo:.0f} ≤ M < {hi:.0f}: {n}")
    n7 = int((df["mag"] >= 7.0).sum()) if len(df) > 0 else 0
    lines.append(f"  M ≥ 7.0        : {n7}")
    lines.append("=" * 65)

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)


def plot_map(df, seq, path):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for z in [0, 1, 2]:
        sub = df[df["zone"] == z]
        if len(sub) > 0:
            ax.scatter(sub["x_km"], sub["y_km"], s=6, alpha=0.55,
                       color=ZONE_COLORS[z], label=ZONE_LABELS[z])
    ax.scatter(0, 0, s=250, marker="*", color="black", zorder=5,
               label=f"Mainshock Mw {seq['mw']}")
    ax.set_xlabel("x (km east of mainshock)")
    ax.set_ylabel("y (km north of mainshock)")
    ax.set_title(f"{seq['name']}\nEpicentral map (colour = depth zone)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_depth_hist(df, seq, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if len(df) > 0:
        max_d = max(df["depth"].max() + 10, D2 + 10)
        zone_edges = [0, D1, D2, max_d]
        for i in range(3):
            ax.axvspan(zone_edges[i], zone_edges[i+1], alpha=0.12,
                       color=ZONE_COLORS[i], label=ZONE_LABELS[i])
        ax.hist(df["depth"], bins=40, color="#2C3E50", edgecolor="white", lw=0.4)
        ax.axvline(D1, color=ZONE_COLORS[1], lw=1.2, linestyle="--")
        ax.axvline(D2, color=ZONE_COLORS[2], lw=1.2, linestyle="--")
    ax.set_xlabel("Focal depth (km)")
    ax.set_ylabel("Event count")
    ax.set_title("Depth distribution")
    ax.legend(fontsize=7)

    ax2 = axes[1]
    for z in [0, 1, 2]:
        sub = df[df["zone"] == z]
        if len(sub) > 0:
            ax2.scatter(sub["t_days"], sub["depth"], s=8, alpha=0.5,
                        color=ZONE_COLORS[z], label=ZONE_LABELS[z])
    ax2.set_xlabel("Days since mainshock")
    ax2.set_ylabel("Focal depth (km)")
    ax2.set_title("Depth vs time")
    ax2.invert_yaxis()
    if len(df) > 0:
        ax2.legend(fontsize=7)

    fig.suptitle(f"{seq['name']} — {seq['mechanism']} — {seq['tectonic']}",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_overview(results, path):
    lines = []
    lines.append("=" * 100)
    lines.append("PHILIPPINE EARTHQUAKE SEQUENCES — CATALOG OVERVIEW")
    lines.append("For: Arcede et al. (in prep.) — DMDHP multi-sequence study")
    lines.append("=" * 100)
    lines.append("")
    hdr = (f"{'ID':<28s}  {'Mw':>4}  {'Mech':>4}  {'N_total':>8}  "
           f"{'N_sh':>6}  {'N_int':>6}  {'N_deep':>7}  "
           f"{'T_days':>7}  {'m0':>4}  {'Identifiable?'}")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in results:
        seq, df = r["seq"], r["df"]
        n = len(df)
        n_by_zone = df["zone"].value_counts().sort_index() if n > 0 else {}
        n_sh   = int(n_by_zone.get(0, 0))
        n_int  = int(n_by_zone.get(1, 0))
        n_deep = int(n_by_zone.get(2, 0))
        T = float(df["t_days"].max()) if n > 0 else 0.0

        # Identifiability check: need at least 50 events in at least 2 zones
        zones_ok = sum([n_sh >= 50, n_int >= 20, n_deep >= 10])
        ident = "2-zone OK" if (n_sh >= 50 and n_int >= 20) else (
                "1-zone only" if n_sh >= 50 else "INSUFFICIENT")
        if n_deep >= 10 and n_int >= 20 and n_sh >= 50:
            ident = "3-zone possible"

        lines.append(
            f"{seq['seq_id']:<28s}  {seq['mw']:>4.1f}  {seq['mechanism']:>4s}  "
            f"{n:>8d}  {n_sh:>6d}  {n_int:>6d}  {n_deep:>7d}  "
            f"{T:>7.1f}  {seq['min_mag']:>4.1f}  {ident}"
        )

    lines.append("")
    lines.append("Zone boundaries: d1=70 km (shallow/intermediate), d2=300 km (intermediate/deep)")
    lines.append("Identifiability thresholds: 2-zone requires N_sh>=50 and N_int>=20")
    lines.append("                            3-zone additionally requires N_deep>=10")
    lines.append("")
    lines.append("Mechanism codes:")
    lines.append("  INT  = Subduction interface thrust")
    lines.append("  INSL = Intraslab (within subducting plate)")
    lines.append("  CSS  = Crustal strike-slip")
    lines.append("  CRV  = Crustal reverse/thrust")
    lines.append("  CNR  = Crustal normal")
    lines.append("=" * 100)

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)


# ── main ──────────────────────────────────────────────────────────────────────

def process_sequence(seq):
    print(f"\n{'='*65}")
    print(f"  {seq['seq_id']}")
    print(f"  {seq['name']}  (Mw {seq['mw']}, {seq['mechanism']})")
    print(f"{'='*65}")

    raw_path = os.path.join(OUTDIR, f"{seq['seq_id']}_raw.csv")
    cln_path = os.path.join(OUTDIR, f"{seq['seq_id']}_catalog.csv")
    sum_path = os.path.join(OUTDIR, f"{seq['seq_id']}_summary.txt")
    map_path = os.path.join(OUTDIR, f"{seq['seq_id']}_map.png")
    dep_path = os.path.join(OUTDIR, f"{seq['seq_id']}_depth_hist.png")

    url = build_usgs_url(seq)
    ok  = download_catalog(url, raw_path, seq["seq_id"])
    if not ok:
        print(f"  WARNING: Download failed for {seq['seq_id']} — skipping.")
        return None

    df, t0 = preprocess(raw_path, seq)

    cols = ["t_days", "latitude", "longitude", "depth", "mag",
            "x_km", "y_km", "zone", "time"]
    df[cols].to_csv(cln_path, index=False)
    print(f"    Saved clean catalog: {cln_path}")

    write_summary(df, seq, t0, sum_path)
    plot_map(df, seq, map_path)
    plot_depth_hist(df, seq, dep_path)

    return {"seq": seq, "df": df}


def main():
    print("\n=== PHILIPPINE EARTHQUAKE SEQUENCES — CATALOG DOWNLOAD ===")
    print(f"Downloading {len(SEQUENCES)} sequences from USGS ComCat...")
    print(f"Output directory: {os.path.abspath(OUTDIR)}/\n")

    results = []
    for seq in SEQUENCES:
        result = process_sequence(seq)
        if result is not None:
            results.append(result)
        time.sleep(2)  # be polite to USGS API

    # Write cross-sequence overview
    overview_path = os.path.join(OUTDIR, "ph_catalog_overview.txt")
    write_overview(results, overview_path)

    print(f"\nAll catalogs saved to: {os.path.abspath(OUTDIR)}/")
    print(f"Sequences downloaded: {len(results)}/{len(SEQUENCES)}")
    print("\nNext step: review ph_catalog_overview.txt to select sequences")
    print("for the DMDHP multi-sequence analysis.")


if __name__ == "__main__":
    main()
