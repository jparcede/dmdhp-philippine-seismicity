"""
download_global_catalogs.py
============================
Downloads aftershock catalogs for global subduction interface sequences
to complement the Philippine sequences in the multi-sequence DMDHP paper.

Sequences selected for maximum:
  - Tectonic diversity (Pacific, South American, Indian Ocean subduction zones)
  - Catalog richness (large Mw, well-recorded by global networks)
  - Depth diversity (both shallow and intermediate aftershocks expected)
  - Temporal coverage (recent, well-recorded by modern networks)

Usage
-----
    python download_global_catalogs.py

Outputs (saved to ./global_catalogs/)
---------------------------------------
    {seq_id}_raw.csv
    {seq_id}_catalog.csv
    {seq_id}_summary.txt
    {seq_id}_map.png
    {seq_id}_depth_hist.png
    global_catalog_overview.txt

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

OUTDIR = "global_catalogs"
os.makedirs(OUTDIR, exist_ok=True)

D1 = 70.0
D2 = 300.0

ZONE_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#2E86C1"}
ZONE_LABELS = {0: f"Shallow (0-{D1:.0f} km)",
               1: f"Intermediate ({D1:.0f}-{D2:.0f} km)",
               2: f"Deep (>{D2:.0f} km)"}

# ── Global sequences ──────────────────────────────────────────────────────────
# Selected to maximize tectonic contrast and catalog richness
# All are confirmed subduction interface events with large aftershock sequences

SEQUENCES = [
    # ── Pacific — Japan ───────────────────────────────────────────────────────
    dict(
        seq_id      = "G01_Tohoku2011",
        name        = "2011 Tohoku, Japan",
        mw          = 9.1,
        lat         = 38.297,
        lon         = 142.373,
        depth_km    = 29.0,
        origin_utc  = "2011-03-11T05:46:00",
        end_utc     = "2011-06-11T00:00:00",
        radius_km   = 400.0,
        min_mag     = 4.5,
        mechanism   = "INT",
        trench      = "Japan Trench",
        region      = "NE Japan",
        notes       = "Largest modern interface rupture; expect diverse depth distribution",
    ),
    dict(
        seq_id      = "G02_Fukushima2021",
        name        = "2021 Fukushima, Japan",
        mw          = 7.1,
        lat         = 37.721,
        lon         = 141.775,
        depth_km    = 44.0,
        origin_utc  = "2021-02-13T14:07:00",
        end_utc     = "2021-05-13T00:00:00",
        radius_km   = 200.0,
        min_mag     = 3.5,
        mechanism   = "INT",
        trench      = "Japan Trench",
        region      = "NE Japan",
        notes       = "Moderate interface event; good depth diversity expected",
    ),
    # ── Pacific — Chile ───────────────────────────────────────────────────────
    dict(
        seq_id      = "G03_Maule2010",
        name        = "2010 Maule, Chile",
        mw          = 8.8,
        lat         = -35.909,
        lon         = -72.733,
        depth_km    = 22.9,
        origin_utc  = "2010-02-27T06:34:00",
        end_utc     = "2010-05-27T00:00:00",
        radius_km   = 400.0,
        min_mag     = 4.5,
        mechanism   = "INT",
        trench      = "Chile Trench",
        region      = "South-Central Chile",
        notes       = "Major Nazca-South America interface; large rich catalog",
    ),
    dict(
        seq_id      = "G04_Illapel2015",
        name        = "2015 Illapel, Chile",
        mw          = 8.3,
        lat         = -31.573,
        lon         = -71.674,
        depth_km    = 22.4,
        origin_utc  = "2015-09-16T22:54:00",
        end_utc     = "2015-12-16T00:00:00",
        radius_km   = 300.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        trench      = "Chile Trench",
        region      = "North-Central Chile",
        notes       = "Well-recorded; expect mix of shallow interface and intraslab",
    ),
    dict(
        seq_id      = "G05_Pedernales2016",
        name        = "2016 Pedernales, Ecuador",
        mw          = 7.8,
        lat         = 0.382,
        lon         = -79.922,
        depth_km    = 20.6,
        origin_utc  = "2016-04-16T23:58:00",
        end_utc     = "2016-07-16T00:00:00",
        radius_km   = 250.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        trench      = "Peru-Chile Trench",
        region      = "Ecuador",
        notes       = "Nazca-South America interface; well recorded by global networks",
    ),
    # ── Indian Ocean — Sumatra ────────────────────────────────────────────────
    dict(
        seq_id      = "G06_Sumatra2012",
        name        = "2012 North Sumatra",
        mw          = 8.6,
        lat         = 2.327,
        lon         = 93.063,
        depth_km    = 20.0,
        origin_utc  = "2012-04-11T08:38:00",
        end_utc     = "2012-07-11T00:00:00",
        radius_km   = 400.0,
        min_mag     = 4.5,
        mechanism   = "INT",
        trench      = "Sunda Trench",
        region      = "North Sumatra / Indian Ocean",
        notes       = "Strike-slip dominated but includes interface components; large catalog",
    ),
    # ── Pacific — New Zealand ─────────────────────────────────────────────────
    dict(
        seq_id      = "G07_Kaikoura2016",
        name        = "2016 Kaikoura, New Zealand",
        mw          = 7.8,
        lat         = -42.737,
        lon         = 173.054,
        depth_km    = 15.1,
        origin_utc  = "2016-11-13T11:02:00",
        end_utc     = "2017-02-13T00:00:00",
        radius_km   = 250.0,
        min_mag     = 3.5,
        mechanism   = "INT",
        trench      = "Hikurangi subduction zone",
        region      = "South Island, New Zealand",
        notes       = "Complex multi-fault rupture; GeoNet provides excellent local catalog",
    ),
    # ── Pacific — Alaska ──────────────────────────────────────────────────────
    dict(
        seq_id      = "G08_Kodiak2018",
        name        = "2018 Kodiak, Alaska",
        mw          = 7.9,
        lat         = 56.046,
        lon         = -149.073,
        depth_km    = 14.1,
        origin_utc  = "2018-01-23T09:31:00",
        end_utc     = "2018-04-23T00:00:00",
        radius_km   = 250.0,
        min_mag     = 3.5,
        mechanism   = "INT",
        trench      = "Alaska-Aleutian subduction zone",
        region      = "Gulf of Alaska",
        notes       = "Pacific-North America interface; well recorded by AK seismic network",
    ),
    dict(
        seq_id      = "G09_Alaska2021",
        name        = "2021 Alaska Peninsula",
        mw          = 8.2,
        lat         = 55.364,
        lon         = -157.888,
        depth_km    = 32.2,
        origin_utc  = "2021-07-29T06:15:00",
        end_utc     = "2021-10-29T00:00:00",
        radius_km   = 300.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        trench      = "Alaska-Aleutian subduction zone",
        region      = "Alaska Peninsula",
        notes       = "Largest Alaska earthquake since 1965; well recorded",
    ),
    # ── Pacific — Mexico ──────────────────────────────────────────────────────
    dict(
        seq_id      = "G10_Mexico2017",
        name        = "2017 Chiapas, Mexico",
        mw          = 8.2,
        lat         = 14.761,
        lon         = -94.103,
        depth_km    = 47.4,
        origin_utc  = "2017-09-08T04:49:00",
        end_utc     = "2017-12-08T00:00:00",
        radius_km   = 300.0,
        min_mag     = 4.0,
        mechanism   = "INT",
        trench      = "Middle America Trench",
        region      = "Chiapas, Mexico / Guatemala",
        notes       = "Cocos-North America interface; deeper mainshock may affect depth distribution",
    ),
]


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
        print(f"    Already exists, skipping: {raw_path}")
        return True
    print(f"    Downloading...")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url,
                headers={"User-Agent": "Arcede-DMDHP-Study/1.0 (carsu.edu.ph)"})
            with urllib.request.urlopen(req, timeout=120) as resp:
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
    df = df[df["time"] > t0].copy()
    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
    df = df[df["mag"] >= seq["min_mag"]].copy()
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    df["x_km"], df["y_km"] = latlon_to_km(
        df["latitude"].values, df["longitude"].values,
        seq["lat"], seq["lon"])
    df["zone"] = df["depth"].apply(assign_depth_zone)
    df = df.sort_values("t_days").reset_index(drop=True)
    print(f"    Clean aftershocks (m>={seq['min_mag']}): {len(df)}")
    return df


def write_summary(df, seq, path):
    n_by_zone = df["zone"].value_counts().sort_index()
    T = float(df["t_days"].max()) if len(df) > 0 else 0.0
    lines = ["=" * 65, f"SEQUENCE: {seq['name']}", "=" * 65]
    lines += [
        f"ID:           {seq['seq_id']}",
        f"Mainshock:    Mw {seq['mw']}  |  {seq['origin_utc']} UTC",
        f"Epicenter:    {seq['lat']}N  {seq['lon']}E  |  depth {seq['depth_km']} km",
        f"Trench:       {seq['trench']}",
        f"Region:       {seq['region']}",
        f"Min mag:      m0 = {seq['min_mag']}",
        f"",
        f"Total aftershocks: {len(df)}",
        f"Time span:         {T:.2f} days",
    ]
    if len(df) > 0:
        lines += [
            f"Magnitude range:   {df['mag'].min():.1f} - {df['mag'].max():.1f}",
            f"Depth range:       {df['depth'].min():.1f} - {df['depth'].max():.1f} km",
            "",
            "Events by depth zone:",
        ]
        for z, label in ZONE_LABELS.items():
            n = int(n_by_zone.get(z, 0))
            pct = 100.0 * n / max(len(df), 1)
            lines.append(f"  Zone {z} {label:<32s}: {n:5d}  ({pct:.1f}%)")
    lines.append("=" * 65)
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)


def plot_depth_hist(df, seq, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    if len(df) > 0:
        ax.hist(df["depth"], bins=50, color="#2C3E50", edgecolor="white", lw=0.3)
        ax.axvline(D1, color=ZONE_COLORS[1], lw=1.5, linestyle="--", label=f"d1={D1} km")
        ax.axvline(D2, color=ZONE_COLORS[2], lw=1.5, linestyle="--", label=f"d2={D2} km")
        ax.legend(fontsize=8)
    ax.set_xlabel("Focal depth (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"{seq['name']}\nDepth distribution")

    ax2 = axes[1]
    for z in [0, 1, 2]:
        sub = df[df["zone"] == z]
        if len(sub) > 0:
            ax2.scatter(sub["t_days"], sub["depth"], s=5, alpha=0.4,
                        color=ZONE_COLORS[z], label=ZONE_LABELS[z])
    ax2.set_xlabel("Days since mainshock")
    ax2.set_ylabel("Depth (km)")
    ax2.set_title("Depth vs time")
    ax2.invert_yaxis()
    if len(df) > 0:
        ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_overview(results, ph_results, path):
    lines = ["=" * 105,
             "COMBINED CATALOG OVERVIEW — PHILIPPINE + GLOBAL SEQUENCES",
             "For: Arcede et al. (in prep.) — DMDHP multi-sequence study",
             "=" * 105, ""]

    hdr = (f"{'ID':<28s}  {'Mw':>4}  {'Trench/Fault':<28s}  "
           f"{'N_total':>8}  {'N_sh':>6}  {'N_int':>6}  {'N_deep':>7}  "
           f"{'T_days':>7}  {'m0':>4}  {'Identifiable?'}")
    lines += ["--- PHILIPPINE SEQUENCES ---", hdr, "-" * len(hdr)]

    for r in ph_results:
        seq, df = r["seq"], r["df"]
        _add_row(lines, seq, df)

    lines += ["", "--- GLOBAL SEQUENCES ---", hdr, "-" * len(hdr)]
    for r in results:
        seq, df = r["seq"], r["df"]
        _add_row(lines, seq, df)

    lines += [
        "",
        "Identifiability thresholds:",
        "  2-zone OK:       N_sh >= 50 AND N_int >= 20",
        "  3-zone possible: additionally N_deep >= 10",
        "  1-zone only:     N_int < 20",
        "  INSUFFICIENT:    N_sh < 50",
        "=" * 105,
    ]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)


def _add_row(lines, seq, df):
    n = len(df)
    n_by_zone = df["zone"].value_counts().sort_index() if n > 0 else {}
    n_sh   = int(n_by_zone.get(0, 0))
    n_int  = int(n_by_zone.get(1, 0))
    n_deep = int(n_by_zone.get(2, 0))
    T = float(df["t_days"].max()) if n > 0 else 0.0
    ident = ("3-zone possible" if n_sh>=50 and n_int>=20 and n_deep>=10 else
             "2-zone OK"      if n_sh>=50 and n_int>=20 else
             "1-zone only"    if n_sh>=50 else "INSUFFICIENT")
    trench_short = seq.get("trench", seq.get("tectonic", ""))[:27]
    lines.append(
        f"{seq['seq_id']:<28s}  {seq['mw']:>4.1f}  {trench_short:<28s}  "
        f"{n:>8d}  {n_sh:>6d}  {n_int:>6d}  {n_deep:>7d}  "
        f"{T:>7.1f}  {seq['min_mag']:>4.1f}  {ident}"
    )


def process_sequence(seq):
    print(f"\n{'='*65}")
    print(f"  {seq['seq_id']}")
    print(f"  {seq['name']}  (Mw {seq['mw']})")
    print(f"  {seq['notes']}")
    print(f"{'='*65}")

    raw_path = os.path.join(OUTDIR, f"{seq['seq_id']}_raw.csv")
    cln_path = os.path.join(OUTDIR, f"{seq['seq_id']}_catalog.csv")
    sum_path = os.path.join(OUTDIR, f"{seq['seq_id']}_summary.txt")
    dep_path = os.path.join(OUTDIR, f"{seq['seq_id']}_depth_hist.png")

    url = build_usgs_url(seq)
    ok  = download_catalog(url, raw_path, seq["seq_id"])
    if not ok:
        print(f"  WARNING: Download failed — skipping.")
        return None

    df = preprocess(raw_path, seq)
    cols = ["t_days", "latitude", "longitude", "depth", "mag", "x_km", "y_km", "zone", "time"]
    df[cols].to_csv(cln_path, index=False)
    write_summary(df, seq, sum_path)
    plot_depth_hist(df, seq, dep_path)
    return {"seq": seq, "df": df}


def main():
    print("\n=== GLOBAL SUBDUCTION SEQUENCE CATALOG DOWNLOAD ===")
    print(f"Downloading {len(SEQUENCES)} global sequences from USGS ComCat...")
    print(f"Output directory: {os.path.abspath(OUTDIR)}/\n")

    results = []
    for seq in SEQUENCES:
        result = process_sequence(seq)
        if result is not None:
            results.append(result)
        time.sleep(3)  # polite delay between USGS requests

    # Load Philippine results for combined overview
    ph_sequences = [
        dict(seq_id="SEQ1_Hinatuan2023",      name="2023 Hinatuan",      mw=7.4, min_mag=4.0, trench="Philippine Trench"),
        dict(seq_id="SEQ2_DavaoOriental2025",  name="2025 Davao Oriental",mw=7.4, min_mag=4.0, trench="Philippine Trench"),
        dict(seq_id="SEQ3_DavaoDeSur2019",     name="2019 Davao del Sur", mw=6.9, min_mag=4.0, trench="Cotabato Trench"),
    ]
    ph_results = []
    for seq in ph_sequences:
        cln_path = os.path.join("ph_catalogs", f"{seq['seq_id']}_catalog.csv")
        if os.path.exists(cln_path):
            df = pd.read_csv(cln_path)
            ph_results.append({"seq": seq, "df": df})

    overview_path = os.path.join(OUTDIR, "global_catalog_overview.txt")
    write_overview(results, ph_results, overview_path)

    print(f"\nAll global catalogs saved to: {os.path.abspath(OUTDIR)}/")
    print(f"Sequences downloaded: {len(results)}/{len(SEQUENCES)}")
    print("\nNext: review global_catalog_overview.txt to select final sequence set.")


if __name__ == "__main__":
    main()
