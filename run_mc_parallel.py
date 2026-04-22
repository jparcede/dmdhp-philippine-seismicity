"""
run_mc_parallel.py
===================
Runs the DMDHP-2zone Monte Carlo study in parallel using all available
CPU cores. No SLURM required — works on any multi-core machine.

Each replication runs as an independent process. Results are saved as
individual JSON files and aggregated at the end.

Usage:
    python3 run_mc_parallel.py                    # uses all cores
    python3 run_mc_parallel.py --n_cores 4        # limit to 4 cores
    python3 run_mc_parallel.py --n_rep 50         # quick test with 50 reps
    python3 run_mc_parallel.py --scenarios A B    # specific scenarios only

Authors: J.P. Arcede (Caraga State University)
Version: 1.0  |  April 2026
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from functools import partial

# Import the single-replication worker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmdhp_2zone_mc_single import run_replication, TRUE_PARAMS

OUTDIR   = "./mc_results"
N_REP    = 200
SEED     = 20260420


def worker(args):
    """Wrapper for multiprocessing — unpacks (rep, scenario, outdir, seed)."""
    rep, scenario, outdir, seed = args
    try:
        result = run_replication(rep, scenario, outdir, seed)
        status = result.get("status", "unknown")
        n      = result.get("n", 0)
        R_hat  = result.get("R_hat")
        lrt_p  = result.get("lrt_p")
        R_cov  = result.get("R_covered")
        print(f"  rep={rep:3d} scen={scenario} "
              f"n={n:4d} "
              f"R={R_hat:.2f if R_hat and R_hat < 1000 else 'large':>8} "
              f"lrt_p={lrt_p:.3f if lrt_p else 'nan':>7} "
              f"covered={R_cov}",
              flush=True)
        return result
    except Exception as e:
        print(f"  rep={rep} scen={scenario} ERROR: {e}", flush=True)
        return {"rep": rep, "scenario": scenario, "status": "error",
                "error": str(e)}


def run_scenario(scenario, n_rep, n_cores, outdir, seed):
    scen_dir = os.path.join(outdir, f"scenario_{scenario}")
    os.makedirs(scen_dir, exist_ok=True)

    params = TRUE_PARAMS[scenario]
    print(f"\n{'='*65}")
    print(f"SCENARIO {scenario}: {params['label']}")
    print(f"  {params['notes']}")
    print(f"  n_rep={n_rep}  n_cores={n_cores}")
    print(f"  True R = {params['K_sh']/params['K_ns']:.2f}")
    print(f"{'='*65}")

    # Check for already completed replications
    done = set()
    for fname in os.listdir(scen_dir):
        if fname.endswith(".json"):
            try:
                rep_id = int(fname.split("_")[1])
                done.add(rep_id)
            except Exception:
                pass
    remaining = [r for r in range(1, n_rep + 1) if r not in done]

    if len(done) > 0:
        print(f"  Resuming: {len(done)} already done, "
              f"{len(remaining)} remaining")
    else:
        print(f"  Starting fresh: {n_rep} replications")

    if not remaining:
        print("  All replications already complete!")
        return

    # Build task list
    tasks = [(rep, scenario, scen_dir, seed) for rep in remaining]

    t0 = time.time()
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker, tasks)
    elapsed = time.time() - t0

    n_ok  = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n  Scenario {scenario} complete: "
          f"{n_ok}/{len(results)} successful "
          f"in {elapsed/60:.1f} minutes")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="DMDHP-2zone parallel MC study")
    parser.add_argument("--n_rep",     type=int, default=N_REP)
    parser.add_argument("--n_cores",   type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Number of parallel cores (default: all-1)")
    parser.add_argument("--scenarios", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--outdir",    type=str, default=OUTDIR)
    parser.add_argument("--seed",      type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"DMDHP-2ZONE MONTE CARLO STUDY — PARALLEL RUNNER")
    print(f"{'='*65}")
    print(f"n_rep    = {args.n_rep}")
    print(f"n_cores  = {args.n_cores}  (available: {mp.cpu_count()})")
    print(f"scenarios = {args.scenarios}")
    print(f"outdir   = {os.path.abspath(args.outdir)}")
    print(f"seed     = {args.seed}")

    t_total = time.time()

    for scenario in args.scenarios:
        run_scenario(scenario, args.n_rep, args.n_cores,
                     args.outdir, args.seed)

    print(f"\n{'='*65}")
    print(f"ALL SCENARIOS COMPLETE")
    print(f"Total wall-clock time: {(time.time()-t_total)/60:.1f} minutes")
    print(f"{'='*65}")
    print(f"\nNext step: python3 aggregate_mc_results.py")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.freeze_support()
    main()
