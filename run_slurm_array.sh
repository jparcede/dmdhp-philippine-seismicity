#!/bin/bash
#SBATCH --job-name=dmdhp_mc
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --array=1-200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jparcede@carsu.edu.ph

WORKDIR="/home/glgonzales/Documents/arcede_dmdhp_paper"
VENV="/home/glgonzales/Documents/news_garch/venv"

cd "${WORKDIR}" || { echo "ERROR: Cannot cd to ${WORKDIR}"; exit 1; }
source "${VENV}/bin/activate"

mkdir -p slurm_logs
mkdir -p mc_results/scenario_A
mkdir -p mc_results/scenario_B
mkdir -p mc_results/scenario_C

REP=${SLURM_ARRAY_TASK_ID}

echo "Rep ${REP} start: $(date)"

python3 -u dmdhp_2zone_mc_single.py --rep ${REP} --scenario A \
    --outdir mc_results/scenario_A --seed 20260420

python3 -u dmdhp_2zone_mc_single.py --rep ${REP} --scenario B \
    --outdir mc_results/scenario_B --seed 20260420

python3 -u dmdhp_2zone_mc_single.py --rep ${REP} --scenario C \
    --outdir mc_results/scenario_C --seed 20260420

echo "Rep ${REP} done: $(date)"
