#!/bin/sh
#SBATCH --job-name=gem5_simulation    
#SBATCH --output=slurm_logs/gem5_test_log.txt     
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu 

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM_PATH=$GEM5_ROOT/build/RISCV

# Task 1d: Register file sweep (64, 96, 128 entries)
# Fixed: pipeline_width=2, rob_size=128

# Original version
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_64-original default/cpu_benchmark1d.py 64 original
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_96-original default/cpu_benchmark1d.py 96 original
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_128-original default/cpu_benchmark1d.py 128 original

# Optimized version
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_64-optimized default/cpu_benchmark1d.py 64 optimized
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_96-optimized default/cpu_benchmark1d.py 96 optimized
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1d_128-optimized default/cpu_benchmark1d.py 128 optimized