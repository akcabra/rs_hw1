#!/bin/sh
#SBATCH --job-name=gem5_simulation    
#SBATCH --output=slurm_logs/gem5_test_log.txt     
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu 

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM_PATH=$GEM5_ROOT/build/RISCV


srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_2-original default/cpu_benchmark1c.py 2 original
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_4-original default/cpu_benchmark1c.py 4 original
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_8-original default/cpu_benchmark1c.py 8 original

# Optimized version
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_2-optimized default/cpu_benchmark1c.py 2 optimized
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_4-optimized default/cpu_benchmark1c.py 4 optimized
srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=sim_log_1c_8-optimized default/cpu_benchmark1c.py 8 optimized