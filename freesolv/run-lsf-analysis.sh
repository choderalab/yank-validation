#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 5:59
#
# Set output file
#BSUB -o analyze.%J.log
#
# Specify node group
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -q cpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 4
#BSUB -R "rusage[mem=48]"
#
# job name (default = name of script file)
#BSUB -J "analyze"

mpirun -np 4 yank analyze report --yaml freesolv-mini.yaml --output reports
mpirun -np 4 yank analyze --yaml freesolv-mini.yaml --serial=analysis.pkl
