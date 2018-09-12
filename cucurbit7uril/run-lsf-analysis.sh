#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 12:00
#
# Set output file
#BSUB -o analyze.%J.log
#
# Specify node group
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -q cpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 9
#BSUB -R "rusage[mem=48] span[hosts=1]"
#
# job name (default = name of script file)
#BSUB -J "analyze"

mpirun -np 9 yank analyze --yaml sams.yaml --serial=sams.pkl
mpirun -np 9 yank analyze --yaml repex.yaml --serial=repex.pkl
mpirun -np 9 yank analyze report --yaml sams.yaml --report --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/sams --serial=sams.pkl
mpirun -np 9 yank analyze report --yaml repex.yaml --report --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/repex --serial=repex.pkl
