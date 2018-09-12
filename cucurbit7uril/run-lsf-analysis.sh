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
#BSUB -R "rusage[mem=48] span[ptile=4]"
#
# job name (default = name of script file)
#BSUB -J "analyze"

yank analyze report --yaml sams.yaml --skipunbiasing --report --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/sams --serial=sams.pkl --fulltraj
yank analyze report --yaml repex.yaml --skipunbiasing --report --output /data/chodera/chodera/2018-08-05/yank-validation/cucurbit7uril/report/repex --serial=repex.pkl --fulltraj
