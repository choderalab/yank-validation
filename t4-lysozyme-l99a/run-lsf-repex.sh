#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 5:59
#
# Set output file
#BSUB -o T4-repex.%J.log
#
# Specify node group
#BSUB -m "ls-gpu lt-gpu lp-gpu lg-gpu"
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 4
#BSUB -gpu "num=4:j_exclusive=yes:mode=shared" -R "rusage[mem=48] span[ptile=4]"
#
# job name (default = name of script file)
#BSUB -J "T4-repex"

build_mpirun_configfile --hostfilepath hostfile.${LSB_JOBNAME} --configfilepath configfile.${LSB_JOBNAME} "yank script --yaml=${LSB_JOBNAME}.yaml --status"
mpiexec.hydra -f hostfile.${LSB_JOBNAME} -configfile configfile.${LSB_JOBNAME}
