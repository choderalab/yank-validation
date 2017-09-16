#!/bin/bash

#BSUB -W 120:00
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -q gpushared
#BSUB -n 16 -R "rusage[ngpus_excl_p=1,mem=2]"
#BSUB -m ls-gpu
#BSUB -J "cyclodextrin-validation"

source activate validation
JOBID=0
build_mpirun_configfile --hostfilepath="hostfile$JOBID" --configfilepath="configfile$JOBID" "yank script --yaml=cyclodextrin.yaml"
mpiexec.hydra -f hostfile$JOBID -configfile configfile$JOBID
#yank script -y cyclodextrin.yaml --jobid=$JOBID --njobs=4
