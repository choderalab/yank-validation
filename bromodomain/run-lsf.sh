#!/bin/bash

#BSUB -W 24:00
#BSUB -q gpushared
#BSUB -n 4 -R "rusage[ngpus_excl_p=1,mem=2]"
#BSUB -m ls-gpu
#BSUB -J "bromodomain"
#BSUB -eo %J.out

source activate validation
build_mpirun_configfile "yank script --yaml=bromodomain.yaml"
mpiexec.hydra -f hostfile -configfile configfile
