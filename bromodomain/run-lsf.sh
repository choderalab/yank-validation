#!/bin/bash

#BSUB -W 120:00
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -q gpuqueue
#BSUB -n 4 -R "rusage[mem=2]"
#BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
#BSUB -J "bromodomain"

source activate validation
build_mpirun_configfile "yank script --yaml=bromodomain.yaml"
mpiexec.hydra -f hostfile -configfile configfile
