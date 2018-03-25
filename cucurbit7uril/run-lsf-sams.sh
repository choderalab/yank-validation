#!/usr/bin/env bash
#Walltime
#BSUB -W 24:00
#
# Set Output file
#BSUB -o  host-guest-sams.%J.log
#
# Specify node group
#BSUB -m "ls-gpu lt-gpu"
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request
#BSUB -n 3 -R "rusage[mem=12]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "host-guest-sams"

echo "LSB_HOSTS: $LSB_HOSTS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
module add cuda/9.0
nvcc --version

build_mpirun_configfile "yank script --yaml=cucurbit7uril-sams.yaml"

blaunch -u hostfile$JOBID "module list"
blaunch -u hostfile$JOBID "nvidia-smi"

mpiexec.hydra -f hostfile -configfile configfile

blaunch -u hostfile$JOBID "nvidia-smi"
