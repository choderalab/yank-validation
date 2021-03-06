#!/bin/bash
#  Batch script for mpirun job on cbio cluster.
#  Adjust your script as needed for your clusters!
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q batch
#
# nodes: number of nodes
#   ppn: how many cores per node to use
#PBS -l nodes=1:ppn=1
#
# 8GB of memory
#PBS -l mem=2G
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N extract-trajectory


if [ -n "$PBS_O_WORKDIR" ]; then 
    cd $PBS_O_WORKDIR
fi

echo "Extracting trajectories..."
yank analyze extract-trajectory --netcdf="old_output/wrongstandstatecorr/experiments/t4ligandbenzene/complex.nc" --state=0 --trajectory="trajectories/benzene_traj_state0.pdb" --skip=4 --nosolvent
date
