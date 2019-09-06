#!/bin/sh
#PBS -N Horovod
#PBS -l select=2:ngpus=1:ncpus=24:mpiprocs=1:mem=96gb
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -j oe
## change next line to correct project code:
#PBS -P Personal
cd $PBS_O_WORKDIR

module load anaconda
# Load container environment
module load singularity/latest

# Run python inside Tensorflow container image
# python2 based on Docker image tensorflow/tensorflow:1.11.0-gpu
# python3 based on Docker image tensorflow/tensorflow:1.11.0-gpu-py3

HOROVOD_TIMELINE=$PWD/timeline.json ; export HOROVOD_TIMELINE
mpirun \
 -x NCCL_DEBUG=INFO \
 -x HOROVOD_TIMELINE \
   singularity exec $SINGULARITY_IMAGES/tensorflow/tensorflow.1.11.0-gpu.simg \
     python Training_theta_C7.py


