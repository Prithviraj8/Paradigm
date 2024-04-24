#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_002-2024sp-18

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=12:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --exclusive
#SBATCH --requeue

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_student.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv \
--bind /scratch/sc9960 \
--overlay /tmp/dataset_student.sqsh \
/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
/bin/bash -c "
# add your script here
cd Paradigm
python3 paradigm_dl_proj_satya_2.py