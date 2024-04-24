#!/bin/bash

conda create --name dl-project
conda activate dl-project
conda install pytorch torchvision torchaudio -c pytorch
conda install pillow
conda install numpy
conda install matplotlib

