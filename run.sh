#!/bin/sh
python3 paradigm_dl_proj_satya_2.py
cd mcvd
CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/paradigm-moving-objects.yml --data_path ../../dataset --exp paradigm-moving-objects-out --ni
python3 test_diffusion_hidden.py ../../dataset paradigm-moving-objects-out/logs/checkpoint_150000.pt ../results
cd ..
python3 run_segmentation.py paradigm_segmentation.pth results/predictions-15000-17000-val/