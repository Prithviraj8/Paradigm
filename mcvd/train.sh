
export user="ak11089"

export DL_PROJ="../.."
# this is config, you can change batch size in this config file
export config="paradigm-moving-objects"

# enter dataset folder, should contain train, validation and hidden
export data="${DL_PROJ}/dataset"

# Set out put folder path 
export exp="${DL_PROJ}/paradigm-moving-objects-out"
export config_mod=""

CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
# change batch size in config file if using 2 gpus
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
