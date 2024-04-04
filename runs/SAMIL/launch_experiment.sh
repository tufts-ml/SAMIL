#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export patience=200
export batch_size=1
export num_workers=8
export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='echo' 
export training_seed=0
export development_size='DEV479' 
export train_epoch=2000 
export start_epoch=0
export eval_every_Xepoch=1


export script="src.SAMIL.main"


#data paths



#changeble setting from here:
export use_data_normalization='False'
export augmentation='RandAug'
export sampling_strategy='first_frame'

export use_class_weights='True'


# export Pretrained='NoPretrain'
export Pretrained='Whole'
# export Pretrained='FeatureExtractor1'
export ViewRegularization_warmup_schedule_type='Linear'
export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch

export checkpoint_dir=$CHECKPOINT_DIR
export data_info_dir=$DATA_INFO_DIR
export data_dir=$DATA_DIR


for data_seed in 0 1 2
do
    
    export data_seed=$data_seed
    export train_dir="$ROOT_DIR/experiments/SAMIL/Pretrained-$Pretrained/data_seed$data_seed/training_seed$training_seed/$development_size/"
    export train_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/train_studies.csv"
    export val_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/val_studies.csv"
    export test_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/test_studies.csv"
 

for lr in 5e-5 8e-4
do
    export lr=$lr
    
for wd in 1e-4 1e-3
do 
    export wd=$wd

for lambda_ViewRegularization in 5 15 20
do
    export lambda_ViewRegularization=$lambda_ViewRegularization


for T in 0.03 0.05 0.1
do
    export T=$T
    echo "Creating Train Dir: $train_dir"
    mkdir -p $train_dir


    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < ./do_experiment.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash "$ROOT_DIR/runs/SAMIL/do_experiment.slurm"
    fi


done
done
done
done
done