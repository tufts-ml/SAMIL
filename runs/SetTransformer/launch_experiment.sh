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

export early_stopping_warmup=800
export patience=200
export batch_size=1
export num_workers=2
export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='echo' 
export data_seed=0
export training_seed=0
export development_size='DEV479' 
export train_epoch=2000 
export start_epoch=0
export eval_every_Xepoch=1

#always use default
# export lr_warmup_epochs=0
# export ViewRegularization_warmup_pos=0.4

export script="src.SetTransformer_And_DeepSet.main"



#changeble setting from here:
# export model='deepset'
export model='set_transformer'
export use_data_normalization='False'
export augmentation='RandAug'
# export augmentation='standard'
export sampling_strategy='first_frame'
export use_class_weights='True'


export Pretrained='NoPretrain'
export optimizer_type='SGD'
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch


export checkpoint_dir=$CHECKPOINT_DIR
export data_info_dir=$DATA_INFO_DIR
export data_dir=$DATA_DIR


for data_seed in 0 1 2
do
    
    export data_seed=$data_seed
    export train_dir="$ROOT_DIR/experiments/$model/Pretrained-$Pretrained/data_seed$data_seed/training_seed$training_seed/$development_size/"
    export train_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/train_studies.csv"
    export val_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/val_studies.csv"
    export test_PatientStudy_list_path="$data_info_dir/DataPartition/seed$data_seed/$development_size/FullyLabeledSet_studies/test_studies.csv"
 
 
for wd in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    export wd=$wd

for lr in 3e-4 5e-4 8e-4 1e-3 3e-3
do
    export lr=$lr

    mkdir -p $train_dir
   
    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < ./do_experiment.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash ./do_experiment.slurm
    fi


done
done
done