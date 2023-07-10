#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
export PYTHONPATH="${PYTHONPATH}:$ROOT_DIR"
python -m $script \
    --patience $patience\
    --use_data_normalization $use_data_normalization\
    --augmentation $augmentation\
    --sampling_strategy $sampling_strategy\
    --use_class_weights $use_class_weights\
    --Pretrained $Pretrained\
    --ViewRegularization_warmup_schedule_type $ViewRegularization_warmup_schedule_type\
    --optimizer_type $optimizer_type\
    --lr_schedule_type $lr_schedule_type\
    --lr_cycle_epochs $lr_cycle_epochs\
    --lr $lr\
    --wd $wd\
    --T $T\
    --lambda_ViewRegularization $lambda_ViewRegularization\
    --batch_size $batch_size\
    --num_workers $num_workers\
    --resume $resume\
    --dataset_name $dataset_name\
    --data_seed $data_seed\
    --training_seed $training_seed\
    --development_size $development_size\
    --train_dir $train_dir\
    --checkpoint_dir $checkpoint_dir\
    --data_info_dir $data_info_dir\
    --data_dir $data_dir\
    --train_epoch $train_epoch\
    --start_epoch $start_epoch\
    --eval_every_Xepoch $eval_every_Xepoch\
    --train_PatientStudy_list_path $train_PatientStudy_list_path\
    --val_PatientStudy_list_path $val_PatientStudy_list_path\
    --test_PatientStudy_list_path $test_PatientStudy_list_path\
    
    