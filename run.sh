#!/bin/bash

## Should run [. .setup.sh] at root directory
export ROOT_DIR=$PWD
echo "Root Dir: $ROOT_DIR"
export DATA_INFO_DIR="$ROOT_DIR/data_info/"
export DATA_DIR="$ROOT_DIR/view_and_diagnosis_labeled_set/"
export CHECKPOINT_DIR="$ROOT_DIR/model_checkpoints/"

bash runs/SAMIL/launch_experiment.sh run_here