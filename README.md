# Google Colab
Here is a [link](https://colab.research.google.com/drive/1dohxqPx9orWh2uws8TK9KYgeYbOaOI2R) to our notebook. A @illinois.edu email
is required to access the notebook along with the Google Drive data. The notebook and additional code can also be found in the notebook folder in 
this repo.

# SAMIL
Code for the paper Detecting Heart Disease from Multi-View Ultrasound Images via Supervised Attention Multiple Instance Learning (MLHC'23)

# Setup
### Prepare datasets
The AS diagnosis task in this study uses the view_and_diagnosis_labeled_set from TMED2 
- Step1. Download TMED2, please visit https://TMED.cs.tufts.edu and follow the instruction.
- Step2. Put the labeled and unlabeled image for the same study under same folder

### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Environment
pytorch 1.11.0

# Running experiments
### Define the environment variable
```export ROOT_DIR="paths to this repo" ```
(e.g.,  '/ab/cd/SAMIL')

```export DATA_INFO_DIR="path to the data_info/ folder" ```
(e.g.,  '/ab/cd/SAMIL/data_info')

```export DATA_DIR="path to the actual data ```
(e.g.,  '/ab/cd/SAMIL/Echo_data/view_and_diagnosis_labeled_set/')

```export CHECKPOINT_DIR="path to where pretrained view classifier and MOCO pretrained weights```
(e.g.,  '/ab/cd/SAMIL/model_checkpoints/')

We releases the pretrained view classifiers and MOCO pretrained checkpoints [here](https://tufts.box.com/s/c5w8123j7h3dpls75jye1363uh8qv8us) 

### Example
Running SAMIL with study-level pretraining, go to [runs/SAMIL](runs/SAMIL)
``` bash launch_experiment.sh run_here ```


### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of Pytorch etc (see https://pytorch.org/docs/stable/notes/randomness.html for more detail).

## Citing this work
@article{huang2023detecting,
  title={Detecting Heart Disease from Multi-View Ultrasound Images via Supervised Attention Multiple Instance Learning},
  author={Huang, Zhe and Wessler, Benjamin S and Hughes, Michael C},
  journal={arXiv preprint arXiv:2306.00003},
  year={2023}
}
