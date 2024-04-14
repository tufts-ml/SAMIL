from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)

def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    # color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, 127)
    return img

def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)

def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)

def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def fixmatch_augment_pool():
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img

@dataclass()
class TrainingArgs():
    patience=200
    batch_size=1
    num_workers=2
    dataset_name='echo' 
    training_seed=0
    development_size='DEV479' 
    train_epoch=10 
    start_epoch=0
    eval_every_Xepoch=1
    script="src.SAMIL.main"
    use_data_normalization=False
    augmentation='RandAug'
    sampling_strategy='first_frame'
    use_class_weights=True
    class_weights = [0.25, 0.25, 0.25]
    Pretrained='Whole'
    ViewRegularization_warmup_schedule_type='Linear'
    ViewRegularization_warmup_pos = 0.4
    optimizer_type='SGD'
    nesterov = "store_true"
    lr_schedule_type='CosineLR'
    lr_warmup_epochs = 0
    lr_cycle_epochs=5

    ema_decay = 0.999

    checkpoint_dir = "checkpoint"
    data_info_dir = "info"
    data_dir = "data"

    assert torch.cuda.is_available(), "Need GPU"
    device = torch.device("cuda")

    data_seed = 0
    train_dir = f"training"
    train_PatientStudy_list_path = f"{data_info_dir}/train_studies.csv"
    val_PatientStudy_list_path = f"{data_info_dir}/val_studies.csv"
    test_PatientStudy_list_path = f"{data_info_dir}/test_studies.csv"

    resume='last_checkpoint.pth.tar'
    resume_checkpoint_fullpath = f"{checkpoint_dir}/{resume}"

    lr=0.0005
    wd=0.0001
    lambda_ViewRegularization=20
    T=0.05

def get_cosine_schedule_with_warmup(
    optimizer,
    lr_warmup_epochs,
    lr_cycle_epochs,
    num_cycles=7./16.,
    last_epoch=-1
    ):
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
        if current_epoch%lr_cycle_epochs==0: 
            current_cycle_epoch=lr_cycle_epochs
        else:
            current_cycle_epoch = current_epoch%lr_cycle_epochs   
        no_progress = float(current_cycle_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch) 

def calculate_balanced_accuracy(
    prediction, true_target, return_type = 'only balanced_accuracy'
    ):
    confusion_matrix = sk_confusion_matrix(
        true_target, prediction.argmax(1)
        )
    n_class = confusion_matrix.shape[0]
    assert n_class == 3
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
    balanced_accuracy = np.mean(np.array(recalls))
    if return_type == 'all':
        return balanced_accuracy * 100, recalls
    elif return_type == 'only balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)











