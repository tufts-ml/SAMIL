import argparse
import logging
import math
import os
import random
import shutil
import time
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
from src.DSMIL.libml.Echo_data import EchoDataset

from src.DSMIL.libml.utils import save_pickle
from src.DSMIL.libml.utils import train_one_epoch, eval_model
from src.DSMIL.libml.utils import EarlyStopping

from src.DSMIL.libml.ema import ModelEMA
from src.DSMIL.libml.randaugment import RandAugmentMC


logger = logging.getLogger(__name__)


# Training settings
parser = argparse.ArgumentParser()

#experiment setting
parser.add_argument('--dataset_name', default='echo', type=str, choices=['echo'], help='dataset name')#没用
parser.add_argument('--data_seed', default=0, type=int, help='which predefined split of TMED2')
parser.add_argument('--development_size', default='DEV479', help='DEV479, DEV165, DEV56')
parser.add_argument('--training_seed', default=0, type=int, help='random seed for training procedure')
parser.add_argument('--sampling_strategy', default='first_frame', type=str, help="either first_frame or '10', '20' etc")

parser.add_argument('--train_epoch', default=7200, type=int, help='total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--eval_every_Xepoch', default=360, type=int)

parser.add_argument('--model', type=str, default='set_transformer', help='Choose b/w set_transformer and deepset')


parser.add_argument('--Pretrained', default='Whole', type=str, help='Whole, FeatureExtractor1, NoPretrain')
# parser.add_argument('--MIL_checkpoint_path', type=str, default='', help='checkpoint for the pretrained MIL model, either the whole or just feature extractor1')


#hyperparameters inherit from Echo_ClinicalManualScript_torch style
parser.add_argument('--resume', default='last_checkpoint.pth.tar', type=str,
                    help='name of the checkpoint (default: last_checkpoint.pth.tar)')

parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)')

parser.add_argument('--train_dir')
parser.add_argument('--data_info_dir')
parser.add_argument('--data_dir')
parser.add_argument('--checkpoint_dir')


#data paths
parser.add_argument('--train_PatientStudy_list_path', type=str)
parser.add_argument('--val_PatientStudy_list_path', type=str)
parser.add_argument('--test_PatientStudy_list_path', type=str)


parser.add_argument('--lr', default=0.0005,type=float, help='learning rate (default: 0.0005)')
parser.add_argument('--lr_warmup_epochs', default=0, type=float, help='warmup epoch for learning rate schedule') #following MixMatch and FixMatch repo
parser.add_argument('--lr_schedule_type', default='CosineLR', choices=['CosineLR', 'FixedLR'], type=str) 
parser.add_argument('--lr_cycle_epochs', default=50, type=int, help='epoch') 

parser.add_argument('--wd', default=10e-5, type=float, help='weight decay')
parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'Adam', 'AdamW'], type=str) 

#default hypers not to search for now
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')

parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--patience', default=200, type=int)
parser.add_argument('--early_stopping_warmup', default=200, type=int)


# #echo config
parser.add_argument('--use_class_weights', default='True', type=str,
                    help='if use_class_weights is True, set class weights to be tie to combo of development_size and data_seed') 

parser.add_argument('--class_weights', default='0.25,0.25,0.25', type=str,
                    help='the weights used for weighted cross entropy loss for the labeled set') #if use_class_weights is 'True', set class weights to be tie to combo of development_size and data_seed. The number is set according to notebook: SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/src_TMED2/build_datasets/calculate_class_weights.ipynb


parser.add_argument('--augmentation', default='standard', type=str,
                    help='either standar or RandAug')

parser.add_argument('--use_data_normalization', default='False', type=str,
                    help='whether to normalize using train set mean and std')


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
        
        
        
#checked
def save_checkpoint(state, checkpoint_dir, filename='last_checkpoint.pth.tar'):
    '''last_checkpoint.pth.tar or xxx_model_best.pth.tar'''
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
        
#checked
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
#learning rate schedule   
def get_cosine_schedule_with_warmup(optimizer,
                                    lr_warmup_epochs,
                                    lr_cycle_epochs, #total train epochs
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
#         no_progress = float(current_epoch - lr_warmup_epochs) / \
#             float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))

        #see if using restart
        ###############################################################
        if current_epoch%lr_cycle_epochs==0: 
            current_cycle_epoch=lr_cycle_epochs
        else:
            current_cycle_epoch = current_epoch%lr_cycle_epochs
        
        no_progress = float(current_cycle_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        #################################################################
        
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)     


def get_fixed_lr(optimizer,
                lr_warmup_epochs,
                lr_cycle_epochs, #total train iterations
                num_cycles=7./16.,
                last_epoch=-1):
    def _lr_lambda(current_epoch):
        
        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    




def create_model(args):

    
    if args.model == 'DSMIL':
        import src.DSMIL.libml.dsmil as mil
        i_classifier = mil.IClassifier()
        b_classifier = mil.BClassifier(args)
        model = mil.MILNet(i_classifier, b_classifier)
            
    else:
        raise NameError('Bug')
        
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    
    return model


def main(args, brief_summary):
    
    args.use_data_normalization = str2bool(args.use_data_normalization)
    
    TMED2SummaryTable = pd.read_csv(os.path.join(args.data_info_dir, 'TMED2SummaryTable.csv'))
    
    echo_mean = [0.059, 0.059, 0.059]
    echo_std = [0.138, 0.138, 0.138]
    
    if args.use_data_normalization:
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=echo_mean, std=echo_std)
        ])
        
        transform_weak = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
        
        if args.augmentation == 'standard':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
            
        elif args.augmentation == 'RandAug':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
            
        else:
            raise NameError('Not implemented augmentation')
            
            

    else:
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
        #         transforms.Normalize(mean=echo_mean, std=echo_std)
        ])
        
        transform_weak = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
#                     transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
        
        if args.augmentation == 'standard':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    transforms.ToTensor(),
            #         transforms.Normalize(mean=echo_mean, std=echo_std)
                ])

        elif args.augmentation == 'RandAug':
            transform_labeledtrain = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=112,
                                         padding=int(112*0.125),
                                         padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
#                     transforms.Normalize(mean=echo_mean, std=echo_std)
                ])
        else:
            raise NameError('Not implemented augmentation')
       

    
    train_PatientStudy_list = pd.read_csv(args.train_PatientStudy_list_path)
    train_PatientStudy_list = train_PatientStudy_list['study'].values
    
    val_PatientStudy_list = pd.read_csv(args.val_PatientStudy_list_path)
    val_PatientStudy_list = val_PatientStudy_list['study'].values

    test_PatientStudy_list = pd.read_csv(args.test_PatientStudy_list_path)
    test_PatientStudy_list = test_PatientStudy_list['study'].values


    train_dataset = EchoDataset(train_PatientStudy_list, TMED2SummaryTable, args.data_dir, sampling_strategy=args.sampling_strategy, training_seed=args.training_seed, transform_fn=transform_labeledtrain)
    
    trainmemory_dataset = EchoDataset(train_PatientStudy_list, TMED2SummaryTable, args.data_dir, sampling_strategy='first_frame', training_seed=args.training_seed, transform_fn=transform_eval)

    val_dataset = EchoDataset(val_PatientStudy_list, TMED2SummaryTable, args.data_dir, sampling_strategy='first_frame', training_seed=args.training_seed, transform_fn=transform_eval)
    
    test_dataset = EchoDataset(test_PatientStudy_list, TMED2SummaryTable, args.data_dir, sampling_strategy='first_frame', training_seed=args.training_seed, transform_fn=transform_eval)
    

    print('Created dataset')
    print("train: {}, trainmemory: {}, val: {}, test: {}".format(len(train_dataset), len(trainmemory_dataset), len(val_dataset), len(test_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainmemory_loader = DataLoader(trainmemory_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    weights = args.class_weights
    weights = [float(i) for i in weights.split(',')]
    weights = torch.Tensor(weights)
#     print('weights used is {}'.format(weights))
    weights = weights.to(args.device)
    
#     #load the view model, the output is unnormalized logits, need to use softmax on the output 
#     view_model = create_view_model(args)
#     view_model.to(args.device)
    
    #create model
    model = create_model(args)
    model.to(args.device)
          
    
    #optimizer_type choice
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer_type == 'SGD':
        
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        
    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(grouped_parameters, lr=args.lr)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    elif args.optimizer_type == 'AdamW':
        optimizer = optim.AdamW(grouped_parameters, lr=args.lr)
        
    else:
        raise NameError('Not supported optimizer setting')
          
    
    
    #lr_schedule_type choice
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    
    else:
        raise NameError('Not supported lr scheduler setting')
          
    
    #instantiate the ema_model object
    ema_model = ModelEMA(args, model, args.ema_decay)
    
    
    args.start_epoch = 0
    
    
    
    #val progression view: tracking as best val performance progress, the corresponding test performance.
    #regular val
    best_val_ema_Bacc = 0
    best_test_ema_Bacc_at_val = 0
    best_train_ema_Bacc_at_val = 0
    
    best_val_raw_Bacc = 0
    best_test_raw_Bacc_at_val = 0
    best_train_raw_Bacc_at_val = 0
    
    
    current_count=0 #for early stopping, when continue training
    #if continued from a checkpoint, overwrite the best_val_ema_Bacc, best_test_ema_Bacc_at_val, 
    #                                              best_val_raw_Bacc, best_test_raw_Bacc_at_val,
    #                                              start_epoch,
    #                                              model weights, ema model weights
    #                                              optimizer state dict
    #                                              scheduler state dict 
          
    if os.path.isfile(args.resume_checkpoint_fullpath):        
#         logger.info("==> Resuming from checkpoint..")
        print('Resuming from checkpoint: {}'.format(args.resume_checkpoint_fullpath))

        checkpoint = torch.load(args.resume_checkpoint_fullpath)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        current_count = checkpoint['current_count']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        #val progression view
        #regular val
        best_val_ema_Bacc = checkpoint['val_progression_view']['best_val_ema_Bacc']
        best_test_ema_Bacc_at_val = checkpoint['val_progression_view']['best_test_ema_Bacc_at_val']
        best_train_ema_Bacc_at_val = checkpoint['val_progression_view']['best_train_ema_Bacc_at_val']
        
        best_val_raw_Bacc = checkpoint['val_progression_view']['best_val_raw_Bacc']
        best_test_raw_Bacc_at_val = checkpoint['val_progression_view']['best_test_raw_Bacc_at_val']
        best_train_raw_Bacc_at_val = checkpoint['val_progression_view']['best_train_raw_Bacc_at_val']
        
        
    else:
        print('!!!!Does not have checkpoint yet!!!!')
        

            
    
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset_name}")
    logger.info(f"  Num Epochs = {args.train_epoch}")
    logger.info(f"  Total optimization steps = {args.train_epoch * len(train_dataset)}")
    
    train_loss_dict = dict()
    train_loss_dict['Totalloss'] = []
    train_loss_dict['LabeledCEloss'] = []
    train_loss_dict['ViewRegularizationLoss'] = []
        
    early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)
    early_stopping_warmup = args.early_stopping_warmup


    for epoch in tqdm(range(args.start_epoch, args.train_epoch)):
        val_predictions_save_dict = dict()
        
        test_predictions_save_dict = dict()
        train_predictions_save_dict = dict()

        LabeledCEloss_list = train_one_epoch(args, weights, train_loader, model, ema_model, optimizer, scheduler, epoch)
#         train_loss_dict['Totalloss'].extend(TotalLoss_list)
        train_loss_dict['LabeledCEloss'].extend(LabeledCEloss_list)
#         train_loss_dict['ViewRegularizationLoss'].extend(ViewRegularizationLoss_list)
#         save_pickle(os.path.join(args.experiment_dir, 'losses'), 'train_losses_dict.pkl', train_loss_dict)

        if epoch % args.eval_every_Xepoch == 0:
            val_raw_Bacc, val_ema_Bacc, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch)
            val_predictions_save_dict['raw_Bacc'] = val_raw_Bacc
            val_predictions_save_dict['ema_Bacc'] = val_ema_Bacc
            val_predictions_save_dict['true_labels'] = val_true_labels
            val_predictions_save_dict['raw_predictions'] = val_raw_predictions
            val_predictions_save_dict['ema_predictions'] = val_ema_predictions
            
#             save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)
            
            test_raw_Bacc, test_ema_Bacc, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch)
        
            test_predictions_save_dict['raw_Bacc'] = test_raw_Bacc
            test_predictions_save_dict['ema_Bacc'] = test_ema_Bacc
            test_predictions_save_dict['true_labels'] = test_true_labels
            test_predictions_save_dict['raw_predictions'] = test_raw_predictions
            test_predictions_save_dict['ema_predictions'] = test_ema_predictions

            
#             save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)

            train_raw_Bacc, train_ema_Bacc, train_true_labels, train_raw_predictions, train_ema_predictions = eval_model(args, trainmemory_loader, model, ema_model.ema, epoch)
            
            train_predictions_save_dict['raw_Bacc'] = train_raw_Bacc
            train_predictions_save_dict['ema_Bacc'] = train_ema_Bacc
            train_predictions_save_dict['true_labels'] = train_true_labels
            train_predictions_save_dict['raw_predictions'] = train_raw_predictions
            train_predictions_save_dict['ema_predictions'] = train_ema_predictions
        

            #val progression view
            #regular Val
            if val_raw_Bacc > best_val_raw_Bacc:

                best_val_raw_Bacc = val_raw_Bacc
                best_test_raw_Bacc_at_val = test_raw_Bacc
                best_train_raw_Bacc_at_val = train_raw_Bacc

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)
                
                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_raw_val'), 'train_predictions.pkl', train_predictions_save_dict)
                
                save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
                    'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
                    'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,                     
                    }, 
               
                }, args.experiment_dir, filename='val_progression_view/best_predictions_at_raw_val/best_model.pth.tar')
                
                

            if val_ema_Bacc > best_val_ema_Bacc:
                
                best_val_ema_Bacc = val_ema_Bacc
                best_test_ema_Bacc_at_val = test_ema_Bacc
                best_train_ema_Bacc_at_val = train_ema_Bacc

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)

                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)
                
                save_pickle(os.path.join(args.experiment_dir, 'val_progression_view', 'best_predictions_at_ema_val'), 'train_predictions.pkl', train_predictions_save_dict)
                
                save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
                    'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
                    'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,                     
                     }, 
                    
               
                }, args.experiment_dir, filename='val_progression_view/best_predictions_at_ema_val/best_model.pth.tar')
                
                
                
            #val progression view
            logger.info('val progression view:')
            #regular val
            logger.info('At RAW Best val, validation/test/train %.2f %.2f %.2f' % (best_val_raw_Bacc, best_test_raw_Bacc_at_val, best_train_raw_Bacc_at_val))
            logger.info('At EMA Best val, validation/test/train %.2f %.2f %.2f' % (best_val_ema_Bacc, best_test_ema_Bacc_at_val, best_train_ema_Bacc_at_val))
            
            
            
            
            #only record the train loss, val_raw_Bacc, val_ema_Bacc, test_raw_Bacc, test_ema_Bacc every eval_every_Xepoch.
            args.writer.add_scalar('train/1.train_raw_Bacc', train_raw_Bacc, epoch)
            args.writer.add_scalar('train/1.train_ema_Bacc', train_ema_Bacc, epoch)
#             args.writer.add_scalar('train/1.Totalloss', np.mean(TotalLoss_list), epoch)
            args.writer.add_scalar('train/1.LabeledCEloss', np.mean(LabeledCEloss_list), epoch)
#             args.writer.add_scalar('train/1.ViewRegularizationLoss', np.mean(ViewRegularizationLoss_list), epoch)
#             args.writer.add_scalar('train/1.scaled_ViewRegularizationLoss', np.mean(scaled_ViewRegularizationLoss_list), epoch)

            #regular val
            args.writer.add_scalar('val/1.val_raw_Bacc', val_raw_Bacc, epoch)
            args.writer.add_scalar('val/2.val_ema_Bacc', val_ema_Bacc, epoch)
            
            
            args.writer.add_scalar('test/1.test_raw_Bacc', test_raw_Bacc, epoch)
            args.writer.add_scalar('test/2.test_ema_Bacc', test_ema_Bacc, epoch)
            
            
            #val progression view
            #regular val
            brief_summary['val_progression_view']['best_val_ema_Bacc'] = best_val_ema_Bacc
            brief_summary['val_progression_view']['best_val_raw_Bacc'] = best_val_raw_Bacc
            brief_summary['val_progression_view']['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
            brief_summary['val_progression_view']['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val
            brief_summary['val_progression_view']['best_train_ema_Bacc_at_val'] = best_train_ema_Bacc_at_val 
            brief_summary['val_progression_view']['best_train_raw_Bacc_at_val'] = best_train_raw_Bacc_at_val
            
            
            
            with open(os.path.join(args.experiment_dir, "brief_summary.json"), "w") as f:
                json.dump(brief_summary, f)
                
            
            #early stopping counting:
            if epoch > early_stopping_warmup:
                current_count = early_stopping(val_raw_Bacc)
            
            save_checkpoint(
                {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'current_count':current_count,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                    
                'val_progression_view': 
                    {'epoch': epoch+1,
                    #regular val
                    'best_val_ema_Bacc': best_val_ema_Bacc,
                    'best_val_raw_Bacc': best_val_raw_Bacc,
                    'best_test_ema_Bacc_at_val': best_test_ema_Bacc_at_val,
                    'best_test_raw_Bacc_at_val': best_test_raw_Bacc_at_val,
                    'best_train_ema_Bacc_at_val': best_train_ema_Bacc_at_val,
                    'best_train_raw_Bacc_at_val': best_train_raw_Bacc_at_val,                     
                    }, 
                    
               
                }, args.experiment_dir, filename='last_checkpoint.pth.tar')
        
            
            if early_stopping.early_stop:
                break

            
    #val progression view
    #regular val
    brief_summary['val_progression_view']['best_val_ema_Bacc'] = best_val_ema_Bacc
    brief_summary['val_progression_view']['best_val_raw_Bacc'] = best_val_raw_Bacc
    brief_summary['val_progression_view']['best_test_ema_Bacc_at_val'] = best_test_ema_Bacc_at_val 
    brief_summary['val_progression_view']['best_test_raw_Bacc_at_val'] = best_test_raw_Bacc_at_val
    brief_summary['val_progression_view']['best_train_ema_Bacc_at_val'] = best_train_ema_Bacc_at_val 
    brief_summary['val_progression_view']['best_train_raw_Bacc_at_val'] = best_train_raw_Bacc_at_val

    
   
    args.writer.close()

    with open(os.path.join(args.experiment_dir, "brief_summary.json"), "w") as f:
        json.dump(brief_summary, f)


        
    
    
    
            
if __name__=='__main__':
    
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError('Not Using GPU?')
        
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

  
    logger.info(dict(args._get_kwargs()))

    if args.training_seed is not None:
        print('setting training seed{}'.format(args.training_seed), flush=True)
        set_seed(args.training_seed)
    
    
    if args.Pretrained == 'Whole':
        
        raise NameError('NOT VALID PRETRAINED MODEL')
        
    elif args.Pretrained == 'FeatureExtractor1':
        
        raise NameError('NOT VALID PRETRAINED MODEL')
            
        
    elif args.Pretrained == 'NoPretrain':
        args.MIL_checkpoint_path=''
    
    else:
        raise NameError('invalid pretrain option')
    
    
    
    ################################################Determining class weights################################################
    #if use_class_weights is 'True', set class weights to be tie to combo of development_size and data_seed. The number is set according to notebook: SSL_Contamination/realistic-ssl-evaluation-pytorch_RE/src_TMED2/build_datasets/calculate_class_weights.ipynb
    
    if args.use_class_weights == 'True':
        print('!!!!!!!!Using pre-calculated class weights!!!!!!!!')
        
        #indeed, every split should have the same class weight for diagnosis by our dataset construction
        if args.data_seed == 0 and args.development_size == 'DEV479':
            args.class_weights = '0.463,0.342,0.195'
        elif args.data_seed == 1 and args.development_size == 'DEV479':
            args.class_weights = '0.463,0.342,0.195'
        elif args.data_seed == 2 and args.development_size == 'DEV479':
            args.class_weights = '0.463,0.342,0.195'
        else:
            raise NameError('not valid class weights setting')
    
    else:
        args.class_weights = '1.0,1.0,1.0'
        print('?????????Not using pre-calculated class weights?????????')
        
        
        
    experiment_name = "Optimizer-{}_LrSchedule-{}_LrCycleEpochs-{}_lr-{}_wd-{}".format(args.optimizer_type, args.lr_schedule_type, args.lr_cycle_epochs, args.lr, args.wd)
    
    
    args.experiment_dir = os.path.join(args.train_dir, experiment_name)
    
    if args.resume != 'None':
        args.resume_checkpoint_fullpath = os.path.join(args.experiment_dir, args.resume)
        print('args.resume_checkpoint_fullpath: {}'.format(args.resume_checkpoint_fullpath))
    else:
        args.resume_checkpoint_fullpath = None
        
    
    
    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)
    
    
    #brief summary:
    brief_summary = {}
    brief_summary['val_progression_view'] = {}
    brief_summary['test_progression_view'] = {}
    
    brief_summary['dataset_name'] = args.dataset_name
    brief_summary['algorithm'] = 'Echo_MIL'
    brief_summary['hyperparameters'] = {
        'train_epoch': args.train_epoch,
        'optimizer': args.optimizer_type,
        'lr': args.lr,
        'wd': args.wd,
    }

    main(args, brief_summary)

    
