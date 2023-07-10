import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
from sklearn.metrics import confusion_matrix as sklearn_cm
import numpy as np
import os
import pickle

import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import confusion_matrix as sklearn_cm


class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""
    
    def __init__(self, patience=300, initial_count=0, delta=0):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            
        """
        
        self.patience = patience
        self.counter = initial_count
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        
    
    def __call__(self, val_acc):
        
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.counter = 0
            
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!counter: {}, score: {}, best_score: {}'.format(self.counter, score, self.best_score))
        
        return self.counter

            
def train_one_epoch(args, weights, train_loader, model, ema_model, view_model, optimizer, scheduler, epoch):
    
    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

    model.train()
    
    TotalLoss_this_epoch, LabeledCELoss_this_epoch, ViewRegularizationLoss_this_epoch, scaled_ViewRegularizationLoss_this_epoch = [], [], [], []
    
    train_iter = iter(train_loader)       
    n_steps_per_epoch = 360 #360 train studies, batch size 1
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
#     for batch_idx, (data, bag_label, view_relevance) in enumerate(tqdm(train_loader)):
    for batch_idx in range(n_steps_per_epoch):
        
        try:
            data, bag_label = train_iter.next()
        except:
            train_iter = iter(train_loader)
            data, bag_label = train_iter.next()

#         print('batch_idx: {}'.format(batch_idx))

#         print('type(data): {}, data.size: {}, require grad: {}'.format(type(data), data.size(), data.requires_grad))
#         print('type(bag_label): {}, bag_label: {}'.format(type(bag_label), bag_label))
#         print('type(view_relevance): {}, view_relevance: {}'.format(type(view_relevance), view_relevance))
        data, bag_label = data.to(args.device), bag_label.to(args.device)
                
        
        outputs, attentions = model(data)
        
        log_attentions = torch.log(attentions)

        
        with torch.no_grad():
            view_predictions = view_model(data.squeeze(0))
            softmax_view_predictions = F.softmax(view_predictions, dim=1)
            predicted_relevance = softmax_view_predictions[:, :2]
            predicted_relevance = torch.sum(predicted_relevance, dim=1)
            predicted_relative_relevance = F.softmax(predicted_relevance/args.T)
            predicted_relative_relevance = predicted_relative_relevance.unsqueeze(0) 
            
            
        #element shape in F.cross_entropy: prediction torch.size([batch_size, num_classes]) and true label torch.size([batch_size])
        if args.use_class_weights == 'True':
            LabeledCELoss = F.cross_entropy(outputs, bag_label, weights, reduction='mean')
        else:
            LabeledCELoss = F.cross_entropy(outputs, bag_label, reduction='mean')
            
        
#         parser.add_argument('--ViewRegularization_warmup_pos', default=0.4, type=float, help='position at which view regularization loss warmup ends') #following MixMatch and FixMatch repo

# parser.add_argument('--ViewRegularization_warmup_schedule_type', default='NoWarmup', choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str) 
        
        #ViewRegularization warmup schedule choice
        if args.ViewRegularization_warmup_schedule_type == 'NoWarmup':
            current_warmup = 1
        elif args.ViewRegularization_warmup_schedule_type == 'Linear':
            current_warmup = np.clip(epoch/(float(args.ViewRegularization_warmup_pos) * args.train_epoch), 0, 1)
        elif args.ViewRegularization_warmup_schedule_type == 'Sigmoid':
            current_warmup = math.exp(-5 * (1 - min(epoch/(float(args.ViewRegularization_warmup_pos) * args.train_epoch), 1))**2)
        else:
            raise NameError('Not supported ViewRegularization warmup schedule')
            
        
        
        ViewRegularizationLoss = F.kl_div(input=log_attentions, target=predicted_relative_relevance, log_target=False, reduction='batchmean')
        
        # backward pass
        total_loss = LabeledCELoss + args.lambda_ViewRegularization * ViewRegularizationLoss * current_warmup
        
        total_loss.backward()
        
        
        
        TotalLoss_this_epoch.append(total_loss.item())
        LabeledCELoss_this_epoch.append(LabeledCELoss.item())
        ViewRegularizationLoss_this_epoch.append(ViewRegularizationLoss.item())
        scaled_ViewRegularizationLoss_this_epoch.append(args.lambda_ViewRegularization * ViewRegularizationLoss.item() * current_warmup)

        # step
        optimizer.step()

        #update ema model
        ema_model.update(model)
        
        model.zero_grad()
    
    scheduler.step()
        
    return TotalLoss_this_epoch, LabeledCELoss_this_epoch, ViewRegularizationLoss_this_epoch, scaled_ViewRegularizationLoss_this_epoch

   

#regular eval_model
def eval_model(args, data_loader, raw_model, ema_model, epoch):
        
    raw_model.eval()
    ema_model.eval()

    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []

        
        for batch_idx, (data, bag_label) in enumerate(data_loader):

#             print('EVAL type(data): {}, data.size: {}, require grad: {}'.format(type(data), data.size(), data.requires_grad))
#             print('EVAL type(bag_label): {}, bag_label: {}'.format(type(bag_label), bag_label))

            data, bag_label = data.to(args.device), bag_label.to(args.device)
            
            raw_outputs, raw_attention_weights = raw_model(data)
            ema_outputs, ema_attention_weights = ema_model(data)
#             print('target is {}, raw_outputs is: {}, ema_outputs is {}'.format(bag_label, raw_outputs, ema_outputs))

            total_targets.append(bag_label.detach().cpu())        
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())


        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
#         print('RegularEval total_targets: {}'.format(total_targets))
#         print('RegularEval total_raw_outputs: {}'.format(total_raw_outputs))
#         print('RegularEval total_ema_outputs: {}'.format(total_ema_outputs))

        raw_Bacc = calculate_balanced_accuracy(total_raw_outputs, total_targets)
        ema_Bacc = calculate_balanced_accuracy(total_ema_outputs, total_targets)

#         print('raw Bacc this evaluation step: {}'.format(raw_Bacc), flush=True)
#         print('ema Bacc this evaluation step: {}'.format(ema_Bacc), flush=True)


    return raw_Bacc, ema_Bacc, total_targets, total_raw_outputs, total_ema_outputs



def calculate_balanced_accuracy(prediction, true_target, return_type = 'only balanced_accuracy'):
    
    confusion_matrix = sklearn_cm(true_target, prediction.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)

    assert n_class==3
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    

    if return_type == 'all':
#         return balanced_accuracy * 100, class0_recall * 100, class1_recall * 100, class2_recall * 100
        return balanced_accuracy * 100, recalls

    elif return_type == 'only balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')

        
 #shared helper fct across different algos
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
               