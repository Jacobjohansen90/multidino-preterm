#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""
import torch 
import torch.nn
import copy

class TaskLoss(torch.nn.Module):
    def __init__(self, loss_cfg, weight, weight_multiplier, cutoff=None, 
                 label_smoothing=None, label_smoothing_window=None, 
                 distance_weight=None, distance_window=None):
        super().__init__()
        self.original_weight = weight
        self.weight = weight
        self.weight_multiplier = weight_multiplier
        self.pos_weight_multiplier = None
        self.ls = label_smoothing
        self.lsw = label_smoothing_window
        self.d_weight = distance_weight
        self.d_window = distance_window
        self.cutoff = cutoff

        loss_map = {'bce': torch.nn.BCEWithLogitsLoss,
                    'l1': torch.nn.L1Loss,
                    'l2': torch.nn.MSELoss}
        
        func = loss_map[loss_cfg.func]

        if loss_cfg.func == 'bce':
            self.register_buffer('initial_pos_weight', torch.tensor(loss_cfg.pos_weight, dtype=torch.float32))
            self.loss_fn = func(reduction='none', pos_weight=self.initial_pos_weight.clone())
            self.pos_weight_multiplier = loss_cfg.pos_multiplier
        else:
            self.loss_fn = func(reduction='none')
        
        
    def forward(self, logits, labels, mask):
        if not mask.any():
            return logits.sum()*0
        else:
            if self.d_weight:
                dw = self.distance_weighting(labels)
            else:
                dw = torch.ones_like(labels)
    
            if self.ls:
                labels = self.label_smoothing(labels)
            else:
                if self.cutoff:
                    labels = (labels < self.cutoff).float()
                
            loss = (self.loss_fn(logits, labels)*dw)[mask].mean()*self.weight
            
            return loss


    def update_weight(self):
        self.weight = self.weight*self.weight_multiplier
        if self.pos_weight_multiplier:
            self.loss_fn.pos_weight.mul_(self.pos_weight_multiplier)

        
    def reset_weight(self):
        self.weight = self.original_weight
        if self.pos_weight_multiplier:
            self.loss_fn.pos_weight.copy_(self.initial_pos_weight)


    def label_smoothing(self, labels):
        mask = (labels >= self.cutoff - self.lsw) & (labels <= self.cutoff + self.lsw)
        soft_labels = torch.sigmoid((self.cutoff - labels) / self.ls)
        labels = (labels < self.cutoff).float()
        labels[mask] = soft_labels[mask]
        
        return labels


    def distance_weighting(self, labels):
        distance = torch.abs(labels - self.cutoff)
        weight = 1 + self.d_weight * (distance / (self.d_window)).clamp(max=1)
        return weight                
        
    
class ModelLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device.type
        self.create_losses(cfg) 
        self.task_loss = {key: {'loss': [], 'count': []} for key in self.loss_fns.keys()}
        self.to(self.device)
        
    def create_losses(self, cfg):
        self.loss_fns = torch.nn.ModuleDict()
        self.key_pairs = []
        for task_name, task_cfgs in cfg.tasks.items():
            self.loss_fns[task_name] = torch.nn.ModuleDict()
            for task_cfg in task_cfgs:
                if 'cutoffs' in task_cfg:
                    n = len(task_cfg.cutoffs)
                else:
                    n = 1

                weights = self.expand(task_cfg,'weights', n)
                weights_multi = self.expand(task_cfg, 'weights_multiplier', n)
                distance_weights = self.expand(task_cfg, 'distance_weight', n)
                distance_windows = self.expand(task_cfg, 'distance_window', n)
                label_smoothing_values = self.expand(task_cfg, 'label_smoothing_value', n)
                label_smoothing_windows = self.expand(task_cfg,'label_smoothing_window', n)   
                cutoffs = self.expand(task_cfg, 'cutoffs', n)
    
                                                    
                for i in range(n):
                    var = task_cfg.variable
                    var = var if isinstance(var, str) else '_'.join(var)
                    if cutoffs[i]:
                        var = var + f"_{cutoffs[i]}"
                    self.key_pairs.append([task_name, var])
                    self.loss_fns[task_name][var] = TaskLoss(task_cfg.loss, weights[i], weights_multi[i], cutoffs[i],
                                                             label_smoothing=label_smoothing_values[i],
                                                             label_smoothing_window=label_smoothing_windows[i],
                                                             distance_weight=distance_weights[i],
                                                             distance_window=distance_windows[i])


    def expand(self, cfg, var, n):
        value = cfg.get(var)

        if value is None:
            return [None]*n        
        elif isinstance(value, (int, float)):
            return [value]*n
        else:
            return value
        
        
    def reset(self):
        self.task_loss = {key: {'loss': [], 'count': []} for key in self.loss_fns.keys()}
    
    
    def get_loss(self):
        losses = copy.deepcopy(self.task_loss)
        return losses
    
    
    def reset_weights(self):
        for task in self.loss_fns.values():
            for loss_fn in task.values():
                loss_fn.reset_weight()
    
    
    def update_weights(self):
        for task in self.loss_fns.values():
            for loss_fn in task.values():
                loss_fn.update_weight()
            
            
    def forward(self, outputs, data):
        loss = 0
        for task, var in self.key_pairs:
            logits = outputs[task][var]['logits']
            labels = data['labels'][task][var].to(self.device)
            mask = data['masks'][task][var].to(self.device)
            mask = mask.bool()
            temp_loss = self.loss_fns[task][var](logits, labels, mask)
            loss = loss + temp_loss
            self.task_loss[task]['loss'].append(temp_loss.item())
            self.task_loss[task]['count'].append(mask.sum().item())
        return loss

    

    







    
