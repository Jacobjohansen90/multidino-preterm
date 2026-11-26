#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:17:10 2026

@author: jacob
"""

import ultradino_finetune.models.dinov2.load as vit_load
import logging
from models.Transform import Transform, CNNTransform
from models.Predictor import FCPredictor
from models.BirthModel import BirthModel
from models.PregnancyAttention import PregnancyTransformer
import torch.nn as nn
import re
import torch

logger = logging.getLogger("model_loader")

def vit_from_conf(cfg):    
    if cfg.data.img_size[0] != cfg.data.img_size[1]:
        raise Exception(f"Models only support square images. You have {cfg.data.img_size}")
    img_size = cfg.data.img_size[0] 
    
    if 'weights_path' not in cfg.model.vit:
        model = vit_load.load_from_scratch('vitb16', img_size=img_size)
        logger.info("Loading vitb16 encoder from scratch")
        
    elif cfg.model.vit.weights_path.endswith('.pth'):
        match = re.search(r'vit\w{3}', cfg.model.vit.weights_path)
        model_type = match.group() if match else None

        if model_type is not None:
            model = vit_load.load_pretrained_ultradino(model_type, cfg.model.vit.weights_path, img_size=img_size)
            logger.info('Loading pretrained encoder from %s', cfg.model.vit.weights_path)
        else:
            raise Exception("Weight path must contain the model type [Vits16/Vitb16/Vitl16]\n"
                            "E.g. 'oversampled_vitb16.pth'")
        
    else:
        raise Exception("Weight path not a .pth file")
    return model


def ehr_from_conf(cfg, **kwargs):
    #Placeholder for EHR model
    #Remember to update freezeing in BirthModel to accomedate EHR model if implemented!
    class PassThrough(nn.Module):
        def __init__(self):
            super(PassThrough, self).__init__()
            self.embed_dim = len(cfg.data.ehr_data)
            
        def forward(self, x):
            return x
    model = PassThrough()
    
    return model


def set_dropout(model, p=0.1):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def model_from_conf(cfg, weights_path=None):
    """Create GA model from configuration"""
    
    device = cfg.device.type
    model_parts = nn.ModuleDict({})
    
    #Foundation models
    vit_model = vit_from_conf(cfg)
    ehr_model = ehr_from_conf(cfg)
    
    model_parts['vit_model'] = vit_model
    model_parts['ehr_model'] = ehr_model
    
    #Transforms
    img_data_transform = Transform(len(cfg.data.img_data), 
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transforms.vit_layer_dims)
    
    ehr_transform = Transform(ehr_model.embed_dim, 
                              vit_model.embed_dim,
                              layer_dims=cfg.model.transforms.ehr_layer_dims)
        
    segmentation_transform = CNNTransform(cfg.data.img_size, 
                                          vit_model.embed_dim,
                                          channel_dims=cfg.model.transforms.seg_channel_dims)
    
    model_parts['img_data_transform'] = img_data_transform
    model_parts['ehr_transform'] = ehr_transform
    model_parts['segmentation_transform'] = segmentation_transform

    #Self attention transformer
    pregnancy_transformer = PregnancyTransformer(vit_model.embed_dim, cfg.model.self_attention.num_heads, 
                                                 cfg.model.self_attention.num_layers, 
                                                 cfg.model.self_attention.dropout)
    
    model_parts['pregnancy_transformer'] = pregnancy_transformer

    #Predictor heads
    heads = nn.ModuleDict({})
    
    for task, task_cfgs in cfg.tasks.items():
        heads[task] = nn.ModuleDict({})
        for task_cfg in task_cfgs:
            var = task_cfg['variable']
            name = var if isinstance(var, str) else '_'.join(var)
    
            cutoffs = task_cfg.get('cutoffs')
    
            if cutoffs:
                for cutoff in cutoffs:
                    heads[task][f"{name}_{cutoff}"] = FCPredictor(vit_model.embed_dim,
                                                                  cfg.model.head.dropout,
                                                                  cfg.model.head.layer_dims)
            else:
                heads[task][name] = FCPredictor(vit_model.embed_dim,
                                                cfg.model.head.dropout,
                                                cfg.model.head.layer_dims)
    
    

              
    model_parts['heads'] = heads

    model = BirthModel(model_parts , aux_method=cfg.auxiliary.method,
                       device=device)

    if weights_path is not None:
        weights=torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    
    return model.to(device)


def update_freezing(model, epoch, cfg):
    if epoch == 0:
        model.freeze_model(model.vit_model)
        model.freeze_model(model.ehr_model)
    if epoch >= cfg.training.vit_frozen_until:
        n = epoch - cfg.training.vit_frozen_until
        model.unfreeze_vit(model.vit_model, n, cfg)

            
