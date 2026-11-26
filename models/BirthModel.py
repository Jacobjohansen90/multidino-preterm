#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:46:47 2026

@author: jacob
"""

from torch import nn
import torch

class BirthModel(nn.Module):
    def __init__(self, model_parts, aux_method='append', device='cuda'):
        
        super().__init__()
        self.vit_model = model_parts['vit_model']        
        self.ehr_model = model_parts['ehr_model']

        self.img_data_transform = model_parts['img_data_transform']
        self.seg_transform = model_parts['segmentation_transform']
        self.ehr_transform = model_parts['ehr_transform']
        
        self.pregnancy_transformer = model_parts['pregnancy_transformer']

        self.heads = model_parts['heads']

        self.aux_method = aux_method
        self.device = device
        
        if self.aux_method == 'append':
            #Early fusion
            self.forward_ = self.forward_append
        
        else:
            raise RuntimeError(f'Unknown fusion type f"{self.aux_method}"')
            
    def forward_append(self, data):
        #Make embeddings
        embeddings = []
        if data['ehr_data'].shape[1] != 0:
            ehr_embedding = self.ehr_model(data['ehr_data'].to(self.device))        
            ehr_embedding = self.ehr_transform(ehr_embedding)
            ehr_embedding = ehr_embedding[data['pregnancy_indices'].to(self.device)] #Expand to image level dim
            embeddings.append(ehr_embedding)
        
        if data['img_data'].shape[1] != 0:        
            img_data_embedding = self.img_data_transform(data['img_data'].to(self.device))
            embeddings.append(img_data_embedding)
        
        if data['segmentations'].shape[1] != 0:
            seg_embedding = self.seg_transform(data['segmentations'].to(self.device))
            embeddings.append(seg_embedding)
        
        #Use embeddings
        if len(embeddings) > 0:
            embeddings = [torch.stack(embeddings, dim=1)]
            features = self.vit_model(data['imgs'].to(self.device), append_tokens=embeddings) 
        else:
            features = self.vit_model(data['imgs'].to(self.device)) 
        
        pregnancy_features = self.pregnancy_transformer(features, data['pregnancy_indices'].to(self.device))
        
        #Generate outputs
        outputs = {}
        features = {ID: features.tolist() for ID, features in zip(data['IDs'], pregnancy_features) if ID is not None}    
        
        for task, task_heads in self.heads.items():
            outputs[task] = {}
            for name, head in task_heads.items():
                outputs[task][name] = head(pregnancy_features)
                

        return outputs, features

    def forward(self, data):
        return self.forward_(data)             
    
    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def unfreeze_vit(self, model, n, cfg):
        total_blocks = len(model.blocks)
    
        #Always unfreeze final normalization layer
        for p in model.norm.parameters():
            p.requires_grad = True
    
        # Maximum number of blocks that are allowed to be trainable
        if cfg.training.top_n_blocks == -1:
            max_blocks = total_blocks
        else:
            max_blocks = cfg.top_n_blocks
    
        #Number of blocks to unfreeze
        if cfg.training.blocks_per_step == -1:
            n_blocks = max_blocks
        else:
            n_blocks = min(max_blocks, cfg.training.blocks_per_step * (n // cfg.training.every_n_epochs))
    
        if n_blocks > 0:
            for block in model.blocks[-n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
    
        # Unfreeze embeddings once the whole backbone is trainable
        if n_blocks == total_blocks:
            model.cls_token.requires_grad = True
            model.pos_embed.requires_grad = True
            model.register_tokens.requires_grad = True


    
