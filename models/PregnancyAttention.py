#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:19:42 2026

@author: jacob
"""
from torch import nn
import torch

class PregnancyTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=1, dropout=0.2):
        super().__init__()
        
        # Learnable pregnancy token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   norm_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers,
                                                 enable_nested_tensor=False)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features, pregnancy_indices):
        device = features.device
        #Count total number of pregnancies
        n_pregnancies = pregnancy_indices.max().item() + 1
            
        #Count images per pregnancy
        counts = torch.bincount(pregnancy_indices, minlength=n_pregnancies)
        max_images = counts.max().item()
    
        #Pad pregnancies to max_images
        grouped = features.new_zeros(n_pregnancies,
                                     max_images,
                                     features.shape[-1])
    
        positions = torch.cat([torch.arange(count, device=device)
                               for count in counts.tolist()])
    
        grouped_indices = torch.repeat_interleave(torch.arange(n_pregnancies, device=device),
                                                  counts)
    
        grouped[grouped_indices, positions] = features
    
        # Create padding mask
        padding_mask = (torch.arange(max_images, device=device)[None, :] >= counts[:, None])
    
        # Add pregnancy token
        cls_token = self.cls_token.expand(n_pregnancies, -1, -1)
    
        grouped = torch.cat([cls_token, grouped], dim=1)
    
        # CLS token must never be masked
        cls_mask = torch.zeros(n_pregnancies, 1,
                               dtype=torch.bool,
                               device=device)
    
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
    
        # Transformer
        output = self.transformer(grouped, src_key_padding_mask=padding_mask)
    
        # Extract pregnancy embedding
        pregnancy_features = self.norm(output[:, 0])
    
        return pregnancy_features