#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:33:56 2026

@author: jacob
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import albumentations as A
import polars as pl
from torch.utils.data import DataLoader, Sampler
import random


FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424
        
class PreTermDataset(Dataset):
    def __init__(self, df, cfg, train, max_images, ID='CPR_CHILD'):

        super().__init__()
        self.img_size = cfg.data.img_size
        self.ehr_vars = cfg.data.ehr_data
        self.img_data_vars = cfg.data.img_data
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424       
        self.train = train
        self.setup_transforms()
        self.ID_var = ID
        self.get_segs = 'segmentation' in cfg.tasks.keys() or cfg.data.segmentation
        self.remove = cfg.data.remove
        self.max_images = max_images
        self.tasks = cfg.tasks

        condition = (pl.col('pprom') |
                     (~pl.col('induced') & ~pl.col('c-section')) |
                     (pl.col('c-section') & (pl.col('c-section_during_birth') |
                                             pl.col('contractions_with_preterm_birth')) &
                      ~pl.col('induced')))

        self.df = df.with_columns(pl.when(condition).then(1000).otherwise(pl.col('GA')).alias('GA_valid'))
                            
        self.groups = (self.df.with_row_index("img_idx").group_by(ID, maintain_order=True).agg(pl.col("img_idx")))
        self.pregnancy_ids = self.groups[ID].to_list()
        self.pregnancy_indices = self.groups["img_idx"].to_list()
        self.image_counts = [len(indices) for indices in self.pregnancy_indices]
        
        
    def setup_transforms(self):
        if self.train:
            self.transforms = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.15),
                                         A.RandomGamma(gamma_limit=(80, 120), p=0.15),
                                         A.GaussNoise(std_range=(0.05, 0.2), p=0.15),
                                         #A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                                         A.HorizontalFlip(p=0.5),
                                         A.VerticalFlip(p=0.5),
                                         A.Resize(height=self.img_size[0], width=self.img_size[1]), #move this up
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])
        
        else:
            self.transforms = A.Compose([A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])        
        
    def __getitem__(self, idx):
        return self.getitem(idx)
        
    
    def __len__(self):
        return len(self.pregnancy_ids)

    def get_pregnancy_data(self, data):
        #Get treatment status
        treatment = bool(data.get('progesterone') or data.get('cerclage'))
        
        #Prepare EHR data
        ehr_data = []
        for key in self.ehr_vars:
            if isinstance(key, str):                
                if key.lower() == 'treatment':
                    ehr_data.append(treatment)
                else:
                    ehr_data.append(float(data.get(key) or 0.0))
            else:
                values = [float(data[k]) for k in key if data.get(k) is not None]
                ehr_data.append(sum(values) / len(values) if values else 0.0)

        ehr_data = torch.tensor(ehr_data)
                
        #Prepare labels & masks
        labels = {}
        masks = {}
        
        GA_weeks = data.get('GA')//7
        
        for task, task_cfgs in self.tasks.items():
            labels[task] = {}
            masks[task] = {}
            for task_cfg in task_cfgs:
                #Get variable value
                var_name = task_cfg.get('variable')
                if var_name == 'GA_weeks':
                    var = GA_weeks
                elif isinstance(var_name, str):
                    var = data.get(var_name)
                    var = float(var) if var is not None else 0.0
                else:
                    values = [float(data[v]) for v in var_name if data.get(v) is not None]
                    var = sum(values) / len(values) if values else 0.0
                    var_name = '_'.join(var_name)
                        
                var = torch.tensor([var])
                #Check for a condition on the task
                conditions = task_cfg.get('conditional') or []
                cond = 1
                for condition in conditions:
                    if data.get(condition[0]) != condition[1]:
                        cond = 0
                
                #Check if task has cutoffs and apply them
                cutoffs = task_cfg.get('cutoffs')
                if cutoffs:
                    for cutoff in cutoffs:
                        name = f"{var_name}_{cutoff}"
                        labels[task][name] = var
                        if not cond or (data.get('GA_valid')//7 < cutoff):
                            masks[task][name] = torch.tensor([0])
                        else:
                            masks[task][name] = torch.tensor([1])
                else:
                    labels[task][var_name] = var
                    masks[task][var_name] = torch.tensor([int(cond)])
                        
        #Get patient identifier
        ID = data.get(self.ID_var)
                    
        return {'ehr_data': ehr_data,
                'labels': labels,
                'masks': masks,
                'ID': ID,
                'treatment': treatment}


    def get_image_data(self, data):
        #Get segmentations
        if self.get_segs:
            segmentation = np.load(data.get('segmentation_path'))['seg_logits']
            segmentation = np.isin(segmentation, self.seg_labels)
            segmentation = torch.from_numpy(segmentation)
        else:
            segmentation = torch.tensor([], dtype=torch.bool)
 
        #Prepare Image
        img_path = data.get('no_ocr_preprocessed_file_path')
        # img = Image.open(img_path)
        img = np.zeros((224,224))
        img = np.asarray(img)

        h,w  = img.shape[:2]

        img = self.transforms(image=img)['image']
        imgs = img.unsqueeze(0)
        
        #Prepare image metadata
        img_data = []
        for key in self.img_data_vars:
            data_temp = data.get(key) or 0.0
            #Correct physical deltas    
            if key == 'physical_delta_x' and data_temp != 0:
                data_temp = (w/self.img_size[1]) * data_temp[0]
            elif key == 'physical_delta_y' and data_temp != 0:
                data_temp = (h/self.img_size[0]) * data_temp[0]
            
            img_data.append(data_temp)
        
        img_data = torch.tensor(img_data)
        


        return {'imgs': imgs, 
                'img_path': img_path,
                'img_data': img_data, 
                'segmentation': segmentation}

    def getitem(self, idx):
        indices = self.pregnancy_indices[idx]
        if (len(indices) > self.max_images) and self.train:
            indices = random.sample(indices, self.max_images)
        
        #Pregnancy information (same across images)
        data = self.df.row(indices[0], named=True)
        pregnancy_data = self.get_pregnancy_data(data)
        
        #Images
        image_data = [self.get_image_data(self.df.row(i, named=True)) for i in indices]
        
        return {**pregnancy_data,
                'images': image_data}

class CollateFn:
    def __init__(self, target_size=None):
        self.target_size = target_size
    
    def __call__(self, batch):
        imgs = []
        img_paths = []
        img_data = []
        segmentations = []
        pregnancy_indices = []
        
        for pregnancy_idx, sample in enumerate(batch):
            for img in sample['images']:
                imgs.append(img['imgs'])
                img_paths.append(img['img_path'])
                img_data.append(img['img_data'])
                segmentations.append(img['segmentation'])
                pregnancy_indices.append(pregnancy_idx)
        
        imgs = torch.cat(imgs, dim=0)
        img_data = torch.stack(img_data)
        segmentations = torch.stack(segmentations)
        pregnancy_indices = torch.tensor(pregnancy_indices, dtype=torch.long)
        
        # Pregnancy-level data
        ehr_data = torch.stack([sample['ehr_data'] for sample in batch])
        masks = {task: {key: torch.stack([sample['masks'][task][key] for sample in batch])
                        for key in batch[0]['masks'][task]} for task in batch[0]['masks']}
        IDs = [sample['ID'] for sample in batch]
        treatment = [sample['treatment'] for sample in batch]
        labels = {task: {key: torch.stack([sample['labels'][task][key] for sample in batch])
                         for key in batch[0]['labels'][task]} for task in batch[0]['labels']}
        
        if self.target_size is not None:
            #Pad batch to batch_size
            n_pad = self.target_size - imgs.shape[0]
    
            if n_pad > 0:
                # Dummy pregnancy index
                dummy_idx = len(batch)
            
                # Pad images and image-level data
                imgs = torch.cat([imgs, torch.zeros((n_pad, *imgs.shape[1:]), dtype=imgs.dtype)])
                assert imgs.shape[0] == self.target_size
            
                img_data = torch.cat([img_data, torch.zeros((n_pad, *img_data.shape[1:]), dtype=img_data.dtype)])
            
                segmentations = torch.cat([segmentations, torch.zeros((n_pad, *segmentations.shape[1:]), dtype=segmentations.dtype)])
            
                img_paths.extend([None] * n_pad)
            
                # Assign padded images to dummy pregnancy
                pregnancy_indices = torch.cat([pregnancy_indices, torch.full((n_pad,), dummy_idx, dtype=torch.long)])
            
                # Add dummy pregnancy-level entry
                ehr_data = torch.cat([ehr_data, torch.zeros((1, *ehr_data.shape[1:]), dtype=ehr_data.dtype)])
            
                for task in labels:
                    for key in labels[task]:
                        labels[task][key] = torch.cat([labels[task][key], torch.zeros((1, *labels[task][key].shape[1:]),
                                                                                      dtype=labels[task][key].dtype)])

                        masks[task][key] = torch.cat([masks[task][key], torch.zeros((1, *masks[task][key].shape[1:]),
                                                                                    dtype=masks[task][key].dtype)])
                        
                IDs.append(None)
                treatment.append(None)
                
        return {"imgs": imgs,
                "img_paths": img_paths,
                "img_data": img_data,
                "ehr_data": ehr_data,
                "masks": masks,
                "IDs": IDs,
                "treatment": treatment,
                "labels": labels,
                "segmentations": segmentations,
                "pregnancy_indices": pregnancy_indices}


class PregnancyBatchSampler(Sampler):
    def __init__(self, image_counts, batch_size, shuffle=True):
        self.image_counts = image_counts
        self.batch_size = batch_size
        self.shuffle = shuffle

        # First-fit decreasing: place larger pregnancies first.
        order = sorted(range(len(image_counts)), key=lambda i: min(image_counts[i], batch_size), reverse=True)

        self.batches = []
        remaining = []

        for idx in order:
            n_images = min(image_counts[idx], batch_size)

            # Find the first batch with enough remaining capacity.
            placed = False
            for batch_idx, capacity in enumerate(remaining):
                if n_images <= capacity:
                    self.batches[batch_idx].append(idx)
                    remaining[batch_idx] -= n_images
                    placed = True
                    break

            if not placed:
                self.batches.append([idx])
                remaining.append(batch_size - n_images)

    def __iter__(self):
        batches = [batch.copy() for batch in self.batches]
        
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return len(self.batches)
    
class DataSplits:
    def __init__(self, cfg, save_path, unique_column='CPR_MOTHER', cutoffs=[28,32,34,37]):
        self.unique_column=unique_column
        self.oversample_ratio = cfg.data.oversample_ratio
        self.folds=cfg.data.folds
        self.cutoffs = sorted(cutoffs)
        
        if self.folds < 2:
            raise Exception(f"{self.folds} folds set, must be atleast 2")        

        if self.folds == 2:
            print('Warining: Only 2 folds selected, one fold will be used for validation and one for testing')
        
        test_df = pl.read_parquet(cfg.data.test_path)

        if cfg.data.train_path is not None:
            train_df = pl.read_parquet(cfg.data.train_path)
        else:
            train_df = pl.DataFrame(schema=test_df.schema)

        self.train_df = train_df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("fold"))
        
        # Group GA cutoffs
        groups = test_df.group_by(self.unique_column).agg(pl.col("GA").min().alias("GA"))

        ga_weeks = pl.col("GA") // 7

        expr = pl.lit(len(self.cutoffs))

        for i, cutoff in reversed(list(enumerate(self.cutoffs))):
            expr = (pl.when(ga_weeks < cutoff).then(i).otherwise(expr))

        groups = groups.with_columns(expr.alias("GA_group"))      
        # Distribute groups evenly within each GA group
        groups = (groups.with_columns(pl.int_range(pl.len())
                                      .shuffle().over("GA_group").alias("fold"))
                  .with_columns((pl.col("fold") % self.folds).alias("fold"))
                  .select([self.unique_column, "fold"]))

        # Assign the group's fold to every row
        self.test_df = test_df.join(groups, on=self.unique_column, how="left")
        
        #Check that self.unique_column is in exactly one fold
        assert (self.test_df.group_by(self.unique_column).agg(pl.col("fold").n_unique().alias("n_folds"))
                .filter(pl.col("n_folds") != 1).height == 0)
        
        self.save_distributions(save_path)
        
    def save_distributions(self, save_path):
        df = self.test_df.unique("CPR_CHILD")
        
        distributions = []
        
        for cutoff in self.cutoffs:
            inclusion = ((pl.col("GA") // 7 >= cutoff)
                         | (~pl.col("induced") & ~pl.col("c-section"))
                         | pl.col("pprom")
                         | (pl.col("c-section")
                            & (pl.col("c-section_during_birth") | pl.col("contractions_with_preterm_birth"))
                            & ~pl.col("induced")))
            
            counts = (df.filter(inclusion).group_by("fold").agg([pl.len().alias(f"n_{cutoff}"),
                                                                 (pl.col("GA") // 7 < cutoff).sum().alias(f"GA < {cutoff}")]))
            
            distributions.append(counts)
            
        fold_counts = distributions[0]
        
        for counts in distributions[1:]:
            fold_counts = fold_counts.join(counts, on="fold")

        fold_counts = fold_counts.sort("fold")
        
        fold_counts.write_csv(save_path + 'GA_distribution.csv')
            
        for fold in range(self.folds):
            test_df_fold = self.test_df.filter(pl.col("fold") == fold)

            test_df_fold.write_parquet(save_path + f"test_df_fold_{fold}.parquet")
        
        
    def get_split(self, fold, cfg):
        if not 0 <= fold < self.folds:
            raise Exception(f"Data only contains {self.folds} folds")

        test_df = self.test_df.filter(pl.col('fold') == fold)
        val_fold = (fold + 1) % self.folds
        val_df = self.test_df.filter(pl.col('fold') == val_fold)

        train_df = self.test_df.filter((pl.col('fold') != fold) & (pl.col('fold') != val_fold))
        train_df = pl.concat([self.train_df, train_df])
                
        for col in ["CPR_MOTHER", "CPR_CHILD", "file_path"]:
            overlap_test = (set(train_df[col].drop_nulls().unique()) & set(test_df[col].drop_nulls().unique()))
            overlap_val = (set(train_df[col].drop_nulls().unique()) & set(val_df[col].drop_nulls().unique()))
            if overlap_test:
                raise ValueError(f"{col}: contains {len(overlap_test)} overlaps in test/train. \n Examples: {list(overlap_test)[:10]}")
            if overlap_val:
                raise ValueError(f"{col}: contains {len(overlap_val)} overlaps in val/train. \n Examples: {list(overlap_val)[:10]}")

            
        if self.oversample_ratio != 0:
            pregnancies = train_df.group_by("CPR_CHILD").agg((pl.col("GA").min() // 7 < max(self.cutoffs)).alias("positive"))
            
            positives = pregnancies.filter(pl.col("positive"))
            negatives = pregnancies.filter(~pl.col("positive"))
            
            n_pos = len(positives)
            n_neg = len(negatives)
            
            if n_pos == n_neg:
                if n_pos == 0 or n_neg == 0:
                    raise Exception(f"Dataset contains {n_neg} negative and {n_pos} samples")
                else:
                    train_df = pl.concat([negatives, positives]) 
            else:
                if n_neg > n_pos:
                    if self.oversample_ratio == 'sqrt':
                        n_target = int(np.sqrt(n_neg*n_pos))
                    else:
                        n_target = int(len(negatives) * self.oversample_ratio)
                    extra = positives.sample(n=max(0, n_target - n_pos), with_replacement=True)
                elif n_pos > n_neg:
                    if self.oversample_ratio == 'sqrt':
                        n_target = int(np.sqrt(n_neg*n_pos))
                    else:
                        n_target = int(len(positives) * self.oversample_ratio)
                    extra = negatives.sample(n=max(0, n_target - n_neg), with_replacement=True)
                
                sampled_ids = pl.concat([pregnancies.select("CPR_CHILD"), extra.select("CPR_CHILD")])
                
                train_df = sampled_ids.join(train_df, on="CPR_CHILD", how="left")
        
        if (cfg.data.batch_size % 4) != 0:
            print(f"Batch size not divisible by 4. Changing it from {cfg.data.batch_size} to {cfg.data.batch_size - (cfg.data.batch_size % 4)}")
        
        batch_size = cfg.data.batch_size - (cfg.data.batch_size % 4)
        val_test_batch_size = batch_size // 4
        
        TrainData = PreTermDataset(train_df, cfg, train=True, max_images=batch_size)
        ValData = PreTermDataset(val_df, cfg, train=False, max_images=val_test_batch_size)
        TestData = PreTermDataset(test_df, cfg, train=False, max_images=val_test_batch_size)       
        
        TrainBatchSampler = PregnancyBatchSampler(TrainData.image_counts, batch_size, shuffle=True)

        ValBatchSampler = PregnancyBatchSampler(ValData.image_counts, val_test_batch_size, shuffle=False)

        TestBatchSampler = PregnancyBatchSampler(TestData.image_counts, val_test_batch_size, shuffle=False)

        TrainLoader = DataLoader(TrainData,
                                 batch_sampler=TrainBatchSampler,
                                 pin_memory=True,
                                 drop_last=False,
                                 num_workers=cfg.data.workers,
                                 collate_fn=CollateFn(target_size=batch_size))
        
        ValLoader = DataLoader(ValData,
                               batch_sampler=ValBatchSampler,
                               pin_memory=False,
                               drop_last=False,
                               num_workers=cfg.data.workers,
                               collate_fn=CollateFn())

        TestLoader = DataLoader(TestData,
                                batch_sampler=TestBatchSampler,
                                pin_memory=False,
                                drop_last=False,
                                num_workers=cfg.data.workers,
                                collate_fn=CollateFn())
        
        return TrainLoader, ValLoader, TestLoader
        