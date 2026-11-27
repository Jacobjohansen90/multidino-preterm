#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tmc
import torchmetrics.regression as tmr
import polars as pl
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class Metrics():
    def __init__(self, cfg, save_path, n_bootstrap=10000):
        self.epoch = 0
        self.fold = 0
        self.n_bootstrap = n_bootstrap
        self.save_path = Path(save_path)
        
        metric_fns = {'classification': self.get_metrics('classification'),
                      'regression': self.get_metrics('regression')}
        
        self.keys = []
        self.dfs = {}
        self.metric_fns = {}
        self.tracking_metrics = {}
        for task, task_cfgs in cfg.tasks.items():
            self.dfs[task] = {}
            self.metric_fns[task] = {}
            self.tracking_metrics[task] = {}
            for task_cfg in task_cfgs:
                if task_cfg.get('compute_metrics', False):
                    var = task_cfg.variable
                    var = var if isinstance(var, str) else '_'.join(var)

                    if task_cfg.get('cutoffs'):
                        for cutoff in task_cfg.cutoffs:
                            var_temp = f'{var}_{cutoff}'
                            self.keys.append([task, var_temp, cutoff, task_cfg.metric_type])
                            self.dfs[task][var_temp] = []
                            self.metric_fns[task][var_temp] = metric_fns[task_cfg.metric_type]
                            self.tracking_metrics[task][var_temp] = task_cfg.tracking_metric

                    else:
                        self.keys.append([task, var, None, task_cfg.metric_type])
                        self.dfs[task][var] = []
                        self.metric_fns[task][var] = metric_fns[task_cfg.metric_type]
                        self.tracking_metrics[task][var] = task_cfg.tracking_metric
              
        self.best_score = {}
        self.best_dfs = {}
        self.metric_history = {}
        self.test_preds = {}
        self.val_preds = {}
        for task, var, *_ in self.keys:
            self.best_score.setdefault(task, {})[var] = {'score': -float('inf'),
                                                         'threshold': None,
                                                         'epoch': -1}
            self.best_dfs.setdefault(task, {})[var] = None
            self.metric_history.setdefault(task, {})[var] = []
            self.test_preds.setdefault(task, {})[var] = []
            self.val_preds.setdefault(task, {})[var] = []

        self.metrics_path = Path(self.save_path) / "metrics"
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        self.plot_path = Path(self.save_path) / "plots"
        self.plot_path.mkdir(parents=True, exist_ok=True)
        
        self.tasks = len(self.keys)
        self.task_counter = 0

        self.pred_path = Path(self.save_path) / "predictions" 
        (self.pred_path / "validation").mkdir(parents=True, exist_ok=True)


    def update(self, outputs, data, test=False):
        if test is False:
            for task, var, cutoff, _ in self.keys:
                mask = data['masks'][task][var]
                labels = data['labels'][task][var]
                if cutoff is not None:
                    labels = labels < cutoff
                #Make img_paths pregnancy level
                img_paths = [[path for path, idx in zip(data['img_paths'], data['pregnancy_indices'].cpu().numpy()) if idx == i]
                             for i in range(len(data['IDs']))]
                df = pl.DataFrame({'CPR_CHILD': data['IDs'],
                                   'img_paths': img_paths,
                                   'preds': outputs[task][var]['preds'].flatten().cpu().numpy(),
                                   'labels': labels.flatten().cpu().numpy(),
                                   'treatment': data['treatment'],
                                   'fold': self.fold}).filter(mask.flatten().cpu().numpy())
                if df.height > 0:
                    self.dfs[task][var].append(df)
        else:
            task, var, cutoff, _ = self.keys[self.task_counter - 1]
            mask = data['masks'][task][var]
            labels = data['labels'][task][var]
            if cutoff is not None:
                labels = labels < cutoff
            #Make img_paths pregnancy level
            img_paths = [[path for path, idx in zip(data['img_paths'], data['pregnancy_indices'].cpu().numpy()) if idx == i]
                         for i in range(len(data['IDs']))]
            df = pl.DataFrame({'CPR_CHILD': data['IDs'],
                               'img_paths': img_paths,
                               'preds': outputs[task][var]['preds'].flatten().cpu().numpy(),
                               'labels': labels.flatten().cpu().numpy(),
                               'treatment': data['treatment'],
                               'threshold': self.best_score[task][var]['threshold'],
                               'fold': self.fold}).filter(mask.flatten().cpu().numpy())
            if df.height > 0:
                self.dfs[task][var].append(df)

    def log_epoch(self, train_loss, val_loss):
        losses = {}

        for loss_task in train_loss:
            losses[f'train_loss_{loss_task}'] = np.average(train_loss[loss_task]['loss'],
                                                           weights=train_loss[loss_task]['count'])
            losses[f'val_loss_{loss_task}'] = np.average(val_loss[loss_task]['loss'],
                                                         weights=val_loss[loss_task]['count'])
        
        losses['train_loss'] = sum(value for key, value in losses.items() if key.startswith('train_loss_'))
        losses['val_loss'] = sum(value for key, value in losses.items() if key.startswith('val_loss_'))
        
        for task, var, cutoff, task_type in self.keys:
            df = pl.concat(self.dfs[task][var])
            metrics = {}
            preds = torch.tensor(df['preds'].to_numpy(), dtype=torch.float32)
            if task_type == 'classification':
                labels = torch.tensor(df['labels'].to_numpy(), dtype=torch.int32)
            else:
                labels = torch.tensor(df['labels'].to_numpy(), dtype=torch.float32)
 
            for name, metric in self.metric_fns[task][var].items():
                metric.reset()
                output = metric(preds, labels)
                
                if isinstance(output, tuple):
                    metrics[name] = round(output[0].item(), 3)
                    metrics[f"{name}_cutoff"] = round(output[1].item(), 3)
                else:
                    metrics[name] = round(output.item(), 3)
            
            if metrics[self.tracking_metrics[task][var]] > self.best_score[task][var]['score']:
                self.best_score[task][var]['score'] = metrics[self.tracking_metrics[task][var]]
                if 'SensAtSpec' in metrics.keys():
                    self.best_score[task][var]['threshold'] = metrics['SensAtSpec_cutoff']
                self.best_score[task][var]['epoch'] = self.epoch
                df = df.with_columns(pl.lit(self.best_score[task][var]['threshold']).alias('threshold'))
                self.best_dfs[task][var] = df
                
            self.metric_history[task][var].append({'epoch': self.epoch,
                                                   **losses,
                                                   **metrics})
            metric_df = pl.DataFrame(self.metric_history[task][var])
            metric_df.write_csv(self.metrics_path / f"metrics_{task}_{var}_fold_{self.fold}.csv")

            self.dfs[task][var] = []
            
            self.plot_metrics(metric_df, task, var)
            
        self.epoch += 1
            

    def plot_metrics(self, metric_df, task, var):
        loss_cols = [col for col in metric_df.columns if 'loss' in col]
        metric_cols = [col for col in metric_df.columns if col not in ['epoch'] + loss_cols]
                       
        linestyles = ['--', ':', '-.', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1))]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][2:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        
        
        for prefix, color in [('train', 'tab:blue'), ('val', 'tab:orange')]:
            cols = [col for col in loss_cols if col.startswith(f'{prefix}_loss')]

            sub_idx = 0
            for col in cols:
                is_total = col == f'{prefix}_loss'
        
                ax.plot(metric_df['epoch'],
                        metric_df[col],
                        label=col,
                        color=color,
                        linestyle='-' if is_total else linestyles[sub_idx % len(linestyles)])
        
                if not is_total:
                    sub_idx += 1
        
        for i, col in enumerate(metric_cols):
            ax.plot(metric_df['epoch'], metric_df[col], label=col, color=colors[i])
        
        ax.set_title(f"Fold {self.fold} - {task} - {var}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        fig.savefig(self.plot_path / f"Fold {self.fold} - {task} - {var}.png", dpi=300)
        plt.close(fig)
         
        
    def get_epoch(self):
        task, var, *_ = self.keys[self.task_counter]
        epoch = self.best_score[task][var]['epoch']
        self.task_counter += 1
        return epoch


    def log_fold(self):
        for task, var, _, task_type in self.keys:
            df = pl.concat(self.dfs[task][var])

            if task_type == 'classification':
                df = df.with_columns((pl.col("preds") >= pl.col("threshold")).cast(pl.float32).alias("preds_binary"),
                                     (pl.col("preds") + 0.5 - pl.col("threshold")).clip(0, 1).alias("preds_normalized"))
            
            self.test_preds[task][var].append(df)
            pl.concat(self.test_preds[task][var]).write_parquet(self.pred_path / f"prediction_{task}_{var}.parquet")      
                    
        self.reset()
        self.fold += 1

    def reset(self):
        self.epoch = 0
        self.task_counter = 0
        for task, var, *_ in self.keys:
            path = self.pred_path / "validation" / f"val_metrics_{task}_{var}.parquet"
            self.val_preds[task][var].append(self.best_dfs[task][var])
            if self.fold != 0:
                df = pl.concat([pl.read_parquet(path), self.best_dfs[task][var]])
            else:
                df = self.best_dfs[task][var]
            
            df.write_parquet(path)
            
            self.dfs[task][var] = []
            self.best_score[task][var] = {'score': -float('inf'),
                                          'threshold': None,
                                          'epoch': -1}
            self.best_dfs[task][var] = None
            self.metric_history[task][var] = []
            

    def log_model(self):
        for task, var, _, task_type in self.keys:
            df = pl.concat(self.val_preds[task][var])
            val_metrics = {}
            #Get validation metrics
            for pop, pop_df in [('All Births', df), ('Non-treated', df.filter(~pl.col('treatment')))]:
                if task_type == 'classification':
                    pop_df = pop_df.with_columns((pl.col('preds') - pl.col('threshold') + 0.5).clip(0, 1).alias('preds_normalized'),
                                                 (pl.col('preds') >= pl.col('threshold')).cast(pl.Int8).alias('preds_binary'))
                
                val_metrics[pop] = {'overall': {}, 'folds': {metric: [] for metric in self.metric_fns[task][var].keys()}}

                #Get overall metric
                for name, metric in self.metric_fns[task][var].items():
                    val_metrics[pop]['overall'][name] = self.calculate_metric(pop_df, name, metric)
                    
                #Get fold level metrics
                for fold in pop_df['fold'].unique().to_list():
                    fold_df = pop_df.filter(pl.col('fold') == fold)
                    for name, metric in self.metric_fns[task][var].items():
                        val_metrics[pop]['folds'][name].append(self.calculate_metric(fold_df, name, metric))

            self.plot_fold_variability(val_metrics, f"{task} - {var}")

            df = pl.concat(self.test_preds[task][var])
            test_metrics = {}
            #Get test metrics
            for pop, pop_df in [('All Births', df), ('Non-treated', df.filter(~pl.col('treatment')))]:
                if task_type == 'classification':
                    pop_df = pop_df.with_columns((pl.col('preds') - pl.col('threshold') + 0.5).clip(0, 1).alias('preds_normalized'),
                                                 (pl.col('preds') >= pl.col('threshold')).cast(pl.Int8).alias('preds_binary'))
                
                test_metrics[pop] = {metric: {} for metric in self.metric_fns[task][var]}
                for name, metric in self.metric_fns[task][var].items():
                    res, ci_lower, ci_upper = self.calculate_metric(pop_df, name, metric, task_type, calculate_CI=True)
                    test_metrics[pop][name]['overall'] = res
                    test_metrics[pop][name]['ci_lower'] = ci_lower
                    test_metrics[pop][name]['ci_upper'] = ci_upper

            report = []
            for pop in test_metrics:
                pop_df = df if pop == 'All Births' else df.filter(~pl.col('treatment'))

                n_preterm = pop_df['labels'].sum()
                n_non_preterm = pop_df.height - n_preterm

                report.append(f"-- {pop} --\n"
                              f"\tPreterm births: {n_preterm}\n"
                              f"\tNon-preterm births: {n_non_preterm}\n"
                              f"\tTotal births: {pop_df.height}\n\n"
                              f"\tMetrics:\n")
            
                for name, result in test_metrics[pop].items():
                    val = val_metrics[pop]['overall'][name]

                    report.append(f"\t{name}: {result['overall']:.4f} "
                                  f"(95% CI: {result['ci_lower']:.4f}–{result['ci_upper']:.4f}) "
                                  f"(Validation: {val:.4f})\n")

                report.append('\n\n')
            report = "".join(report)
            
            with open(self.save_path / f"{task}_{var}.txt", "w") as f:
                f.write(report)
                
            
    def calculate_metric(self, df, name, metric, task_type, calculate_CI=False):
        if task_type == 'classification':
            labels = torch.tensor(df['labels'].to_numpy(), dtype=torch.int32)
        else:
            labels = torch.tensor(df['labels'].to_numpy(), dtype=torch.float32)
        if name == 'AUC':
            preds = df['preds_normalized']
        elif name in ['Sensitivity', 'Specificity']:
            preds = df['preds_binary']
        else:
            preds = df['preds']
        
        metric.reset()
        output = metric(torch.tensor(preds.to_numpy(), dtype=torch.float32), labels)
        result = output[0].item() if isinstance(output, tuple) else output.item()
        
        if calculate_CI:
            ci_lower, ci_upper = self.calculate_CI(df, name, metric)
            return result, ci_lower, ci_upper
        else:
            return result
    
    
    def calculate_CI(self, df, name, metric):
        rng = np.random.default_rng()
        bootstrap_metrics = []
        counter = 0
        while len(bootstrap_metrics) < self.n_bootstrap:
            counter += 1
            #Break loop if we continue failing
            if counter > 2*self.n_bootstrap:
                return np.nan, np.nan

            indices = rng.choice(df.height, size=df.height, replace=True)
            bootstrap_df = df[indices]

            #Check if both labels are present
            if bootstrap_df['labels'].sum() == 0 or bootstrap_df['labels'].sum() == bootstrap_df.height:
                continue
            
            result = self.calculate_metric(bootstrap_df, name, metric)
            bootstrap_metrics.append(result)
        
        ci_lower, ci_upper = np.percentile(bootstrap_metrics, [2.5, 97.5])
        
        return ci_lower, ci_upper
            
    
    def plot_fold_variability(self, metrics, figure_name):

        for pop, pop_metrics in metrics.items():
            metrics = list(pop_metrics['folds'].keys())
    
            mins = [np.min(pop_metrics['folds'][m]) for m in metrics]
            maxs = [np.max(pop_metrics['folds'][m]) for m in metrics]
            overall = [pop_metrics['overall'][m] for m in metrics]
    
            x = np.arange(len(metrics))
    
            fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.2), 5))
    
            # Fold variability
            ax.vlines(x, mins, maxs, color='steelblue', linewidth=2)
            ax.scatter(x, mins, marker='_', color='steelblue', s=100)
            ax.scatter(x, maxs, marker='_', color='steelblue', s=100)
    
            # Overall metric
            ax.scatter(x, overall, color='steelblue', marker='o', s=50, zorder=3)
    
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylabel('Metric value')
            ax.set_xlabel('Metric')
            ax.set_title(f'Validation fold variability - {pop}')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
    
            ax.legend(handles=[plt.Line2D([0], [0], color='steelblue', linewidth=2, label='Fold range'),
                               plt.Line2D([0], [0], marker='o', color='steelblue', linestyle='None', label='Overall')])
    
            fig.tight_layout()
    
            filename = f"{figure_name} - {pop}.png"
            fig.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            
    def get_metrics(self, task_type, t=0.5):
        if task_type == 'classification':
            metrics = {'Sensitivity': tmc.Recall(task='binary', threshold=t),
                       'Specificity': tmc.Specificity(task='binary', threshold=t),
                       'SensAtSpec': tmc.SensitivityAtSpecificity(min_specificity=0.85, task='binary'),
                       'AUC': tmc.AUROC(task='binary')}
            
        elif task_type == 'regression':
            metrics = {'MSE': tmr.MeanSquaredError()}
            
        else:
            raise ValueError(f"Task type {task_type} not understood")
            
        return metrics
    
        
        