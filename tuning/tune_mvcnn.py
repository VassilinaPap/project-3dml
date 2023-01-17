from pathlib import Path

import sys
sys.path.append('../models')
sys.path.append('../src')

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
import random

import matplotlib.pyplot as plt

from MVCNN import MVCNN
from dataset import ShapeNetDataset
import optuna

plt.switch_backend('agg')

def ioU(predictions_rec, voxel):
    # prob to vox grid
    predictions_rec[predictions_rec >= 0.5] = 1
    predictions_rec[predictions_rec < 0.5] = 0

    intersection = (predictions_rec + voxel)
    #intersection = torch.count_nonzero(intersection == 2).item()
    intersection = torch.numel(intersection[intersection==2])
    union = predictions_rec.sum().item() + voxel.sum().item()
    return (intersection/union) * 100

def objective(trial):
    config = {
            'experiment_name': 'mvcnn_tuningmb',
            'device': 'cuda:0',
            'is_overfit': False,
            'batch_size': 64,
            'resume_ckpt': None,
            'learning_rate': 0.00001,
            'max_epochs': 5,
            'validate_every_n': 5, # In epochs 
            'num_views': 3,
            'augmentation_json_flag': False,
            'augmentations_flag': False,
            'plot_train_images': True,
            'early_stopping': False,
            'early_stopping_patience': 10,
            'scheduler_factor': 0.1,
            'scheduler_patience': 5,
            "cl_weight": 0.5,
            'flag_rec':True,
            'flag_multibranch':True
    }

    config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    #config['batch_size'] = trial.suggest_categorical('batch_size', [4, 8])
    #config['cl_weight'] = trial.suggest_float('cl_weight', 0.0, 1.0)

    # Declare device #
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetDataset(split='train' if not config['is_overfit'] else 'overfit', num_views=config['num_views'], augmentation_json_flag=config['augmentation_json_flag'], augmentations_flag=config['augmentations_flag'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = ShapeNetDataset(split='val' if not config['is_overfit'] else 'overfit', num_views=config['num_views'], augmentation_json_flag=config['augmentation_json_flag'], augmentations_flag=config['augmentations_flag'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Instantiate model #
    model = MVCNN(num_views=config['num_views'],flag_rec = config['flag_rec'],flag_multibranch = config['flag_multibranch'])

    # Move model to specified device #
    model.to(device)

    loss_criterion_cl = nn.CrossEntropyLoss().to(device)
    loss_criterion_rec = nn.BCELoss().to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])

    model.train()
    best_loss_val = np.inf

    best_accuracy = 0.0

    for epoch in range(config['max_epochs']):

        train_loss_running = 0.
        train_iou = 0.

        for batch_idx, batch in enumerate(train_dataloader):
            ShapeNetDataset.move_batch_to_device(batch, device)

            optimizer.zero_grad()

            predictions_cl,predictions_rec = model(batch['images'])

            _, predicted_labels = torch.max(predictions_cl, dim=1)
            target = batch['label']
            loss_cl = loss_criterion_cl(predictions_cl, target)

            if config['flag_rec']:
                voxel = batch['voxel']
                # TODO: Compute loss, Compute gradients, Update network parameters
                
                loss_rec = loss_criterion_rec(predictions_rec,voxel)
                loss = config["cl_weight"] * loss_cl + (1 - config["cl_weight"]) * loss_rec
                iou = ioU(predictions_rec.detach().clone(),voxel)
                train_iou += iou
            else:
                loss = loss_cl

            loss.backward()

            # TODO: update network params
            optimizer.step()

            # Logging #
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

        # train_loss_running = 0.0
        # train_iou = 0

        # Validation evaluation #
        if epoch % config['validate_every_n'] == (config['validate_every_n'] - 1):
            model.eval()

            loss_val = 0.
            val_iou = 0.
            total = 0.0
            correct = 0.0
            for batch_val in val_dataloader:
                ShapeNetDataset.move_batch_to_device(batch_val, device)

                with torch.no_grad():
                    predictions_cl,predictions_rec = model(batch_val['images'])
                    _, predicted_labels = torch.max(predictions_cl, dim=1)
                    target = batch_val['label']
                    val_loss_cl = loss_criterion_cl(predictions_cl, target)

                    if config['flag_rec']:
                        val_loss_rec = loss_criterion_rec(predictions_rec, batch_val['voxel'])
                        val_loss = 0.5 * val_loss_cl + 0.5 * val_loss_rec
                        iou = ioU(predictions_rec.detach().clone(),batch_val['voxel'])
                        val_iou += iou
                    else:
                        val_loss = val_loss_cl


                total += predicted_labels.shape[0]
                correct += (predicted_labels == target).sum().item()

            loss_val += val_loss.item()
            loss_val /= len(val_dataloader)

            trial.report(loss_val, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            accuracy = 100 * correct / total

        model.train()

    return loss_val

##################################

if __name__ == "__main__":

    # Seeds #
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)

    for key, value in study.best_trial.params.items():
        print(str(key) + ": " + str(value))
