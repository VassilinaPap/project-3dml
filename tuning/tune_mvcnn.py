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

def objective(trial):
    config = {
            'experiment_name': 'mvcnn_overfitting',
            'device': 'cuda:0',
            'is_overfit': True,
            'batch_size': 8,
            'resume_ckpt': None,
            'learning_rate': 0.00001,
            'max_epochs': 5,
            'validate_every_n': 5, # In epochs 
            'num_views': 6,
            'augmentation_json_flag': False,
            'augmentations_flag': False,
            'plot_train_images': True,
            'early_stopping': True,
            'early_stopping_patience': 10,
            'scheduler_factor': 0.1,
            'scheduler_patience': 5
    }

    config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    config['batch_size'] = trial.suggest_categorical('batch_size', [4, 8])

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
    model = MVCNN(num_views=config['num_views'])

    # Move model to specified device #
    model.to(device)

    loss_criterion = nn.CrossEntropyLoss().to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])


    model.train()
    best_loss_val = np.inf

    best_accuracy = 0.0
    train_loss_running = 0.0

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            ShapeNetDataset.move_batch_to_device(batch, device)

            optimizer.zero_grad()

            # Predict classes - [batch, classes] #
            predictions = model(batch['images'])

            # Get indexes #
            _, predicted_labels = torch.max(predictions, dim=1)
            target = batch['label']

            # BCE #
            loss = loss_criterion(predictions, target)
            loss.backward()
            optimizer.step()

            # Logging #
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

        train_loss_running = 0.0

        # Validation evaluation #
        if epoch % config['validate_every_n'] == (config['validate_every_n'] - 1):
            model.eval()

            loss_val = 0.
            total = 0.0
            correct = 0.0

            for batch_val in val_dataloader:
                ShapeNetDataset.move_batch_to_device(batch_val, device)

                with torch.no_grad():
                    predictions = model(batch_val['images'])
                    _, predicted_labels = torch.max(predictions, dim=1)
                    val_loss = loss_criterion(predictions, batch_val['label'])
                    target = batch_val['label']

                total += predicted_labels.shape[0]
                correct += (predicted_labels == batch_val["label"]).sum().item()

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
