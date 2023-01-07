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

plt.switch_backend('agg')

def train(model, train_dataloader, val_dataloader, device, config):
    loss_criterion = nn.CrossEntropyLoss().to(device)
 
    optimizer = torch.optim.Adam(model.parameters(),config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["scheduler_factor"], patience=config["scheduler_patience"])
    early_stopper = EarlyStopper(patience=config['early_stopping_patience'], min_delta=0.0)

    logger = SummaryWriter()

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

        print(f'[{epoch + 1:03d}] train_loss: {train_loss_running / len(train_dataloader):.6f}')
        logger.add_scalar('loss/train_classification', train_loss_running / len(train_dataloader), epoch)
        train_loss_running = 0.0

        # batch['images'] -> [batch, views, 3, 137, 137] #
        # predicted_labels -> [batch]                    #
        # predictions -> [batch, 13]                     #
        # target -> [batch]                              #
        if(config['plot_train_images'] == True):
            logger.add_figure('train/predictions vs. actuals', plot_classes_preds(batch['images'], predicted_labels, predictions, target), epoch)

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

            if config['early_stopping'] == True and early_stopper.early_stop(loss_val):
                break

            logger.add_scalar('loss/val_classification', loss_val, epoch)
            logger.add_figure('val/predictions vs. actuals', plot_classes_preds(batch_val['images'], predicted_labels, predictions, target), epoch)

            if loss_val < best_loss_val:
                torch.save(model.state_dict(), f'./saved_models/{config["experiment_name"]}/model_best_loss.ckpt')
                best_loss_val = loss_val

            print(f'[{epoch:03d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

            accuracy = 100 * correct / total
            logger.add_scalar('loss/val_acc', accuracy, epoch)

            # Run scheduler step #
            scheduler.step(loss_val)
            logger.add_scalar('loss/lr_rate', scheduler.optimizer.param_groups[0]['lr'], epoch)

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), f'./saved_models/{config["experiment_name"]}/model_best_acc.ckpt')
                best_accuracy = accuracy

        model.train()

###########
# Helpers #
###########

# Plot the images in the batch, along with predicted and true labels #
def plot_classes_preds(images, predicted_labels, predictions, labels):
    fig = plt.figure(figsize=(10, 5))

    probs_max, _ = torch.max(F.softmax(predictions, dim=1), dim=1)

    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx][0])

        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            ShapeNetDataset.index_to_class(predicted_labels[idx].item()),
            probs_max[idx] * 100,
            ShapeNetDataset.index_to_class(labels[idx].item())),
                    color=("green" if labels[idx]==predicted_labels[idx].item() else "red"))

    return fig

def matplotlib_imshow(img):
    img = ShapeNetDataset.denormalize_image(img) # Remove ImageNet mean/std 
    img = img.permute(1, 2, 0).numpy().astype("uint8")

    plt.imshow(img)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

##################################

def main(config):
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

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device #
    model.to(device)

    # Create folder for saving checkpoints #
    Path(f'saved_models/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Train - models saved based on val loss and acc #
    train(model, train_dataloader, val_dataloader, device, config)

if __name__ == "__main__":

    # Seeds #
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)

    config = {
            'experiment_name': 'mvcnn_overfitting',
            'device': 'cuda:0',
            'is_overfit': True,
            'batch_size': 8,
            'resume_ckpt': None,#'./saved_models/mvcnn_overfitting/model_best_acc.ckpt',
            'learning_rate': 0.00001,
            'max_epochs': 350,
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

    main(config)
