from pathlib import Path

import sys
sys.path.append('../models')
sys.path.append('../src')

from MVCNN import MVCNN
from dataset import ShapeNetDataset

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import random
import torch.nn.functional as F

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

def train(model, train_dataloader, val_dataloader, device, config):
    # TODO: Declare loss and move to device; we need both smoothl1 and pure l1 losses here
    loss_criterion_cl = nn.CrossEntropyLoss().to(device)
    loss_criterion_rec = nn.BCELoss().to(device)

    # TODO: Declare optimizer with learning rate given in config
    optimizer = torch.optim.Adam(model.parameters(),config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["scheduler_factor"], patience=config["scheduler_patience"])
    early_stopper = EarlyStopper(patience=config['early_stopping_patience'], min_delta=0.0)

    logger = SummaryWriter()

    model.train()
    best_loss_val = np.inf
    best_accuracy = 0.
    best_iou = 0.
    train_loss_running = 0.0

    for epoch in range(config['max_epochs']):
        train_loss_running = 0.
        train_iou = 0.
        for batch_idx, batch in enumerate(train_dataloader):
            ShapeNetDataset.move_batch_to_device(batch, device)

            optimizer.zero_grad()
            predictions_cl,predictions_rec = model(batch['images'])

            _, predicted_labels = torch.max(predictions_cl, dim=1)
            target = batch['label']
            
            # TODO: Compute loss, Compute gradients, Update network parameters
            loss_cl = loss_criterion_cl(predictions_cl, target)

            if config['flag_rec']:
                voxel = batch['voxel']
                loss_rec = loss_criterion_rec(predictions_rec,voxel)
                loss = config["cl_weight"] * loss_cl + (1 - config["cl_weight"]) *loss_rec
                iou = ioU(predictions_rec.detach().clone(),voxel)
                train_iou += iou
            else:
                loss =  loss_cl

            loss.backward()

            # TODO: update network params
            optimizer.step()

            # Logging #
            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

        print(f'[{epoch + 1:03d}] train_loss: {train_loss_running / len(train_dataloader):.6f}')
        logger.add_scalar('loss/train_classification', train_loss_running / len(train_dataloader), epoch)
        #train_loss_running = 0.0
        #train_iou = 0.

        # batch['images'] -> [batch, views, 3, 137, 137] #
        # predicted_labels -> [batch]                    #
        # predictions -> [batch, 13]                     #
        # target -> [batch]                              #
        if(config['plot_train_images'] == True):
            logger.add_figure('train/predictions vs. actuals', plot_classes_preds(batch['images'], predicted_labels, predictions_cl, target, batch['class'], config["plot_images_num"]), epoch)

        # Validation evaluation and logging
        if epoch % config['validate_every_n'] == (config['validate_every_n'] - 1):
            model.eval()
            # Evaluation on entire validation set
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
                        val_loss = config["cl_weight"] * val_loss_cl + (1 - config["cl_weight"]) * val_loss_rec
                        iou = ioU(predictions_rec.detach().clone(),batch_val['voxel'])
                        val_iou += iou
                    else:
                        val_loss =  val_loss_cl
                    

                total += predicted_labels.shape[0]
                correct += (predicted_labels == target).sum().item()

            loss_val += val_loss.item()
            loss_val /= len(val_dataloader)

            if config['early_stopping'] == True and early_stopper.early_stop(loss_val):
                break

            logger.add_scalar('val/loss', loss_val, epoch)
            logger.add_figure('val/predictions vs. actuals', plot_classes_preds(batch_val['images'], predicted_labels, predictions_cl, target, batch['class'], config["plot_images_num"]), epoch)

            if loss_val < best_loss_val:
                torch.save(model.state_dict(), f'./saved_models/{config["experiment_name"]}/model_best_loss.ckpt')
                best_loss_val = loss_val

            print(f'\n[{epoch + 1:03d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

            accuracy = 100 * correct / total
            logger.add_scalar('val/acc', accuracy, epoch)

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), f'./saved_models/{config["experiment_name"]}/model_best_acc.ckpt')
                best_accuracy = accuracy

            print('Accuracy:' + '{:5}'.format(correct) + '/' +
                  '{:5}'.format(total) + ' (' +
                  '{:4.2f}'.format(100.0 * correct / total) + '%)' + ' | best_acc: ' + '{:2.2f}'.format(best_accuracy) + ' %')

            if config['flag_rec']:
                val_iou /= len(val_dataloader)
                if val_iou > best_iou:
                    torch.save(model.state_dict(), f'./saved_models/{config["experiment_name"]}/model_best_iou.ckpt')
                    best_iou = val_iou

                logger.add_scalar('val/iou', val_iou, epoch)
                print(f'[{epoch + 1:03d}] IoU: {val_iou:.6f} | best_iou: {best_iou:.6f}\n')

            if(config["enable_scheduler"]):
                scheduler.step(loss_val)

        model.train()

# plot the images in the batch, along with predicted and true labels
def plot_classes_preds(images, predicted_labels, predictions, labels, classes, plot_images_num):
    # batch['images'] -> [batch, views, 3, 137, 137] # 
    fig = plt.figure(figsize=(10, 5))

    probs_max, _ = torch.max(F.softmax(predictions, dim=1), dim=1)

    for idx in np.arange(plot_images_num):
        ax = fig.add_subplot(1, plot_images_num, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx][0])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            ShapeNetDataset.index_to_class(predicted_labels[idx].item()),
            probs_max[idx] * 100,
            classes[idx]),
                    color=("green" if labels[idx]==predicted_labels[idx].item() else "red"))

    return fig 

def matplotlib_imshow(img):
    img = ShapeNetDataset.denormalize_image(img)
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

    # Instantiate model
    model = MVCNN(num_views=config['num_views'],flag_rec = config['flag_rec'], flag_multibranch = config['flag_multibranch'])

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
        'experiment_name': 'mvcnn_trainingmbranch3v',
        'device': 'cuda:0',
        'is_overfit': False,
        'batch_size': 64,
        'resume_ckpt': None,#'./saved_models/mvcnn_overfitting/model_best_acc.ckpt',
        'learning_rate': 0.00001,
        'max_epochs': 250,
        'validate_every_n': 5, # In epochs 
        'num_views': 24,
        'augmentation_json_flag': False,
        'augmentations_flag': False,
        'plot_train_images': True,
        'early_stopping': True,
        'early_stopping_patience': 10,
        'enable_scheduler': False,
        'scheduler_factor': 0.1,
        'scheduler_patience': 5,
        'cl_weight': 0.5,
        'plot_images_num': 1,
        'flag_rec': True,
        'flag_multibranch':True

    }

    print("=======")
    print("hparams")
    print("=======")
    print("lr: " + str(config["learning_rate"]))
    print("cl_weight: " + str(config["cl_weight"]))
    print("max_epochs: " + str(config["max_epochs"]))
    print("num_views: " + str(config["num_views"]))
    print("augmentation_json_flag: " + str(config["augmentation_json_flag"]))
    print("augmentations_flag: " + str(config["augmentations_flag"]))
    print("early_stopping: " + str(config["early_stopping"]))
    print("early_stopping_patience: " + str(config["early_stopping_patience"]))
    print("enable_scheduler: " + str(config["enable_scheduler"]))
    print("scheduler_factor: " + str(config["scheduler_factor"]))
    print("scheduler_patience: " + str(config["scheduler_patience"]))
    print("plot_images_num: " + str(config["plot_images_num"]))

    main(config)
