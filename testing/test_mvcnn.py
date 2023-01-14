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

from utils import read_as_3d_array, save_voxel_grid

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

def test(model, test_dataloader, device, config):
    loss_criterion_cl = nn.CrossEntropyLoss().to(device)
    loss_criterion_rec = nn.BCELoss().to(device)

    logger = SummaryWriter()

    model.eval()

    # Evaluation on entire validation set
    loss_test = 0.
    test_iou = 0.
    total = 0.0
    correct = 0.0

    best_batch_iou = -np.inf
    best_batch_iou_id = 0
    best_batch_iou_data = None
    best_batch_iou_labels = None
    for batch_test in test_dataloader:
        ShapeNetDataset.move_batch_to_device(batch_test, device)

        with torch.no_grad():
            predictions_cl,predictions_rec = model(batch_test['images'])
            _, predicted_labels = torch.max(predictions_cl, dim=1)
            test_loss_cl = loss_criterion_cl(predictions_cl, batch_test['label'])
            test_loss_rec = loss_criterion_rec(predictions_rec, batch_test['voxel'])
            test_loss = config["cl_weight"] * test_loss_cl + (1 - config["cl_weight"]) * test_loss_rec
            iou = ioU(predictions_rec.detach().clone(),batch_test['voxel'])
            test_iou += iou
            target = batch_test['label']

            if(iou > best_batch_iou):
                best_batch_iou = iou
                best_batch_iou_data = predictions_rec.cpu().numpy()
                best_batch_iou_labels = batch_test['label']

        total += predicted_labels.shape[0]
        correct += (predicted_labels == batch_test["label"]).sum().item()

    loss_test += test_loss.item()
    loss_test /= len(test_dataloader)

    logger.add_scalar('loss/test_classification', loss_test, 0)
    logger.add_figure('test/predictions vs. actuals', plot_classes_preds(batch_test['images'], predicted_labels, predictions_cl, target, batch_test['class'], config["plot_images_num"]), 0)

    print(f'test_loss: {loss_test:.6f}')

    accuracy = 100 * correct / total
    logger.add_scalar('loss/test_acc', accuracy, 0)

    print('\nAccuracy:' + '{:5}'.format(correct) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100.0 * correct / total) + '%)\n')

    test_iou /= len(test_dataloader)

    logger.add_scalar('test/iou', test_iou, 0)
    print(f'IoU: {test_iou:.6f}')

    # Save best batch recon #
    print("Saving the reconstructions of the best batch with IoU: " + str(best_batch_iou))
    for i in range(config["batch_size"]):
        class_tmp = ShapeNetDataset.index_to_class(best_batch_iou_labels[i].item())
        save_voxel_grid(config["recon_folder"] + "/" + str(class_tmp)  + ".ply", best_batch_iou_data[i, :, :, :])

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
    test_dataset = ShapeNetDataset(split='test', num_views=config['num_views'])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Instantiate model
    model = MVCNN(num_views=config['num_views'])

    model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device #
    model.to(device)

    # Train - models saved based on val loss and acc #
    test(model, test_dataloader, device, config)

if __name__ == "__main__":

    # Seeds #
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)

    config = {
        'experiment_name': 'mvcnn_overfitting',
        'device': 'cuda:0',
        'batch_size': 8,
        'resume_ckpt': '../training/saved_models/mvcnn_overfitting/model_best_iou.ckpt',
        'num_views': 2,
        'cl_weight': 0.5,
        'plot_images_num': 1,
        'recon_folder': "./recon"
    }

    Path(config["recon_folder"]).mkdir(exist_ok=True, parents=True)

    print("=======")
    print("hparams")
    print("=======")
    print("cl_weight: " + str(config["cl_weight"]))
    print("num_views: " + str(config["num_views"]))
    print("plot_images_num: " + str(config["plot_images_num"]))

    main(config)
