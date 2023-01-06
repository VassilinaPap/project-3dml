from pathlib import Path
import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

import numpy as np
import cv2

from utils import read_as_3d_array, save_voxel_grid

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir="../data/", images_dir="ShapeNetRendering/", voxels_dir="ShapeNetVox32/", split="train", num_views=4, max_views=24, augmentation_json_flag=False, augmentations_flag=False):

        # Dir names #
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.voxels_dir = voxels_dir

        self.split = split

        self.num_views = num_views
        self.max_views = max_views

        self.classes_mapping = None
        self.classes_mapping_id = {}
        self.data_ids = []

        # Augmentation json -> replace some views with views from different category #
        self.augmentation_json_flag = augmentation_json_flag
        self.augmentation_json_dict = {}

        # Enable augmentations #
        self.augmentations_flag = augmentations_flag

        # Class mapping             #
        # Class name -> "folder" id #
        with open(data_dir  + "classes.json") as classes_file:
            self.classes_mapping = json.load(classes_file)

            index_x = 0
            for key in self.classes_mapping:
                self.classes_mapping_id[self.classes_mapping[key]] = index_x
                index_x += 1

        # Save all dataset ids #
        with open(data_dir + self.split + ".txt") as ids_files:
            self.data_ids = [line.rstrip() for line in ids_files]

        # Load augmentation json file. "class_1": "class_2", when having class_1 add some samples from class_2 #
        if(augmentation_json_flag):
            with open("./augmentation.json") as json_file:
                self.augmentation_json_dict = json.load(json_file)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):

        curr_id = self.data_ids[idx]
        class_label = self.classes_mapping[curr_id.split('/')[0]]
        path_images = self.data_dir + self.images_dir + str(curr_id) +  "/rendering/"

        # When augmentation_json_flag == True #
        other_class = None
        images_other_idxs = None
        path_other_images = None

        # How many views to sample from the max_views #
        images_idxs = random.sample(range(self.max_views), self.num_views)

        if(self.augmentation_json_flag):

            # Find other class to add if exits in json #
            other_class = None
            for key in self.augmentation_json_dict:
                if(class_label == key):
                    other_class = self.augmentation_json_dict[key]
                if(class_label == self.augmentation_json_dict[key]):
                    other_class = key

            # For given sample, its class exists in the json file. Hence, add some views from a different category (defined in json) #
            if(other_class != None):

                # Reduce views from current class #
                images_idxs = random.sample(range(self.num_views), self.num_views - self.augmentation_json_dict["samples"])

                # For the other class - number of views to add #
                images_other_idxs = random.sample(range(self.num_views), self.augmentation_json_dict["samples"])

                # Scan for the other class all avail samples #
                other_samples = []
                for item in self.data_ids:
                    if(self.classes_mapping[item.split('/')[0]] == other_class):
                        other_samples.append(item)

                # Select a single sample #
                other_id = random.sample(other_samples, 1)[0]

                # Path of images #
                path_other_images = self.data_dir + self.images_dir + str(other_id) + "/rendering/"

        # Parse images #
        images = None
        for image_idx in images_idxs:
            image_idx = [str(image_idx) if image_idx > 9 else "0" + str(image_idx)]

            image = torchvision.io.read_image(path_images + image_idx[0] + ".png", mode=torchvision.io.ImageReadMode.RGB)

            #cv2.imshow("test", image.permute(1,2,0).numpy())
            #cv2.waitKey(0)

            image = image.unsqueeze(0).float() # Float -> will be normalized afterwards
            if(images == None):
                images = image
            else:
                images = torch.cat((images, image), dim=0)

        # Parse images of other class if exists #
        if(other_class != None):
            for image_idx in images_other_idxs:
                image_idx = [str(image_idx) if image_idx > 9 else "0" + str(image_idx)]

                image = torchvision.io.read_image(path_other_images + image_idx[0] + ".png", mode=torchvision.io.ImageReadMode.RGB)

                image = image.unsqueeze(0).float()
                images = torch.cat((images, image), dim=0)

        ####################################
        # images: [num_views, 3, 137, 137] #
        ####################################

        # Perform extra augmentations #
        if(self.augmentations_flag):
            aug_2 = ShapeNetDataset.get_augmentations_2()
            if(aug_2 != None):
                images = aug_2(images)

        # Normalize for image net #
        aug_1 = ShapeNetDataset.get_augmentations_1()
        images = aug_1(images)

        # Visualize #
        #if class_label == "airplane":
        #for image in images:
        #   image = ShapeNetDataset.denormalize_image(image) # Data are normalized for ImageNet undo 
        #   cv2.imshow("test", image.permute(1,2,0).numpy().astype("uint8")) # uint8!!
        #   cv2.waitKey(0)

        # Voxel #
        path_voxel = self.data_dir + self.voxels_dir + str(curr_id) +  "/model.binvox"
        voxel = read_as_3d_array(open(path_voxel, "rb")).astype(np.float32)
        voxel = torch.from_numpy(voxel)
        # [32, 32, 32] #

        # Save it and then view it with meshlab for debugging #
        #save_voxel_grid("./test.ply", voxel)

        class_label_id = int(self.classes_mapping_id[class_label])
        class_label_id = torch.as_tensor(class_label_id)

        return {'images': images,
                'voxel': voxel,
                'label': class_label_id,
                'class': class_label
                }

    @staticmethod
    def get_augmentations_1(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        transforms = T.Compose([T.Normalize(mean, std)])

        return transforms

    @staticmethod
    def get_augmentations_2(prob=0.3):
        pick = np.random.binomial(1, prob, 1)[0]

        if(pick == 1):
            transforms = T.Compose([T.GaussianBlur(kernel_size=(5,9), sigma=(4,5))]) #, T.ColorJitter(brightness=(0.0,0.0), hue=(0.0,0.0))])
        else:
            transforms = None

        return transforms

    @staticmethod
    def denormalize_image(image, mean=np.asarray([0.485, 0.456, 0.406]), std=np.asarray([0.229, 0.224, 0.225])):
        # Numpy format - for visualization #
        image_cpu = image.cpu() * std[:, None, None] + mean[:, None, None]

        return image_cpu

    @staticmethod
    def denormalize_batch(batch, mean=np.asarray([0.485, 0.456, 0.406]), std=np.asarray([0.229, 0.224, 0.225]), device="cuda"):
        # Torch format # 
        mean = torch.from_numpy(mean).to(device)
        std = torch.from_numpy(std).to(device)
        batch_tmp = batch.to(device)

        denormalized_batch = batch_tmp * std[:, None, None] + mean[:, None, None]

        return denormalized_batch

    @staticmethod
    def move_batch_to_device(batch, device):
        batch["images"] = batch["images"].to(device)
        batch["label"] = batch["label"].to(device)
        batch["voxel"] = batch["voxel"].to(device)

    @staticmethod
    def index_to_class(idx):
        mapping = {0: "airplane",
                   1: "bench",
                   2: "cabinet",
                   3: "car",
                   4: "chair",
                   5: "display",
                   6: "lamp",
                   7: "loudspeaker",
                   8: "rifle",
                   9: "sofa",
                   10: "table",
                   11: "telephone",
                   12: "watercraft"
                }

        return mapping[idx]

if __name__ == "__main__":

    #dataset = ShapeNetDataset(split="train", num_views=15, augmentation_json_flag=True, augmentations_flag=True)
    #dataset = ShapeNetDataset(split="val", num_views=15, augmentation_json_flag=True, augmentations_flag=False)
    #dataset = ShapeNetDataset(split="train", num_views=15, augmentation_json_flag=True, augmentations_flag=False)
    dataset = ShapeNetDataset(split="train", num_views=24, augmentation_json_flag=False, augmentations_flag=False)

    # Seeds #
    torch.manual_seed(15)
    random.seed(15)
    np.random.seed(15)

    train_loder = DataLoader(dataset, batch_size=15, shuffle=True, num_workers=8)

    """
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    """

    print(len(dataset))
    for data in train_loder:
        print(data["label"])
        print(data["images"][0].shape)
        print(data["images"].shape)
        print(data["voxel"].shape)
        #ShapeNetDataset.move_batch_to_device(data, "cuda")
        #ShapeNetDataset.denormalize_image(data["images"][0][0])
        #ShapeNetDataset.denormalize_batch(data["images"][0])
        break
