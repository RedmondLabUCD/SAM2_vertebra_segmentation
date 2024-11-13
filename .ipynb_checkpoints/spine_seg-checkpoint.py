from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import shutil
from glob import glob
import gc
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchvision import datasets, transforms, models

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pydicom import dcmread


def main():

    sam2_checkpoint = "//trinity/home/r094879/repositories/SAM2_vertebra_segmentation/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "//trinity/home/r094879/repositories/SAM2_vertebra_segmentation/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint)

    predictor = SAM2ImagePredictor(sam2_model)

    csv_file = '//data/scratch/r094879/data/annotations/annotations.csv' 
    df = pd.read_csv(csv_file)

    image_dir = '//data/scratch/r094879/data/images'
    output_dir = '//data/scratch/r094879/data/sam_seg'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        image_name = row['image']  # Get the DICOM image name from the 'image' column
        output_file_path = os.path.join(output_dir,image_name+'.png')

        img = dcmread(os.path.join(image_dir,image_name+".dcm"))
        img_size = img.pixel_array.shape
        img_size = np.asarray(img_size).astype(float)

        image = img.pixel_array
        image = image.astype(float)
        image = (image-image.min())/(image.max()-image.min())*255.0
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert(mode='RGB')
        image = np.asarray(image)

        print(image.shape)

        x_values = row.iloc[3:29:2].values 
        y_values = row.iloc[4:29:2].values

        # Combine x and y values and filter out NaN pairs
        xy_pairs = np.array(list(zip(x_values, y_values)))
        xy_pairs = xy_pairs[~np.isnan(xy_pairs).any(axis=1)]
        labels = []

        for i in range(len(xy_pairs)):
            labels.append(1)

        print(labels)
        
        predictor.set_image(image)

        masks, scores, _ = predictor.predict(
            point_coords=xy_pairs,
            point_labels=labels,
            box=None,
            multimask_output=False,
        )

        mask_sum = masks[0,0,:,:]
        for mask in masks:
            mask_sum += mask

        plt.imshow(mask_sum, cmap='gray')
        plt.savefig(os.path.join(output_dir,image_name+".png"))
        plt.close()        
        

if __name__ == '__main__':

    main()