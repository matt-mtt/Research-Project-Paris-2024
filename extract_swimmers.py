from multiprocessing.sharedctypes import Value
import os
import glob
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

def is_inside(x,y,b_xmin,b_xmax,b_ymin,b_ymax):
    if x > b_xmin and x < b_xmax and y > b_ymin and y < b_ymax:
        return True
    return False

for folder_i, img_folders in enumerate(sorted(os.listdir('./12_labels/labeled_images/'))):
    if folder_i != 5:
        continue
    # Load labels
    labels = pd.read_csv(f"./12_labels/centroid_labels_video_{folder_i+1}.csv")
    # Timesteps indexes
    t_idx = sorted(labels['timeframe'].astype(float).unique())
    # Load images
    path = f"./12_labels/labeled_images/"+img_folders+"/"
    # Init labels list
    imagette_labels = []
    # Get file paths
    f_paths = os.listdir(path)
    f_paths.sort(key = lambda x: float(x.split('t=')[1].split('.jpg')[0]))
    for i, img_path in tqdm(enumerate(f_paths)):
        # Read image
        img = cv.imread(path+img_path)
        # Details
        new_h = img.shape[0]
        new_w = img.shape[1]
        # Load corresponding labels df
        img_labels = labels[labels['image_name'].str.contains(img_path)].copy()
        img_labels['sid'] = img_labels['label'].str.split("_").str[0]
        for sid,swimmer in (img_labels.groupby('sid')):
            # Box coords extracted from keypoints
            b_xmin = swimmer['x'].min()
            b_xmax = swimmer['x'].max()
            b_ymin = swimmer['y'].min()
            b_ymax = swimmer['y'].max()
            box_w = b_xmax - b_xmin
            box_h = b_ymax - b_ymin
            b_xmin -= box_w*50/100
            b_xmax += box_w*25/100
            b_ymin -= box_w*40/100
            b_ymax += box_w*40/100
            # Crop
            imagette = img[round(b_ymin):round(b_ymax), round(b_xmin):round(b_xmax)]
            # Paths
            imagette_name = f"{img_path}-{sid}"
            imagette_path = f"./12_labels/extracted_swimmers/video_{folder_i+1}/"
            Path(imagette_path).mkdir(parents=True, exist_ok=True)
            # Save
            cv.imwrite(f"{imagette_path}{imagette_name}.png", imagette)
            # Label infos
            for img_label in swimmer.values:
                label_name, x, y, img_name, img_timeframe, img_path, _ = img_label
                relative_x = x - b_xmin
                relative_y = y - b_ymin
                imagette_labels.append([imagette_name, imagette_path, label_name, relative_x, relative_y])
    pd.DataFrame(imagette_labels, columns=['img_name','img_path','label_name','x','y']).to_pickle(f"{imagette_path}/labels.csv")