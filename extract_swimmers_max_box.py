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
    # Load labels
    labels = pd.read_csv(f"./12_labels/centroid_labels_video_{folder_i+1}.csv")
    labels[["sid","label_name"]] = labels['label'].str.split("_",1,expand=True)
    # Get boxes sizes
    box_sizes = dict()
    for sid, swimmer_data in labels.groupby('sid'):
        max_w, max_h = 0,0
        for t, t_swimmer in swimmer_data.groupby('timeframe'):
            w = t_swimmer['x'].max() - t_swimmer['x'].min()
            h = t_swimmer['y'].max() - t_swimmer['y'].min()
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h
        # Arbitrary
        max_h = max_h*1.5
        max_w = max_w*1.2
        box_sizes[sid] = max_w,max_h
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
        for sid,swimmer in (img_labels.groupby('sid')):
            # Box size
            b_w, b_h = box_sizes[sid]
            swimmer_anchors = swimmer[swimmer['label_name'].str.contains("head") | swimmer['label_name'].str.contains("pelvis")][["x","y"]]
            x_c = swimmer_anchors['x'].sum()/2
            y_c = swimmer_anchors['y'].min()
            # x_c ,y_c = swimmer_centroid
            box_w, box_h = box_sizes[sid]
            b_xmin = round(x_c-box_w/2)
            b_xmax = round(x_c+box_w/2)
            b_ymin = round(y_c-box_h/2)
            b_ymax = round(y_c+box_h/2)
            # Crop
            imagette = img[b_ymin:b_ymax, b_xmin:b_xmax]
            # Paths
            imagette_name = f"{img_path}-{sid}"
            imagette_path = f"./12_labels/extracted_swimmers/video_{folder_i+1}/"
            Path(imagette_path).mkdir(parents=True, exist_ok=True)
            # Save
            cv.imwrite(f"{imagette_path}{imagette_name}.png", imagette)
            # Label infos
            for img_label in swimmer.values:
                label_name, x, y, img_name, img_timeframe, img_path, _, _ = img_label
                relative_x = x - b_xmin
                relative_y = y - b_ymin
                imagette_labels.append([imagette_name, imagette_path, label_name, relative_x, relative_y])
    pd.DataFrame(imagette_labels, columns=['img_name','img_path','label_name','x','y']).to_pickle(f"{imagette_path}/labels.csv")