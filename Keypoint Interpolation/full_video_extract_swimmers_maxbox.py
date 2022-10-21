from multiprocessing.sharedctypes import Value
import os
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm

def is_inside(x,y,b_xmin,b_xmax,b_ymin,b_ymax):
    if x > b_xmin and x < b_xmax and y > b_ymin and y < b_ymax:
        return True
    return False

folders = './12_labels/full_images_extracted_swimmers/'

for folder_i, lbl_folder in tqdm(enumerate(sorted(os.listdir('./12_labels/full_images_extracted_swimmers/')))):
    folder_path = folders+lbl_folder
    # Load labels
    labels = pd.read_pickle(folder_path+"/labels_interpolation.csv")
    # Get boxes sizes
    box_sizes = dict()
    for sid, swimmer_data in labels.groupby('swimmer_id'):
        max_w, max_h = 0,0
        for t, t_swimmer in swimmer_data.groupby('img_time'):
            coords = t_swimmer['coords'].values[0]
            x_axis, y_axis = coords[:,0], coords[:,1]
            # Remove absent keypoints
            x_axis = x_axis[x_axis!=0]
            y_axis = y_axis[y_axis!=0]
            w = x_axis.max() - x_axis.min() 
            h = y_axis.max() - y_axis.min()
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h
        # Arbitrary
        max_h = max_h*1.5
        max_w = max_w*1.2
        box_sizes[sid] = max_w,max_h

    for i,(path,sid,timestamp,coords) in enumerate(labels.values):
        if len(np.unique(coords)) == 1:
            continue
        # Read image
        img = cv.imread(path)
        x_axis, y_axis = coords[:,0], coords[:,1]
        # Remove absent keypoints
        x_axis = x_axis[x_axis!=0]
        y_axis = y_axis[y_axis!=0]
        # Box size
        b_w, b_h = box_sizes[sid]
        swimmer_anchors = coords[[0,3]]
        x_c = swimmer_anchors[:,0].sum()/2
        y_c = swimmer_anchors[:,1].min()
        # x_c ,y_c = swimmer_centroid
        box_w, box_h = box_sizes[sid]
        b_xmin = round(x_c-box_w/2)
        b_xmax = round(x_c+box_w/2)
        b_ymin = round(y_c-box_h/2)
        b_ymax = round(y_c+box_h/2)
        # Crop
        imagette = img[b_ymin:b_ymax, b_xmin:b_xmax]
        # Replace absolute values by relative values
        coords[:,0] = np.where(coords[:,0]>0, coords[:,0] - b_xmin, coords[:,0])
        coords[:,1] = np.where(coords[:,1]>0, coords[:,1] - b_ymin, coords[:,1])
        labels.at[i,'coords'] = coords
        # Paths
        imagette_p = f"{folder_path}/{timestamp}-{sid}.png"
        labels.at[i,'full_image_path'] = imagette_p
        # Save
        cv.imwrite(imagette_p, imagette)
    # Modify labels
    labels = labels.rename(columns={'full_image_path':'path'})
    labels.to_pickle(f"{folder_path}/labels.csv")