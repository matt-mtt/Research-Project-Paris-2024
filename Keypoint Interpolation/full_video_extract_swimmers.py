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
    for i,(path,sid,timestamp,coords) in enumerate(labels.values):
        if len(np.unique(coords)) == 1:
            continue
        # Read image
        img = cv.imread(path)
        x_axis, y_axis = coords[:,0], coords[:,1]
        # Remove absent keypoints
        x_axis = x_axis[x_axis!=0]
        y_axis = y_axis[y_axis!=0]
        # Define boxes bounds, and enlarge them in order to have a full swimmer on each thumbnail    
        b_xmin = min(x_axis)
        b_xmax = max(x_axis)
        b_ymin = min(y_axis)
        b_ymax = max(y_axis)
        box_w = b_xmax - b_xmin
        box_h = b_ymax - b_ymin
        b_xmin -= box_w*50/100
        b_xmax += box_w*25/100
        b_ymin -= box_w*40/100
        b_ymax += box_w*40/100
        # Crop
        imagette = img[round(b_ymin):round(b_ymax), round(b_xmin):round(b_xmax)]
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