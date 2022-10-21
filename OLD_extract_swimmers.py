from multiprocessing.sharedctypes import Value
import os
import glob
from pathlib import Path
import cv2 as cv
import json
import pandas as pd
from tqdm import tqdm

def is_inside(x,y,b_xmin,b_xmax,b_ymin,b_ymax):
    if x > b_xmin and x < b_xmax and y > b_ymin and y < b_ymax:
        return True
    return False

video_names = glob.glob("./12_labels/video_bboxes_swimmers/*.avi")

for folder_i, img_folders in enumerate(sorted(os.listdir('./12_labels/labeled_images/'))):
    v_res = int(video_names[folder_i].split('/')[-1].split('_')[3])
    # Load labels
    labels = pd.read_csv(f"./12_labels/centroid_labels_video_{folder_i+1}.csv")
    # Timesteps indexes
    t_idx = sorted(labels['timeframe'].astype(float).unique())
    # Load boxes
    with open(f"./12_labels/video_bboxes_swimmers/boxes_video_{folder_i+1}.json", 'r') as f:
        boxes = json.load(f)
    # Load images
    path = f"./12_labels/labeled_images/"+img_folders+"/"
    # Init labels list
    imagette_labels = []
    # Get file paths
    f_paths = os.listdir(path)
    f_paths.sort(key = lambda x: float(x.split('t=')[1].split('.jpg')[0]))
    for i, img_path in tqdm(enumerate(f_paths)):
        # Reset swimmer counter
        swim_count = 1
        # Read image
        img = cv.imread(path+img_path)
        # Details
        new_h = img.shape[0]
        new_w = img.shape[1]
        # Load corresponding labels df
        img_labels = labels[labels['image_name'].str.contains(img_path)]
        # Get associated boxes
        img_boxes = boxes[img_path]
        for box in img_boxes:
            # Marker to avoid saving the same imagette twice
            already_saved = False
            b_xmin, b_ymin, b_xmax, b_ymax = box.values()
            # Resize coordinates to extracted img
            b_xmin = b_xmin/v_res*new_w
            b_xmax = b_xmax/v_res*new_w
            b_ymin = b_ymin/v_res*new_h
            b_ymax = b_ymax/v_res*new_h
            if labels is None:
                continue
            for img_label in img_labels.values:
                # Read infos from label
                label_name, x, y, img_name, img_timeframe, img_path = img_label
                if is_inside(x,y,b_xmin,b_xmax,b_ymin,b_ymax):
                    imagette = img[round(b_ymin):round(b_ymax), round(b_xmin):round(b_xmax)]
                    imagette_name = f"{img_path}-{swim_count}"
                    imagette_path = f"./12_labels/extracted_swimmers/video_{folder_i+1}/"
                    Path(imagette_path).mkdir(parents=True, exist_ok=True)
                    relative_x = x - b_xmin
                    relative_y = y - b_ymin
                    imagette_labels.append([imagette_name, imagette_path, label_name, relative_x, relative_y])
                    
                    # Save imagette
                    if not already_saved:
                        cv.imwrite(f"{imagette_path}{imagette_name}.png", imagette)
                        already_saved = True
            swim_count += 1
    pd.DataFrame(imagette_labels, columns=['img_name','img_path','label_name','x','y']).to_csv(f"{imagette_path}/labels.csv", index=False)