from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image



def visualize_kp_on_img(img_name, video_id):
    img_path = f"./12_labels/images/video_{video_id}/{img_name}"
    labels = pd.read_csv(f"./12_labels/centroid_labels_video_{video_id}.csv")
    labels = labels[labels['image_name'] == img_name]
    plot_info = labels[['label','x','y']].values
    # Plot
    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(0,0,1,1)
    plt.axis('off')
    img = Image.open(img_path)
    im_plot = plt.imshow(img)
    c_dict = {'s1_right_hand':'blue', 's1_left_hand':'red', 's1_head':'green', 's1_pelvis':'purple', 
             's2_right_hand':'blue', 's2_left_hand':'red', 's2_head':'green', 's2_pelvis':'purple', 
             's3_right_hand':'blue', 's3_left_hand':'red', 's3_head':'green', 's3_pelvis':'purple'}
    shape_dict = {"s1":"o", "s2":"*", "s3":"D"}
    for label_name, x, y in plot_info:
        shape = shape_dict[label_name[:2]]
        plt.scatter(x,y, marker=shape, c=c_dict[label_name], s=15, label=label_name)
    plt.legend()
    plt.show()

visualize_kp_on_img("2021_Nice_freestyle_50_serie3_dames_fixeDroite_Trim.mp4#t=0.533333.jpg", 1)