"""
NOTE : Constructor to have keypoints coordinates in the same format as
made in the dataset_XXXX.py files. Relative coordinates of keypoints were used
We adapt the code in order to use the general coordinates.
"""
import pandas as pd
import os
import numpy as np

def _pad(group, seq_len):
    pad_number = seq_len - (len(group) % seq_len)
    if pad_number != seq_len:
        group = pd.concat([group, group.loc[[group.index[-1]]*(pad_number)]])
        group = group.ffill().reset_index(drop=True)
    return group

class Format():
    def __init__(self, general_coordinates_label_path, labeled_image_folder, seq_len):
        self.folder_path = labeled_image_folder
        self.lebel_path = general_coordinates_label_path
        self.labels = pd.read_csv(general_coordinates_label_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels)
    
    def get_video_folder_name(self):
        folders = dict()
        for folder in os.listdir(self.folder_path):
            first_file = os.listdir(os.path.join(self.folder_path,folder))[0]
            folders[first_file.split('#')[0]] = folder
        return folders

    def create_vector(self):
        """
        Creates a vector containing labels, in order to have one row per swimmer.
        For one swimmer, it looks like :
        [
            swimmer_image_path, [
                            [x_head, y_head],
                            [x_left_hand, y_left_hand],
                            [x_right_hand, y_left_hand],
                            [x_pelvis, y_pelvis]
                           ]
        ]
        """
        # Dissociate swimmer id and label name
        self.labels[['swimmer_id','label_name']] = self.labels['label'].str.split('_', 1, expand=True)
        self.labels['swimmer_id'] = self.labels['swimmer_id'].apply(lambda s: int(s[1:]))
        self.labels.drop('label', axis=1,inplace=True)
        # Create a complete file path and drop previous cols
        video_name = self.get_video_folder_name()
        self.labels['video_name'] = self.labels['video_name'].apply(lambda x: video_name[x])
        self.labels['image_path'] = "./12_labels/labeled_images/" + self.labels['video_name'] + "/" + self.labels['image_name']
        self.labels = self.labels.drop(['video_name', 'image_name'], axis = 1)
        # Create a multiindex for each swimmer and image to have four labels inside
        self.muix = pd.MultiIndex.from_product([self.labels['image_path'].unique(),
                                           self.labels['label_name'].unique(),self.labels['swimmer_id'].unique()]
                                           , names = ('path','label','swimmer_id'))
        # Reindex df, drop useless labels, and order labels
        self.labels = self.labels.set_index(['image_path','label_name','swimmer_id']).reindex(self.muix).reset_index()
        # Fill NaNs with 0 in order to have numeric values (corresponds to no label)
        self.labels[['x','y']] = self.labels[['x','y']].fillna(0)
        self.labels['swimmer_id'] = self.labels.groupby('path')['swimmer_id'].apply(lambda x: x.ffill().bfill())
        # Declare labels as categorical
        self.labels['label'] = pd.Categorical(self.labels['label'], 
                      categories=["head","left_hand","right_hand","pelvis"],
                      ordered=True)
        # Sort values by path, timeframe and labels to have them in the right order
        self.labels = (self.labels.sort_values(['path','swimmer_id','label']))
        # Keep array of values with path and labels
        self.labels['coords'] = self.labels[['x','y']].values.tolist()
        self.labels = self.labels.drop(['x','y'], axis=1)
        self.labels = (self.labels.groupby(['path','swimmer_id'], sort=False)
                                 .agg({'timeframe':'first','coords':np.vstack})
                                 .reset_index()
                                 .sort_values(['swimmer_id','timeframe']))
        # Pad sequences to have a multiple of seq_len for each swimmer
        self.labels = (self.labels.groupby('swimmer_id')
                                  .apply(_pad, (self.seq_len)))

if __name__ == "__main__":
    test = Format(general_coordinates_label_path="./12_labels/centroid_labels_video_1.csv",labeled_image_folder="./12_labels/labeled_images/",seq_len=1)
    test.create_vector()
    test.labels.to_csv('test2.csv')