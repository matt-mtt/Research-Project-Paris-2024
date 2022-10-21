from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from sklearn.preprocessing import normalize

class VideoDatasetConstructor():
    def __init__(self, path):
        self.path = path
        self.labels = pd.read_pickle(path)

    def __len__(self):
        return len(self.labels)
    
    def keep_main_swimmer(self):
        self.labels = (self.labels.assign(prefix=self.labels['label_name'].str.split('_').str[0])
                                .groupby('img_name')
                                .apply(lambda g: g[g['prefix'] == (g['prefix'].mode())[0]]) # Keep first mode by default if there are same label numbers
                                .reset_index(drop=True)
                                .drop('prefix',axis=1))

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
        self.labels[['swimmer_id','label_name']] = self.labels['label_name'].str.split('_', 1, expand=True)
        # Create a complete file path and drop previous cols
        self.labels['swimmer_image_path'] = self.labels['img_path'] + self.labels['img_name'] + '.png'
        self.labels = self.labels.drop(['img_path', 'img_name'], axis = 1)
        # Remove thumbnail duplicates so that the most labelled one is kept
        self.labels['ts'] = self.labels['swimmer_image_path'].str.split("#t=").str[1].str.split(".jpg").str[0]
        self.labels['count_grp'] = self.labels.groupby(['swimmer_image_path'])['x'].transform('count')
        self.labels = self.labels[self.labels['count_grp'] == self.labels.groupby(['swimmer_id','ts'])['count_grp'].transform('max')]
        self.labels = self.labels.drop(['ts','count_grp'], axis=1)
        # Create a multiindex for each swimmer and image to have four labels inside
        self.muix = pd.MultiIndex.from_product([self.labels['swimmer_image_path'].unique(),
                                           self.labels['label_name'].unique()], names = ('path','label'))
        # Reindex df, drop useless labels, and order labels
        self.labels = self.labels.set_index(['swimmer_image_path','label_name']).reindex(self.muix)
        # Fill NaNs with 0 in order to have numeric values (corresponds to no label)
        self.labels = self.labels.fillna(0).reset_index()
        # Declare labels as categorical
        self.labels['label'] = pd.Categorical(self.labels['label'], 
                      categories=["head","left_hand","right_hand","pelvis"],
                      ordered=True)
        # Sort values by image (time) and label (arbitrarily)
        self.labels = (self.labels.assign(img_time=self.labels['path'].str.split("#t=").str[1].str.split(".jpg").str[0])
                                  .assign(swimmer_id=self.labels['path'].str.split(".jpg-").str[1].str.split(".png").str[0])
                                  .sort_values(['img_time','swimmer_id','label'])
                                  .drop('img_time', axis=1))
        # Drop swimmer's ids NOTE: ids will be used in order to keep track of swimmers
        self.labels = self.labels.drop('swimmer_id', axis=1)
        # Keep array of values with path and labels
        self.labels['coords'] = self.labels[['x','y']].values.tolist()
        self.labels = self.labels.drop(['x','y'], axis=1)
        self.labels = self.labels.groupby('path', sort=False).coords.apply(np.vstack).reset_index()
        

class SwimmersDataset(Dataset):
    def __init__(self, annotation_folder_path, transform=None, img_size=None, normalized=True,construct_df=True):
        self.folder_path = annotation_folder_path
        self.video_folders = os.listdir(annotation_folder_path)
        self.transform = transform
        self.img_size = img_size
        self.annotations = pd.DataFrame()
        self.construct = construct_df
        self.construct_annotations_df()
        self.normalized = normalized

    def __len__(self):
        return len(self.annotations)
    
    def construct_annotations_df(self):
        for video_folder in self.video_folders:
            label_path = os.path.join(self.folder_path, video_folder, "labels.csv")
            
            if self.construct:
                temp_df = VideoDatasetConstructor(label_path)
                temp_df.keep_main_swimmer()
                temp_df.create_vector()
                temp_labels = temp_df.labels
            else:
                temp_labels = pd.read_pickle(label_path)[['path','coords']]
            self.annotations = pd.concat([self.annotations,temp_labels], axis = 0)
        self.annotations = self.annotations.reset_index(drop=True)

    def __getitem__(self, index):
        (img_path, coords) = self.annotations.iloc[index].values
        org_img = Image.open(img_path).convert("RGB")
        w,h = org_img.size
        # Normalize label's coordinates by img size
        if self.img_size is not None:
            if self.normalized:
                coords = coords/(w,h) # Normalized
            else:
                n_h,n_w = self.img_size
                coords = coords/(w,h)*(n_w,n_h) # Not normalized 
            

        y_coords = torch.flatten(torch.tensor(coords)).float() # Flatten outputs and convert from double to float32

        if self.transform is not None:
            tr_img = self.transform(org_img)
            return (tr_img, y_coords)
        return (org_img, y_coords)

def get_train_test_size(dataset, train_percent, _print_size = True):
    train_size = int(len(dataset)*train_percent/100)
    test_size = len(dataset) - train_size
    if _print_size:
        print(f"Train size : {train_size}\nTest size : {test_size}")
    return [train_size, test_size] 

if __name__ == "__main__":
    ds = VideoDatasetConstructor(path="./12_labels/extracted_swimmers/video_2/labels.csv")
    ds.keep_main_swimmer()
    ds.create_vector()
    ds.labels.to_csv("test.csv")