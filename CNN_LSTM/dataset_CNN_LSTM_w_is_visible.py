from torchvision import transforms
from torch.utils.data import Dataset
from copy import deepcopy
import pandas as pd
import numpy as np
import os
from PIL import Image
from natsort import natsorted, ns
import torch

def _pad(group, seq_len):
    pad_number = seq_len - (len(group) % seq_len)
    if pad_number != seq_len:
        group = pd.concat([group, group.loc[[group.index[-1]]*(pad_number)]])
        group = group.ffill().reset_index(drop=True)
    return group

class SequentialVideoDatasetConstructor():
    def __init__(self, path, seq_len):
        self.path = path
        self.labels = pd.read_pickle(path)
        self.seq_len = seq_len

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
        self.labels['swimmer_id'] = self.labels['swimmer_id'].apply(lambda s: int(s[1:]))
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
        self.labels = self.labels.set_index(['swimmer_image_path','label_name']).reindex(self.muix).reset_index()
        # Fill NaNs with 0 in order to have numeric values (corresponds to no label)
        self.labels[['x','y']] = self.labels[['x','y']].fillna(0)
        self.labels['swimmer_id'] = self.labels.groupby('path')['swimmer_id'].apply(lambda x: x.ffill().bfill())
        # Declare labels as categorical
        self.labels['label'] = pd.Categorical(self.labels['label'], 
                      categories=["head","left_hand","right_hand","pelvis"],
                      ordered=True)
        # Sort values by image (time) and label (arbitrarily)
        self.labels = (self.labels.assign(img_time=self.labels['path'].str.split("#t=").str[1].str.split(".jpg").str[0])
                                  .sort_values(['img_time','swimmer_id','label']))
        # Keep array of values with path and labels
        self.labels['coords'] = self.labels[['x','y']].values.tolist()
        self.labels = self.labels.drop(['x','y'], axis=1)
        self.labels = (self.labels.groupby('path', sort=False)
                                 .agg({'swimmer_id':'first','img_time':'first','coords':np.vstack})
                                 .reset_index()
                                 .sort_values(['swimmer_id','img_time']))
        # Pad sequences to have a multiple of seq_len for each swimmer
        self.labels = (self.labels.groupby('swimmer_id')
                                  .apply(_pad, (self.seq_len)))

        

class SequentialSwimmersDataset(Dataset):
    def __init__(self, annotation_folder_path, seq_len, transform=None, img_size=None, normalized=False, org_images=False, construct_df=True):
        self.folder_path = annotation_folder_path
        self.video_folders = sorted(os.listdir(annotation_folder_path))
        self.seq_len = seq_len
        self.transform = transform
        self.img_size = img_size
        self.construct = construct_df
        self.annotations = pd.DataFrame()
        self.construct_annotations_df()
        self.normalized = normalized
        self.org_images = org_images

    def __len__(self):
        return len(self.annotations)//self.seq_len-1
    
    def construct_annotations_df(self):
        for video_folder in self.video_folders:
            label_path = os.path.join(self.folder_path, video_folder, "labels.csv")
            if self.construct:
                temp_df = SequentialVideoDatasetConstructor(label_path, self.seq_len)
                temp_df.keep_main_swimmer()
                temp_df.create_vector()
                temp_labels = temp_df.labels
            else:
                temp_labels = pd.read_pickle(label_path)
                # Pad sequences to have a multiple of seq_len for each swimmer
                temp_labels = (temp_labels.groupby('swimmer_id')
                                        .apply(_pad, (self.seq_len)))
            self.annotations = pd.concat([self.annotations,temp_labels], axis = 0)
        self.annotations = (self.annotations.reset_index(drop=True)
                                           .drop(["swimmer_id","img_time"], axis=1))

    def __getitem__(self, index):
        seq = deepcopy(self.annotations.iloc[self.seq_len*index:self.seq_len*(index+1)].values)
        (img_paths, coords) = (seq[:,0],seq[:,1])
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            coord = coords[i]
            # Add a column to indicate if a dot is visible or not
            is_visible = np.ones(coord.shape[0])
            arg_no_coord = np.where(np.all(coord==0,axis=1)==True)
            is_visible[arg_no_coord] = 0
            coord = np.hstack((coord,is_visible.reshape(-1,1)))
            # Open image
            org_img = Image.open(img_path).convert("RGB")
            w,h = org_img.size
            # Normalize label's coordinates by img size
            if self.img_size is not None:
                if self.normalized:
                    coord[:,:2] = coord[:,:2]/(w,h) # Normalized
                else:
                    n_h,n_w = self.img_size
                    coord[:,:2] = coord[:,:2]/(w,h)*(n_w,n_h) # Not normalized 
                

            y_coords = torch.flatten(torch.tensor(coord)).float() # Flatten outputs and convert from double to float32

            if self.transform is not None:
                tr_img = self.transform(org_img)
                seq[i,:] = [tr_img, y_coords]
            else:
                seq[i,:] = [org_img, y_coords]
        # Return original image & labels
        if self.org_images:
            return seq
        X = torch.cat(list(seq[:,0]))
        X = X.reshape(self.seq_len,int(X.size(0)/self.seq_len),X.size(1),X.size(2))
        y = torch.cat(list(seq[:,1]))
        y = y.reshape(self.seq_len,-1)
        return X,y

class ValidSequentialSwimmersDataset(Dataset):
    def __init__(self, annotation_folder_path, seq_len, transform=None, img_size=None, normalized=False, org_images=False, construct_df=True):
        self.folder_path = annotation_folder_path
        self.seq_len = seq_len
        self.transform = transform
        self.img_size = img_size
        self.construct = construct_df
        self.normalized = normalized
        self.org_images = org_images
        self.paths = natsorted(os.listdir(self.folder_path))
        # Add last element to have multiple :
        el_to_add = len(self.paths)%seq_len
        self.paths.extend([self.paths[-1]] * (seq_len-el_to_add))
        
    def __len__(self):
        return len(self.paths)//self.seq_len
    

    def __getitem__(self, index):
        img_paths = deepcopy(self.paths[self.seq_len*index:self.seq_len*(index+1)])
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            # Open image
            org_img = Image.open(self.folder_path+img_path).convert("RGB")
            if self.transform is not None:
                tr_img = self.transform(org_img)
                img_paths[i] = tr_img
            else:
                img_paths[i] = org_img
        # Return original image & labels
        if self.org_images:
            return img_paths
        X = torch.cat(list(img_paths))
        X = X.reshape(self.seq_len,int(X.size(0)/self.seq_len),X.size(1),X.size(2))
        return X

def get_train_test_size(dataset, train_percent, _print_size = True):
    train_size = int(len(dataset)*train_percent/100)
    test_size = len(dataset) - train_size
    if _print_size:
        print(f"Train size : {train_size}\nTest size : {test_size}")
    return [train_size, test_size] 

if __name__ == "__main__":
    # ds = SequentialVideoDatasetConstructor(path="./12_labels/extracted_swimmers/video_1/labels.csv", seq_len=1)
    # ds.keep_main_swimmer()
    # ds.create_vector()
    # ds = SequentialSwimmersDataset("./12_labels/extracted_swimmers/",4, construct_df=True)
    ds = SequentialSwimmersDataset("./12_labels/full_images_extracted_swimmers",4, construct_df=False)
    ds.annotations.to_csv("test2.csv")