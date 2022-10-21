from array import array
from enum import unique
from PIL import Image
from pathlib import Path
import imagehash
import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=180)
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from torch import inverse
sys.path.insert(1, './CNN_LSTM/')
from formatting import Format

def pic_remove_duplicates(dirpath, hash_size, delete):
    """
    Find and Delete Duplicates
    """
    for image_dir in os.listdir(dirpath):
        print(image_dir)
        fnames = sorted(list(os.listdir(os.path.join(dirpath,image_dir))))
        hashes = {}
        duplicates = []
        print("Finding Duplicates Now!\n")
        for image in sorted(fnames):
            with Image.open(os.path.join(dirpath,image_dir,image)) as img:
                temp_hash = imagehash.phash(img, hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image,hashes[temp_hash]))
                    duplicates.append(image)
                else:
                    hashes[temp_hash] = image

        if len(duplicates) != 0:
            if delete:
                space_saved = 0
                for duplicate in duplicates:
                    space_saved += os.path.getsize(os.path.join(dirpath,image_dir,duplicate))
                    
                    os.remove(os.path.join(dirpath,image_dir,duplicate))
                    print("{} Deleted Succesfully!".format(duplicate))

                print("\n\nYou saved {} mb of Space!".format(round(space_saved/1000000),2))
        else:
            print("No Duplicates Found :(")

# Only needed one time
### pic_remove_duplicates("./12_labels/full_images/", hash_size=20,delete=True)

def get_corresponding_frames(labeldir, labeled_folder, hash_size):
    labeled_images_path = os.listdir(os.path.join(labeldir, labeled_folder))
    matches = dict()
    for labeled_image_path in labeled_images_path:
        labeled_image_path = os.path.join(labeldir, labeled_folder,labeled_image_path)
        matches[labeled_image_path] = None
    # Get the labelled images dirname by taking the name of the first image
    full_images_dirpath = os.path.join("./12_labels/full_images/",labeled_images_path[0].split("#")[0])
    fi_hashes = dict()
    for full_image in os.listdir(full_images_dirpath):
        fi_fullpath = os.path.join(full_images_dirpath,full_image)
        with Image.open(fi_fullpath) as img:
            fi_hashes[fi_fullpath] = imagehash.phash(img, hash_size)
    # Now iterate over labeled images and find their matches
    for l_img_path in matches:
        with Image.open(l_img_path) as l_img:
            l_hash = imagehash.phash(l_img, hash_size)
        hash_dist = 0 # e.g. hashes are exactly the same : 100% similarity
        while matches[l_img_path] is None:
            for fi_path, fi_hash in fi_hashes.items():
                if (l_hash - fi_hash) == hash_dist:
                    matches[l_img_path] = fi_path
            # If no matches are found, increase tolerance
            hash_dist += 1
    
    matches = pd.DataFrame(matches.items(), columns=['labeled_path','full_image_path'])
    fi_paths = pd.DataFrame(fi_hashes.keys(), columns=['full_image_path'])
    fi_matches = fi_paths.merge(right = matches, how = "left", right_on="full_image_path", left_on="full_image_path")
    # Sort dataframe in temporal order
    fi_matches = fi_matches.sort_values(by="full_image_path", key=lambda col: col.map(lambda s: float(s.split(".mp4/")[1].split(".jpg")[0])))
    # Get labels of corresponding video 
    label_path = f"./12_labels/centroid_labels_{labeled_folder}.csv"
    vc = Format(general_coordinates_label_path=label_path,
                labeled_image_folder="./12_labels/labeled_images/",seq_len=1)
    vc.create_vector()
    vc.labels['path'] = vc.labels['path'].str.replace("extracted_swimmers","labeled_images").str.split(".jpg").str[0]+".jpg"
    fi_matches.to_csv('matches.csv')
    # Merge to associate labels to full frames
    fi_matches = fi_matches.merge(right = vc.labels, how = "outer", right_on="path", left_on="labeled_path").drop(["path","labeled_path","timeframe"], axis=1)
    fi_matches["img_time"] = fi_matches["full_image_path"].apply(lambda s: float(s.split("/")[-1].split(".jpg")[0]))
    # Duplicate NaN rows in order to have one for each swimmer
    fi_matches['swimmer_id'] = fi_matches['swimmer_id'].fillna(1)
    repeated_rows = pd.DataFrame(np.repeat(fi_matches[fi_matches["coords"].isna()].values, 2, axis=0), columns=fi_matches.columns)
    fi_matches = pd.concat([fi_matches,repeated_rows])
    # Numerotate each unlabeled frame for each swimmer id
    fi_matches.loc[fi_matches['coords'].isna(),'swimmer_id'] = fi_matches[fi_matches['coords'].isna()].groupby('full_image_path').cumcount()+1
    # Fill images with missing swimmers
    # Get number of unique images and swimmers
    unique_images = pd.DataFrame(fi_matches['full_image_path'].unique(), columns=["full_image_path"])
    unique_swimmers = pd.DataFrame(fi_matches['swimmer_id'].unique(), columns=["swimmer_id"])
    # Create carthesian product
    product = unique_images.merge(unique_swimmers, how = "cross")
    # Outer merge dataframe on the product 
    fi_matches = fi_matches.merge(product, how = "right", on=['full_image_path',"swimmer_id"])
    # Replace NaN coords by zeroed coordinates
    null_coords = fi_matches['coords'].isnull()
    fi_matches.loc[null_coords, 'coords'] = [[[0,0],[0,0],[0,0],[0,0]]]*(null_coords.sum())
    # Duplicate image time NaNs
    fi_matches['img_time'] = fi_matches.groupby('full_image_path')['img_time'].apply(lambda x: x.bfill().ffill())
    return fi_matches

def compute_swimmer_periods(values, threshold):
    vdf = pd.DataFrame(values.reshape(-1,1), columns=['values'])
    # Boolean array of values different of 0
    m1 = vdf['values'].ne(0)
    # Boolean array where sequence of inverse condition of m1
    # superior to Threshold are True, rest are False
    m2 = vdf.groupby(m1.cumsum())['values'].transform('count').gt(threshold)
    # sequences are 
    zero_periods = (m1&m2).cumsum().where((~m1)&m2)
    zero_periods = np.where(zero_periods>0,1,0)
    # Periods
    m3 = vdf.groupby(zero_periods.cumsum())['values'].transform('count').gt(2)
    periods = (zero_periods&m3).cumsum().where(~zero_periods&m3)

    return periods


def interpolate(data):
    """ 
    Coordinates reminder :
    [
        [x_head, y_head],
        [x_left_hand, y_left_hand],
        [x_right_hand, y_left_hand],
        [x_pelvis, y_pelvis]
    ]
    """
    data['org_index'] = data.index
    data = data.sort_values(['img_time','swimmer_id'])
    new_coords_arr = np.zeros((len(data),4,2))
    for sid, group in data.groupby(['swimmer_id']):
        print(sid)
        # To be sure to convert all potential lists into arrays
        arr_values = np.array(group['coords'].values.tolist())
        if len(np.unique(arr_values)) == 1:
            continue
        x_time = group['img_time'].values
        """
        arr_values is an array containing all of the correponding
        swimmer's coordinates.
        arr_values[x,y,z] :
            x : frame_number, we perform a regression on all the frames
            so we take every frame
            y : the row of each coordinate array, 0 is head, 1 is left
            hand... refer to previous notes
            z : the column of each coordinate array, 0 is the x axis
            while 1 is the y axis
        """
        # Beginning with Head and Pelvis : interpolation is made on
        # the whole swimmer sequence
        for label_id in [0,3]: # Head, Pelvis
            for axis in [0,1]: # x, y
                label_values = arr_values[:,label_id, axis]
                stacked_coords = np.vstack((x_time,label_values)).T
                # Remove 0 values for fitting i.e. non labelled frames
                non_zero_ind = np.where(label_values != 0)[0]
                stacked_coords = stacked_coords[non_zero_ind]
                # Fit with arbitrarily 6 degrees of freeom
                reg = np.poly1d(np.polyfit(stacked_coords[:,0], stacked_coords[:,1], 6))
                # Predict on time values
                preds = reg(x_time)
                # Merge interpolation with real labels to have full predictions on one label/axis
                full_label_coords = np.where(label_values == 0, preds, label_values)
                # Fill the final array column
                new_coords_arr[group['org_index'],label_id,axis] = full_label_coords
                # print(r2_score(stacked_coords[:,1], mymodel(stacked_coords[:,0])))
                # plt.plot(stacked_coords[:,0],stacked_coords[:,1])
                # plt.plot(stacked_coords[:,0], mymodel(stacked_coords[:,0]))
                # plt.show()

        # For hands, "swimming periods" are isolated and interpolations
        # are made on those periods.
        for label_id in [1,2]: # Hands
            for axis in [0,1]: # x,y
                label_values = arr_values[:,label_id, axis]
                periods = compute_swimmer_periods(label_values, threshold=4)
                unique_periods = periods.unique()
                stacked_coords = np.vstack((x_time,label_values,periods)).T
                for period_id in unique_periods:
                    if np.isnan(period_id) or period_id <= 0:
                        continue
                    period_coords = stacked_coords[periods.eq(period_id)]
                    period_time, period_values = period_coords[:,0], period_coords[:,1]
                    # Remove 0 values for fitting i.e. non labelled frames
                    non_zero_ind = np.where(period_coords[:,1] != 0)[0]
                    nonzero_period_coords = period_coords[non_zero_ind]
                    # Skip empty coords
                    if len(nonzero_period_coords) == 1:
                        continue
                    # Fit with arbitrarily 3 degrees of freedom
                    reg = np.poly1d(np.polyfit(nonzero_period_coords[:,0], nonzero_period_coords[:,1], 3))
                    # Predict on time values
                    preds = reg(period_time)
                    # Merge interpolation with real labels to have full predictions on one label/axis
                    full_label_period_values = np.where(period_values == 0, preds, period_values)
                    # Overwrite with inferred values
                    stacked_coords[periods.eq(period_id),1] = full_label_period_values
                    # print(r2_score(nonzero_period_coords[:,1], reg(nonzero_period_coords[:,0])))
                    # plt.plot(nonzero_period_coords[:,0], nonzero_period_coords[:,1])
                    # plt.plot(nonzero_period_coords[:,0], reg(nonzero_period_coords[:,0]))
                    # print(nonzero_period_coords)
                    # plt.show()
                # Fill the final array column
                new_coords_arr[group['org_index'],label_id,axis] = stacked_coords[:,1]

    data.drop('coords', axis=1, inplace=True)
    data = data.merge(pd.Series(list(new_coords_arr),name="coords"), right_index=True, left_on="org_index", how="left")
    data.drop('org_index', axis=1, inplace=True)
    data = data.reset_index(drop=True)
    return data

if __name__ == "__main__":
    labeldir = "./12_labels/labeled_images/"
    saved_labeldir = "./12_labels/full_images_extracted_swimmers/"
    for labeled_folder in os.listdir(labeldir):
        print(labeled_folder)
        processed_labels = get_corresponding_frames(labeldir,labeled_folder, 30)
        print("Interpolation ...\n")
        full_labels = interpolate(processed_labels)
        video_labeldir = os.path.join(saved_labeldir,labeled_folder)
        Path(video_labeldir).mkdir(parents=True, exist_ok=True)
        # Save
        full_labels.to_pickle(video_labeldir+"/labels_interpolation.csv")
        print(f"Saved at {video_labeldir}")
    