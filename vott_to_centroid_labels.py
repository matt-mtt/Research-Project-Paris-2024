import pandas as pd



# label_file_path = "./12_labels/VoTT Exports/Budapest 2021/vott-csv-export/Swimmer-skeleton-tracking-12-labels-export.csv"
# label_file_path = "/home/admin/Desktop/Swimmers skeleton reconsitution/12_labels/labeled_images/video_4/Swimmer-skeleton-tracking-12-labels-export.csv"
# label_file_path = "./12_labels/VoTT Exports/2_Nice_Videos/vott_labels/vott-export/12_labels_2_videos-export.csv"
label_file_path = "./label-12-export.csv"

labels = pd.read_csv(label_file_path)
labels['x'] = (labels['xmin']+labels['xmax'])/2
labels['y'] = (labels['ymin']+labels['ymax'])/2

labels['video_name'] = labels['image'].apply(lambda x: x.split("#t=")[0])
labels['timeframe'] = labels['image'].apply(lambda x: x.split("#t=")[1].split(".jpg")[0])
labels['image_name'] = labels['image'] + ".jpg"
labels.drop(["xmin","xmax","ymin","ymax","image"], axis = 1, inplace = True)

labels['timeframe'] = labels['timeframe'].astype(float)

for i,v in enumerate(sorted(labels['video_name'].unique())):
    print(sorted(labels['timeframe'].astype(float).unique()))
    sorted_vid_df = labels[labels["video_name"] == v].sort_values(by='timeframe')
    sorted_vid_df.to_csv(f'./12_labels/centroid_labels_video_6.csv', index = False)
