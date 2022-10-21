import cv2
import os

video_root = "./12_labels/videos/"

for video_path in sorted(os.listdir(video_root), reverse = True):
    vidcap = cv2.VideoCapture(video_root+video_path)
    success,image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    folder_path = f"./12_labels/full_images/{video_path}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    count = 0
    while success:
        cv2.imwrite(f"{folder_path}/{count*100/fps}.jpg", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    print(f"Ok,{count} frames")