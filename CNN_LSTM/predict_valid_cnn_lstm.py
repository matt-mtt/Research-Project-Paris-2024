import sys
import numpy as np
import torch
import copy
import shutil
torch.set_printoptions(threshold=sys.maxsize)
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from dataset_CNN_LSTM import ValidSequentialSwimmersDataset
from model import CNNLSTM

device = ("cuda" if torch.cuda.is_available() else "cpu")
valid_folder = "./data/valid/"
shutil.rmtree(valid_folder+"/extracted_images")
# Get video images into list
video_path = valid_folder + "Train_nageur_extrait_davinci.mov" 

Path(valid_folder + "extracted_images").mkdir(parents=True, exist_ok=True)
vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(valid_folder + "extracted_images/frame%d.png" % count, image)     # save frame as PNG file      
  success,image = vidcap.read()
  count += 1
print("Done")

img_size = (200,100)
model_path = "./CNN_LSTM/saved_models/model_CNNLSTM.pkl"
model = CNNLSTM(numChannels=3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

video_flow = cv2.VideoWriter("./test.avi", cv2.VideoWriter_fourcc(*'XVID'), 25, (640,480))

# Construct dataloader
transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
transformed_dataset = ValidSequentialSwimmersDataset(annotation_folder_path=valid_folder + "extracted_images/", seq_len=40, 
                                                transform=transform, img_size=img_size, normalized=False, construct_df=False)
loader = DataLoader(dataset=transformed_dataset, shuffle=False, batch_size=4,num_workers=0,pin_memory=False,drop_last=True)
org_dataset = ValidSequentialSwimmersDataset(annotation_folder_path=valid_folder + "extracted_images/", seq_len=40, 
                                        org_images=True, construct_df=False)



with torch.no_grad():
    for i, x in enumerate(loader):
        x = x.to(device=device)
        out = model(x)
        # Detach to cpu  
        preds = out.clone().detach().to(device).cpu()
        x = x.cpu()
        for i_b, pred_seq in enumerate(preds): # iterate over batch samples
            pred_seq = copy.deepcopy(pred_seq)

            for i_seq, pred_coord in enumerate(pred_seq):
                image = org_dataset[i_b][i_seq]
                fig = plt.figure()
                im = plt.imshow(image)
                w_img, h_img = image.size
                for i in range(len(pred_coord)):
                    if i%2 == 0:
                        pred_coord[i] *= w_img
                    else:
                        pred_coord[i] *= h_img

                x_head, y_head, x_left_hand, y_left_hand, \
                x_right_hand, y_right_hand, x_pelvis, y_pelvis = pred_coord

                # Display image
                plt.imshow(image)
                # Display predicted points
                plt.scatter(x_head,y_head,c="green",label="Head")
                plt.scatter(x_left_hand,y_left_hand,c="red",label="Left hand")
                plt.scatter(x_right_hand,y_right_hand,c="blue",label="Right hand")
                plt.scatter(x_pelvis,y_pelvis,c="purple",label="Pelvis")
                plt.axis('off')
                fig.canvas.draw()
                # convert canvas to image
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # img is rgb, convert to opencv's default bgr
                img_scatter = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imwrite("aaaaaaa.jpg",img_scatter)
                video_flow.write(img_scatter)
                fig.clear()
                plt.close()
video_flow.release()