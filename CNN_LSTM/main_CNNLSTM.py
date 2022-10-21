import sys
import torch
torch.set_printoptions(threshold=sys.maxsize)
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/Inception")
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import copy

from model import CNNLSTM
from dataset_CNN_LSTM import SequentialSwimmersDataset, get_train_test_size
import utils as u

device = ("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
sequence_length = 40
train_percent = 70
learning_rate = 0.0001
batch_size = 4 #Batch of sequences
pin_memory = True
normalize = True
num_workers = 0
img_size = (200,100)
seed = 42

# num_epochs = 1500
# sequence_length = 8
# train_percent = 85
# learning_rate = 0.0001
# batch_size = 8 #Batch of sequences
# pin_memory = True
# normalize = True
# num_workers = 0
# img_size = (200,100)
# seed = 42

transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

# # Create dataset with transformed images for training purposes
# transformed_dataset = SequentialSwimmersDataset(annotation_folder_path="./12_labels/extracted_swimmers/", seq_len=sequence_length, 
#                                                 transform=transform, img_size=img_size, normalized=normalize, construct_df=True)
# train_set, validation_set = random_split(transformed_dataset, get_train_test_size(transformed_dataset,train_percent), generator=torch.Generator().manual_seed(seed))
# train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
# validation_loader = DataLoader(dataset=validation_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

# # Create the same dataset with untransformed images for visualization purposes
# org_dataset = SequentialSwimmersDataset(annotation_folder_path="./12_labels/extracted_swimmers/", seq_len=sequence_length, 
#                                         org_images=True, construct_df=True)
# viz_train_set, viz_validation_set = random_split(org_dataset, get_train_test_size(org_dataset,train_percent,_print_size=False), generator=torch.Generator().manual_seed(seed))

# Create dataset with transformed images for training purposes
transformed_dataset = SequentialSwimmersDataset(annotation_folder_path="./12_labels/full_images_extracted_swimmers/", seq_len=sequence_length, 
                                                transform=transform, img_size=img_size, normalized=normalize, construct_df=False)
train_set, validation_set = random_split(transformed_dataset, get_train_test_size(transformed_dataset,train_percent), generator=torch.Generator().manual_seed(seed))
train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

# Create the same dataset with untransformed images for visualization purposes
org_dataset = SequentialSwimmersDataset(annotation_folder_path="./12_labels/full_images_extracted_swimmers/", seq_len=sequence_length, 
                                        org_images=True, construct_df=False)
viz_train_set, viz_validation_set = random_split(org_dataset, get_train_test_size(org_dataset,train_percent,_print_size=False), generator=torch.Generator().manual_seed(seed))

model = CNNLSTM(numChannels=3).to(device)
reg_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def compute_metric(loader, viz_set, model, metric, plot = False, n_seq_plots = 4):
    score = 0
    c = 0
    if plot:
        num_plots = 0
        fig, axs = plt.subplots(n_seq_plots, sequence_length, figsize=(25,10))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            out = model(x)
            # Detach to cpu  
            preds = out.clone().detach().to(device).cpu()
            y = y.cpu()
            x = x.cpu()
            # Plot ?
            if plot:
                num_plots = u.plot_sequential_predictions(viz_set, preds,axes=axs,num_plots=num_plots,
                            max_seq_plots=n_seq_plots*sequence_length,
                            indice_list=range(i*batch_size,(i+1)*batch_size))
            # Compute metric for every image of the sequence
            for i_b in range(preds.shape[0]):
                for i_s in range(sequence_length):
                    t_preds = preds[i_b,i_s,:]
                    t_y = y[i_b,i_s,:]
                    # Mask to ignore zeroed values in val dataset
                    mask = t_y!=0
                    t_y = t_y[mask]
                    t_preds = t_preds[mask]
                    score += metric(t_preds, t_y)
                    c += 1
        score /= c
    model.train()

    if plot:
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        fig.show()
        plt.savefig('plots_resultats.png', dpi=600)
    return score


def train():
    best_val_mse = float('inf')
    train_losses = []
    val_metrics = [] 
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total = len(train_loader), leave = True) # Loop is tqdmed loader
        if epoch%2 == 0 and epoch > 0:
            val_mse = compute_metric(validation_loader, viz_validation_set, model, mean_squared_error)
            val_metrics.append([epoch, val_mse])
            print(val_mse)
            if best_val_mse > val_mse:
                best_model = copy.copy(model)

        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            # Remove zero values in order to keep only real keypoints
            mask = labels!=0
            labels = labels[mask]
            out = out[mask]
            loss = reg_criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())
        train_losses.append([epoch,loss.item()])
    # Plot res with best model
    compute_metric(validation_loader, viz_validation_set, best_model, mean_squared_error, plot=True)
    u.plot_losses(train_losses, val_metrics)
    torch.save(best_model.state_dict(), './CNN_LSTM/saved_models/model_CNNLSTM.pkl')


if __name__ == "__main__":
    train()