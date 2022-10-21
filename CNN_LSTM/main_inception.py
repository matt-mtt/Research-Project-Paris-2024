import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/Inception")
from model import CNN
from utils import plot_predictions
import utils as u
from dataset_CNN import SwimmersDataset, get_train_test_size
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
device = ("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
train_percent = 85
learning_rate = 0.00001
train_CNN = False
batch_size = 32
shuffle = False
pin_memory = True
normalize = True
num_workers = 0
img_size = (100,200)
seed = 42

transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
# Create dataset with transformed images for training purposes
transformed_dataset = SwimmersDataset(annotation_folder_path="./12_labels/extracted_swimmers", transform=transform, img_size=img_size, normalized=normalize)
train_set, validation_set = torch.utils.data.random_split(transformed_dataset, get_train_test_size(transformed_dataset,train_percent), generator=torch.Generator().manual_seed(seed))
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

# Create the same dataset with untransformed images for visualization purposes
org_dataset = SwimmersDataset(annotation_folder_path="./12_labels/extracted_swimmers", transform=None, img_size=None, normalized=False)
viz_train_set, viz_validation_set = torch.utils.data.random_split(org_dataset, get_train_test_size(org_dataset,train_percent,_print_size=False), generator=torch.Generator().manual_seed(seed))

# # Create dataset with transformed images for training purposes
# transformed_dataset = SwimmersDataset(annotation_folder_path="./12_labels/full_images_extracted_swimmers/", 
#                                                 transform=transform, img_size=img_size, normalized=normalize, construct_df=False)
# train_set, validation_set = torch.utils.data.random_split(transformed_dataset, get_train_test_size(transformed_dataset,train_percent), generator=torch.Generator().manual_seed(seed))
# train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
# validation_loader = DataLoader(dataset=validation_set, shuffle=False, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

# # Create the same dataset with untransformed images for visualization purposes
# org_dataset = SwimmersDataset(annotation_folder_path="./12_labels/full_images_extracted_swimmers/", 
#                                     construct_df=False)
# viz_train_set, viz_validation_set = torch.utils.data.random_split(org_dataset, get_train_test_size(org_dataset,train_percent,_print_size=False), generator=torch.Generator().manual_seed(seed))

model = CNN().to(device)

reg_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for name, param in model.inception.named_parameters():
    if "fc.weight" in name or "fc.bias" in name:
        param.requires_grad = True
    else:
        param.requires_grad = train_CNN

def compute_metric(loader, viz_set, model, metric, plot = False, n_plots =20):
    score = 0
    if plot:
        num_plots = 0
        fig, axs = plt.subplots(4, n_plots//4, figsize=(25,10))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            out = model(x)
            # Detach to cpu  
            preds = out.clone().detach().to(device).cpu()
            x = x.cpu()
            y = y.cpu()
            # Plot ?
            if plot:
                num_plots = u.plot_predictions(viz_set, preds,axes=axs,num_plots=num_plots,max_plots=n_plots,indice_list=range(i*batch_size,(i+1)*batch_size))
            # Compute metric
            score += metric(preds, y) 
        score /= len(loader)
    model.train()
    return score

def plot_losses(train, val):
    x_train_l, y_train_l = np.array(train).T
    x_val_l, y_val_l = np.array(val).T
    plt.figure(figsize=(20,10))
    plt.plot(x_train_l, y_train_l, label="Train loss")
    plt.plot(x_val_l, y_val_l, label="Val metric")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def train():
    do_plot = False
    train_losses = []
    val_metrics = [] 
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total = len(train_loader), leave = True)
        if epoch > 97:
            do_plot = True
        if epoch%2 == 0 and epoch > 0:
            val_mse = compute_metric(validation_loader, viz_validation_set, model, mean_squared_error, do_plot)
            val_metrics.append([epoch, val_mse])
            print(val_mse)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = reg_criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())
        train_losses.append([epoch,loss.item()])
    plot_losses(train_losses, val_metrics)

if __name__ == "__main__":
    train()