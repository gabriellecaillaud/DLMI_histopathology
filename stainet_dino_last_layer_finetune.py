import sys
import os
sys.path.append(os.getcwd())
import h5py
import torch
import random
import numpy as np
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from DLMI_histopathology.train import *
from DLMI_histopathology.dataset import *
from copy import deepcopy
from DLMI_histopathology.lora_finetune import LoraConv, LoraLinear
import torch.nn as nn
from DLMI_histopathology.staingan_training import StainNet

TRAIN_IMAGES_PATH = 'train.h5'
VAL_IMAGES_PATH = 'val.h5'
TEST_IMAGES_PATH = 'test.h5'
SEED = 0
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')
stainet = StainNet().cuda()
stainet.load_state_dict(torch.load("StainNet-Public_layer3_ch32.pth"))
stainet.eval()
image_size=(98,98)

print("Using StainNet statisitcs on center 3")
mean = [0.6715, 0.3569, 0.5937]
std = [0.1269, 0.1916, 0.1157]

train_transform = transforms.Compose([
    transforms.Resize(image_size),
])
resizer = transforms.Resize(image_size)
train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, resizer, 'train')
val_dataset = BaselineDataset(VAL_IMAGES_PATH, resizer, 'train')

BATCH_SIZE=64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

for name, param in feature_extractor.named_parameters():
    if  "blocks.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


def train_model(model, train_dataloader, val_dataloader, device, optimizer_name='Adam', optimizer_params={'lr': 0.001}, 
                 loss_name='BCELoss', metric_name='Accuracy', num_epochs=100, patience=10, save_path='best_model.pth'):
    """
    Trains a PyTorch model with early stopping.
    
    Args:
        model: PyTorch model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        device: Device to run training on ('cuda' or 'cpu').
        optimizer_name: Name of the optimizer (e.g., 'Adam').
        optimizer_params: Dictionary of optimizer parameters.
        loss_name: Name of the loss function (e.g., 'BCELoss').
        metric_name: Name of the evaluation metric (e.g., 'Accuracy').
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before stopping.
        save_path: Path to save the best model.
    """
    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_params)
    criterion = getattr(torch.nn, loss_name)()
    metric = getattr(torchmetrics, metric_name)('binary')
    
    min_loss, best_epoch = float('inf'), 0
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_metrics, train_losses = [], []
        
        for train_x, train_y in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            train_x = stainet(train_x.to(device))
            train_x = resizer(train_x)
            train_pred = model(train_x)
            # print(f"{train_pred=}")
            # print(f"{train_y=}")
            train_pred = train_pred.squeeze(1)
            loss = criterion(train_pred, train_y.to(device).to(torch.float32))
            loss.backward()
            optimizer.step()
            
            train_losses.extend([loss.item()] * len(train_y))
            train_metric = metric(train_pred.cpu(), train_y.int().cpu())
            train_metrics.extend([train_metric.item()] * len(train_y))
        
        model.eval()
        val_metrics, val_losses = [], []
        
        for val_x, val_y in tqdm(val_dataloader, leave=False):
            with torch.no_grad():
                val_x = stainet(val_x.to(device))
                val_x = resizer(val_x)
                val_pred = model(val_x)
                val_pred = val_pred.squeeze(1)
            loss = criterion(val_pred, val_y.to(device).to(torch.float32))
            
            val_losses.extend([loss.item()] * len(val_y))
            val_metric = metric(val_pred.cpu(), val_y.int().cpu())
            val_metrics.extend([val_metric.item()] * len(val_y))
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {np.mean(train_losses):.4f} | Train Metric {np.mean(train_metrics):.4f} '
              f'Val Loss: {np.mean(val_losses):.4f} Val Metric {np.mean(val_metrics):.4f}')
        
        if np.mean(val_losses) < min_loss:
            mean_val_loss = np.mean(val_losses)
            print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
            min_loss = mean_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        
        if epoch - best_epoch == patience:
            print(f"Model has not improved in val set for {patience} epochs. Stopping early.")
            break


from DLMI_histopathology.models import HistoClassifierHead

classif = HistoClassifierHead(dim_input=feature_extractor.num_features, hidden_dim=64, dropout=0.2)
model=nn.Sequential(feature_extractor,classif)
print((sum(p.numel() for p in model.parameters() if p.requires_grad )))
model.to(device)

OPTIMIZER = 'AdamW'
OPTIMIZER_PARAMS = {'lr': 5e-4, 'weight_decay' : 0.02}
LOSS = 'BCELoss'
METRIC = 'Accuracy'
NUM_EPOCHS = 100
PATIENCE = 20
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), **OPTIMIZER_PARAMS)
criterion = getattr(torch.nn, LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0
print("Starting training")
train_model(model, train_dataloader, val_dataloader, device, optimizer_name=OPTIMIZER, optimizer_params=OPTIMIZER_PARAMS, 
                 loss_name=LOSS, metric_name=METRIC, num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path='best_model_dino_last_block2.pth')

model.load_state_dict(torch.load('best_model_dino_last_block3.pth', weights_only=True))
model.eval()
model.to(device)
prediction_dict = {}

with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    test_ids = list(hdf.keys())
     

solutions_data = {'ID': [], 'Pred': []}
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    for test_id in tqdm(test_ids):
        img = np.array(hdf.get(test_id).get('img'))
        img = val_transform(torch.tensor(img)).unsqueeze(0).float()
        img = img.to(device)
        pred = model(img).detach().cpu()
        solutions_data['ID'].append(int(test_id))
        solutions_data['Pred'].append(int(pred.item() > 0.5))
solutions_data = pd.DataFrame(solutions_data).set_index('ID')
solutions_data.to_csv('results_3.csv')

