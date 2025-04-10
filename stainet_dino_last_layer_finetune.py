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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from DLMI_histopathology.train import *
from DLMI_histopathology.dataset import *
 
from DLMI_histopathology.models import HistoClassifierHead
from copy import deepcopy
import torch.nn as nn
from DLMI_histopathology.staingan_training import StainNet
from copy import deepcopy

class DinoFeatureExtractor(nn.Module):
    def __init__(self, dino_model, split_at=11):
        super().__init__()
        self.patch_embed = dino_model.patch_embed
        self.blocks = nn.Sequential(*dino_model.blocks[:split_at])  # up to block 10
        self.norm = nn.Identity()  # skip final norm for now

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x

def normalize_img(img):
    return (img - 0.5)/0.5

def unnormalize_img(img):
    return img * 0.5 + 0.5 

class DinoFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(h5_path, "r") as f:
            self.length = f["features"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        feat = torch.tensor(self.file["features"][idx])
        label = torch.tensor(self.file["labels"][idx])
        return feat, label

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
        print(f"{len(train_dataloader)=}")
        for i,(train_x, train_y) in enumerate(train_dataloader):
            print(i)
            optimizer.zero_grad()
            train_pred = model(train_x.to(device))
            train_pred = train_pred.squeeze(1)
            loss = criterion(train_pred, train_y.to(device).to(torch.float32))
            loss.backward()
            optimizer.step()
            
            train_losses.extend([loss.item()] * len(train_y))
            train_metric = metric(train_pred.cpu(), train_y.int().cpu())
            train_metrics.extend([train_metric.item()] * len(train_y))
        
        model.eval()
        val_metrics, val_losses = [], []
        print("Validating")
        for val_x, val_y in tqdm(val_dataloader):
            with torch.no_grad():
                val_pred = model(val_x.to(device))
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


class DinoFinetuneHead(nn.Module):
    def __init__(self, dino_model):
        super().__init__()
        self.block = deepcopy(dino_model.blocks[11])  # last block
        self.norm = deepcopy(dino_model.norm)
        self.head = HistoClassifierHead(dim_input=feature_extractor.num_features, hidden_dim=64, dropout=0.2)

    def forward(self, x):
        x = self.block(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # usually the CLS token
        return self.head(cls_token)

print("test")  
print(__name__)
if __name__=="__main__":

    TRAIN_IMAGES_PATH = 'train.h5'
    VAL_IMAGES_PATH = 'val.h5'
    TEST_IMAGES_PATH = 'test.h5'
    SEED = 0
    random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Working on {device}.')
 
    image_size=(98,98)

    print("Using StainNet statisitcs on center 3")
    mean = [0.6715, 0.3569, 0.5937]
    std = [0.1269, 0.1916, 0.1157]

    dino_normalize = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=mean, std=std)
    ])

    def preprocess_stain_and_normalize(images):
        with torch.no_grad():
            stained = model_Net(images.cuda())  # Apply StainNet
        stained = stained.clamp(0, 1)  # Just to be safe

        normalized = torch.stack([dino_normalize(img) for img in stained])
        return normalized

    resizer =  transforms.Resize(image_size)

    train_dataset_path = "train_stainnet_dino_features_block10.h5"
    val_dataset_path  = "val_stainnet_dino_features_block10.h5"
    train_dataset = DinoFeatureDataset(train_dataset_path)
    val_dataset = DinoFeatureDataset(val_dataset_path)
    print("Dataset created")

    BATCH_SIZE=64
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
   


    model=DinoFinetuneHead(dino_model = feature_extractor)
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
    # print("Starting training")
    save_path_model = "dino_last_block_finetuned_on_stainet2.pth"
    # train_model(model, train_dataloader, val_dataloader, device, optimizer_name=OPTIMIZER, optimizer_params=OPTIMIZER_PARAMS, 
    #                 loss_name=LOSS, metric_name=METRIC, num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=save_path_model)

    print("Evaluating model on test data")

    model.load_state_dict(torch.load(save_path_model, weights_only=True))
    model.eval()
    model.to(device)
    prediction_dict = {}

    stainet = StainNet().cuda()
    stainet.load_state_dict(torch.load("StainNet-Public_layer3_ch32.pth"))
    stainet.eval()
    # statistics center 3
    mean = [0.6708, 0.4186, 0.6388]
    std = [0.1809, 0.2079, 0.1446]
    image_size = (98,98)
    after_stainnet_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=mean, std=std),
    ])
    dino_feature_extractor_body = DinoFeatureExtractor(feature_extractor)
    def preprocess_stain_and_normalize(images):
        with torch.no_grad():
            images = normalize_img(images)
            stained = stainet(images.cuda())  # Apply StainNet
        stained = unnormalize_img(stained)
        return after_stainnet_transform(stained)

    resizer = transforms.Resize(image_size)
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        test_ids = list(hdf.keys())
        
    solutions_data = {'ID': [], 'Pred': []}
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        for test_id in tqdm(test_ids):
            img = hdf.get(test_id).get('img')
            img = resizer(torch.tensor(img).to(device).to(torch.float32))
            img = preprocess_stain_and_normalize(img)
            feats = dino_feature_extractor_body(img.unsqueeze(0))  
            pred = model(feats).detach().cpu()
            solutions_data['ID'].append(int(test_id))
            solutions_data['Pred'].append(int(pred.item() > 0.5))
    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv('results_stainet_dino2.csv')

