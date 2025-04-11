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
from DLMI_histopathology.train import train_model
from DLMI_histopathology.dataset import *
from DLMI_histopathology.models import HistoClassifierHead
from copy import deepcopy
import torch.nn as nn


TRAIN_IMAGES_PATH = 'train.h5'
VAL_IMAGES_PATH = 'val.h5'
TEST_IMAGES_PATH = 'test.h5'
SEED = 0
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

image_size=(98,98)
# statistics center 4
mean = [0.7997, 0.6722, 0.8196]
std = [0.1200, 0.1492, 0.0916]

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, train_transform, 'train')
val_dataset = BaselineDataset(VAL_IMAGES_PATH, train_transform, 'train')

BATCH_SIZE=64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

for name, param in feature_extractor.named_parameters():
    if  "blocks.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

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

model_save_path = 'best_model_dino_last_block2.pth'

print("Starting training")
train_model(model, train_dataloader, val_dataloader, device, optimizer_name=OPTIMIZER, optimizer_params=OPTIMIZER_PARAMS, 
                 loss_name=LOSS, metric_name=METRIC, num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=model_save_path)

print("Evaluating model")
model.load_state_dict(torch.load(model_save_path, weights_only=True))
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
solutions_data.to_csv('results_last_layer_finetune_classic.csv')

