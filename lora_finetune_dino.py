""" Code for training a LoRA adaptator on Dino """

from pathlib import Path
import os
import h5py
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from models import HistoClassifierHead
from train import trani_model
from dataset import BaselineDataset, PrecomputedDataset,precompute
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import random
import numpy as np
import pandas as pd
base_path = Path(os.getcwd())
print(f"{base_path=}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

dino_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

# Define LoRA config (apply to self-attention layers)
lora_config = LoraConfig(
    r=8,  # Rank of LoRA update matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    target_modules=["qkv", "proj"],  # Apply LoRA to attention layers
    bias="none"
)

dino_vit_lora = get_peft_model(dino_vit, lora_config)

# add the classifier head
classif = HistoClassifierHead(dim_input=768, hidden_dim=64, dropout=0.2)
model=nn.Sequential(dino_vit_lora,classif)

# Freeze every parameter that is not Lora
for param in model.parameters():
    param.requires_grad = False  
# enable gradients computation on lora parameters
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True  # Only fine-tune LoRA layers

# Unfreeze classifier head
for name, param in model[1].named_parameters():
    param.requires_grad = True

print(f"{sum(p.numel() for p in model.parameters())}")
model.to(device)

TRAIN_IMAGES_PATH = base_path / 'train.h5'
VAL_IMAGES_PATH = base_path / 'val.h5'
TEST_IMAGES_PATH =base_path / 'test.h5'
SEED = 0

torch.random.manual_seed(SEED)
random.seed(SEED)

# Statistics on whole train set
mean = [0.7439, 0.5892, 0.7210] 
std = [0.1717, 0.2065, 0.1664]

transform = transforms.Compose([
            transforms.Resize((98, 98)),
            transforms.Normalize(mean=mean, std=std)
        ])

train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, transform, 'train')
val_dataset = BaselineDataset(VAL_IMAGES_PATH, transform, 'train')

BATCH_SIZE=64
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

OPTIMIZER = 'AdamW'
OPTIMIZER_PARAMS = {'lr': 5e-4, 'weight_decay' : 0.02}
LOSS = 'BCELoss'
METRIC = 'Accuracy'
NUM_EPOCHS = 100
PATIENCE = 15
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), **OPTIMIZER_PARAMS)
criterion = getattr(torch.nn, LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0

SAVE_PATH = base_path / 'lora_finetune_dino.pth'

train_model(model, train_dataloader, val_dataloader, device, optimizer_name=OPTIMIZER, optimizer_params=OPTIMIZER_PARAMS, 
                 loss_name=LOSS, metric_name=METRIC, num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=SAVE_PATH)


print("Evaluating model.")
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
model.eval()
model.to(device)
prediction_dict = {}

with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    test_ids = list(hdf.keys())

solutions_data = {'ID': [], 'Pred': []}
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    for test_id in tqdm(test_ids):
        img = np.array(hdf.get(test_id).get('img'))
        img = transform(torch.tensor(img)).unsqueeze(0).float()
        img = img.to(device)
        pred = model(img).detach().cpu()
        solutions_data['ID'].append(int(test_id))
        solutions_data['Pred'].append(int(pred.item() > 0.5))
solutions_data = pd.DataFrame(solutions_data).set_index('ID')
solutions_data.to_csv('results_2.csv')


