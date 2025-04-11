""" In this file: code for pre-processing images using StainGAN, compute Dino features and a classifier head."""

import torch

import functools
import torch.nn as nn
from torchvision.models import squeezenet1_1
from tqdm import tqdm
from pathlib import Path
import os
import random
import torchvision.transforms as transforms
from models import HistoClassifierHead, ResnetGenerator
from dataset import BaselineDataset, PrecomputedDataset,precompute
from train import train_model
from torch.utils.data import Dataset, DataLoader
base_path = Path(os.getcwd())
print(f"{base_path=}")
import numpy as np
from pathlib import Path

def normalize_img(img):
    return (img - 0.5)/0.5

def unnormalize_img(img):
    return img * 0.5 + 0.5

def precompute_StainGan(dataloader, model,feature_extractor, device):
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            x = normalize_img(x)
            res = model(x.to(device))
            res = unnormalize_img(res)
            # shape
            res = transform(res)
            res = feature_extractor(res).cpu().numpy()
            xs.append(res)
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)


if __name__=="__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Working on {device}.')

    TRAIN_IMAGES_PATH = base_path / 'train.h5'
    VAL_IMAGES_PATH = base_path / 'val.h5'
    TEST_IMAGES_PATH =base_path / 'test.h5'
    SEED = 0

    torch.random.manual_seed(SEED)
    random.seed(SEED)

    # Create the normalization transform
    transform = transforms.Compose([
                transforms.Resize((98, 98)),
            ])

    train_dataset_path = 'train_dataset_staingan_dino.pt'
    val_dataset_path = "val_dataset_staingan_dino.pt"

    if Path(train_dataset_path).exists():
        train_dataset_dict = torch.load(train_dataset_path)
        train_dataset = PrecomputedDataset(features = train_dataset_dict['features'], labels=train_dataset_dict['labels'].squeeze(1))
        val_dataset_dict = torch.load(val_dataset_path)
        val_dataset = PrecomputedDataset(features = val_dataset_dict['features'], labels=val_dataset_dict['labels'].squeeze(1))
    else:
        model_GAN = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).cuda()
        model_GAN.load_state_dict(torch.load('latest_net_G_A.pth'))

        train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, transform, 'train')
        val_dataset = BaselineDataset(VAL_IMAGES_PATH, transform, 'train')

        BATCH_SIZE=64
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        feature_extractor.eval()
        train_dataset = PrecomputedDataset(*precompute_StainGan(train_dataloader, model_GAN, feature_extractor,device))
        val_dataset = PrecomputedDataset(*precompute_StainGan(val_dataloader, model_GAN,feature_extractor, device))

        torch.save({
            'features': train_dataset.features,
            'labels': train_dataset.labels
        }, train_dataset_path)

        # Save validation dataset
        torch.save({
            'features': val_dataset.features,
            'labels': val_dataset.labels
        }, val_dataset_path)

    BATCH_SIZE=64
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
    model = HistoClassifierHead(dim_input=768, hidden_dim=128, dropout=0.1)
    # Training
    model_path = "classifier_dino_staingan.pth"
    train_model(model, train_dataloader, val_dataloader, device, optimizer_name='Adam', optimizer_params={'lr': 0.001}, 
                    loss_name='BCELoss', metric_name='Accuracy', num_epochs=100, patience=10, save_path=model_path)

    model.load_state_dict(torch.load(model_path))
    
