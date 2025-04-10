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
from copy import deepcopy
from DLMI_histopathology.lora_finetune import LoraConv, LoraLinear
import torch.nn as nn
from DLMI_histopathology.staingan_training import StainNet

def normalize_img(img):
    return (img - 0.5)/0.5

def unnormalize_img(img):
    return img * 0.5 + 0.5 

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


if __name__ =="__main__":
    print("Starting feature extraction...")  
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

    dino_normalize = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize(mean=mean, std=std)
    ])

    def preprocess_stain_and_normalize(images):
        with torch.no_grad():
            images = normalize_img(images)
            stained = stainet(images.cuda())  # Apply StainNet
        stained = unnormalize_img(stained)

        normalized = dino_normalize(stained)
        return normalized

    resizer =  transforms.Resize(image_size)

    train_dataset_path = "train_stainnet_dino_features_block10.pt"
    train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, resizer, 'train')

    BATCH_SIZE=64
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    
    dino_feature_extractor_body = DinoFeatureExtractor(feature_extractor)

    all_features = []
    all_labels = []
    print(f"{len(train_dataloader)=}")
    with torch.no_grad():
        for i,(imgs, labels) in tqdm(enumerate(train_dataloader)):
            print(i)
            imgs = preprocess_stain_and_normalize(imgs)
            feats = dino_feature_extractor_body(imgs)  # Extract features from DINO
            all_features.append(feats.cpu())
            all_labels.append(labels)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    torch.save((all_features, all_labels), train_dataset_path)
