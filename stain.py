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

import torch
from torchvision.transforms import ToTensor
from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.augmentor import AugmentorBuilder
from dataset import BaselineDataset
import torch.nn as nn

# Import data
TRAIN_IMAGES_PATH = 'train.h5'
VAL_IMAGES_PATH = 'val.h5'
TEST_IMAGES_PATH = 'test.h5'
SEED = 0
torch.random.manual_seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

# Compute statistics on training dataset based on hospital center
centers={}
with h5py.File(TRAIN_IMAGES_PATH, 'r') as hdf:    
    idx_train=list(hdf.keys())
    for indexes in idx_train:
        centers_nb=int(hdf.get(indexes).get("metadata")[0])
        if centers_nb not in centers:
            centers[centers_nb]=1
        else:
            centers[centers_nb]+=1


class BaselineDatasetCenter(Dataset):
    """ 
    Create a Dataset containing only data from center i
    """
    def __init__(self, dataset_path, preprocessing, mode,center):
        super(BaselineDatasetCenter, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        self.image_ids=[]
        with h5py.File(self.dataset_path, 'r') as hdf:        
            indx = np.array(list(hdf.keys()))
            for indexes in indx:
                centers_nb=int(hdf.get(indexes).get("metadata")[0])
                if centers_nb==center:
                    self.image_ids.append(indexes)


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(np.array(hdf.get(img_id).get('img')))
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
        return self.preprocessing(img).float(), label


preprocessing = transforms.Resize((98, 98))
train_set_3=BaselineDatasetCenter(TRAIN_IMAGES_PATH,preprocessing,"train",3)
train_set_4=BaselineDatasetCenter(TRAIN_IMAGES_PATH,preprocessing,"train",4)
train_set_0=BaselineDatasetCenter(TRAIN_IMAGES_PATH,preprocessing,"train",0)


train_loader_3=DataLoader(train_set_3,shuffle=False,batch_size=256)
train_loader_4=DataLoader(train_set_4,shuffle=False,batch_size=256)
train_loader_0=DataLoader(train_set_0,shuffle=False,batch_size=256)


def compute_stats(loader):
    """ 
    Compute the mean and std of the center i
    """
    mean=torch.zeros(3).to(device)
    std=torch.zeros(3).to(device)
    for data,lab in loader:
        mean+=torch.mean(data.to(device),dim=[0,2,3]).to(device)
        std+=torch.std(data.to(device),dim=[0,2,3]).to(device)   
    return mean/len(loader), std/len(loader)


# The center 4 has the most data, we adapt the distibution of the other centers to this one
mean_center_4,std_center_4=compute_stats(train_loader_4)

normalizer_macenko = NormalizerBuilder.build('macenko',concentration_method='ista').to(device)

def find_best_image(dataset):
    """  
    Find the image of center 4 that is the 
    closest to the "average" image of the center 
    """
    criterion=nn.MSELoss()
    best_loss=float("inf")
    best_img=None
    for img,_ in dataset:
        img=img.to(device)
        mean_img=torch.mean(img,dim=[1,2]).to(device)
        std_img=torch.std(img,dim=[1,2]).to(device)
        loss = criterion(mean_center_4,mean_img) + criterion(std_center_4,std_img)
    
        if loss.item() < best_loss:
            best_img=img
            best_loss=loss.item()
        
    return best_img

best_rep=find_best_image(train_set_4)

# We fit the macenko staining normalizer to this image
normalizer_macenko.fit(best_rep.unsqueeze(0))

mean = [0.7439, 0.5892, 0.7210] 
std = [0.1717, 0.2065, 0.1664]

def normalize_stain(batch):
    return normalizer_macenko.transform(batch.to(device).unsqueeze(0)).squeeze()

# We define our preprocessing pipeline for the data
# First: we resize them
# Second: we adapt the staining to the most representative image of center 4
# Third: we normalize val and test set with the statistic of training data
transform = transforms.Compose([
            transforms.Resize((98, 98)),
            normalize_stain,
            transforms.Normalize(mean=mean, std=std),
        ])


# We use the large dino version for the feature extraction
feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
feature_extractor.eval()

# We unfreeze the last block to fine tune it with our classification head
for name, param in feature_extractor.named_parameters():
    if  "blocks.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


### Defining our Dataset and Data Loader ###
train_dataset_stained = BaselineDataset(TRAIN_IMAGES_PATH, transform, 'train')
val_dataset_stained = BaselineDataset(VAL_IMAGES_PATH, transform, 'train')

BATCH_SIZE=64

train_dataloader = DataLoader(train_dataset_stained, shuffle=False, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset_stained, shuffle=False, batch_size=BATCH_SIZE)


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
            train_pred = model(train_x.to(device)).squeeze()
            train_y = train_y.float()
            loss = criterion(train_pred, train_y.to(device))
            loss.backward()
            optimizer.step()

            train_losses.extend([loss.item()] * len(train_y))
            train_metric = metric(train_pred.cpu(), train_y.int().cpu())
            train_metrics.extend([train_metric.item()] * len(train_y))
        
        model.eval()
        val_metrics, val_losses = [], []
        
        for val_x, val_y in tqdm(val_dataloader, leave=False):
            with torch.no_grad():
                val_pred = model(val_x.to(device)).squeeze()
            val_y = val_y.float()
            loss = criterion(val_pred, val_y.to(device))
            
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


class HistoClassifierHead(nn.Module):

    def __init__(self, dim_input, hidden_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(dim_input, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.layernorm1 = nn.LayerNorm(dim_input)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.layernorm3 = nn.LayerNorm(hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.dropout(self.layernorm2(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.sigmoid(x) 

        return x 
    

# We define our classification head
model_prob = HistoClassifierHead(dim_input=feature_extractor.num_features, hidden_dim=64, dropout=0.2)
model_prob.to(device)

model=nn.Sequential(feature_extractor,model_prob)

OPTIMIZER = 'AdamW'
OPTIMIZER_PARAMS = {'lr': 5e-4, 'weight_decay' : 0.02}
LOSS = 'BCELoss'
METRIC = 'Accuracy'
NUM_EPOCHS = 100
PATIENCE = 25

# Training process with early stopping on the val loss
train_model(model, train_dataloader, val_dataloader, device, optimizer_name=OPTIMIZER, optimizer_params=OPTIMIZER_PARAMS, 
                 loss_name=LOSS, metric_name=METRIC, num_epochs=NUM_EPOCHS, patience=PATIENCE, save_path='finetune_dino_stained.pth')



### EVALUATION ON THE TEST SET ### 
model.eval()
model.to(device)

prediction_dict = {}

TEST_IMAGES_PATH = 'test.h5'
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    test_ids = list(hdf.keys())

solutions_data = {'ID': [], 'Pred': []}
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    for test_id in tqdm(test_ids):
        img = np.array(hdf.get(test_id).get('img'))
        img = transform(torch.tensor(img)).unsqueeze(0).float()
        #with torch.no_grad():
        pred = model(img.to(device)).detach().cpu()
        solutions_data['ID'].append(int(test_id))
        solutions_data['Pred'].append(int(pred.item() > 0.5))
solutions_data = pd.DataFrame(solutions_data).set_index('ID')
solutions_data.to_csv('finetune_dino_stained.csv')