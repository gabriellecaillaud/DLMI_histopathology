import torch

import functools
import torch.nn as nn
from torchvision.models import squeezenet1_1
from tqdm import tqdm
from pathlib import Path
import os
import random
import torchvision.transforms as transforms
from models import HistoClassifierHead
from dataset import BaselineDataset, PrecomputedDataset,precompute
from train import train_model
from torch.utils.data import Dataset, DataLoader
base_path = Path(os.getcwd())
print(f"{base_path=}")
import numpy as np
from pathlib import Path

class SqueezeNet(nn.Module):
    def __init__(self, n_class=2):
        super(SqueezeNet, self).__init__()
        self.n_class = n_class
        self.base_model = squeezenet1_1(pretrained=True)
        temp = squeezenet1_1(pretrained=False, num_classes=n_class)
        self.base_model.classifier = temp.classifier
        del temp

    def forward(self, x):
        return self.base_model(x)


class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super(StainNet, self).__init__()
        model_list = []
        model_list.append(nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
        model_list.append(nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
            model_list.append(nn.ReLU(True))
        model_list.append(nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))

        self.rgb_trans = nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def normalize_img(img):
    return (img - 0.5)/0.5

def unnormalize_img(img):
    return img * 0.5 + 0.5

def precompute_StainGan(dataloader, model, device):
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
        train_dataset = PrecomputedDataset(*precompute_StainGan(train_dataloader, model_GAN, device))
        val_dataset = PrecomputedDataset(*precompute_StainGan(val_dataloader, model_GAN, device))

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

