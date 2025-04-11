# DLMI_histopathology

This repository contains the code of our team "Doctor House" for the Kaggle Challenge MVA DLMI 2025 - Histopathology OOD classification

Useful files:
- `dataset.py`: helper file containing the dataset classes
- `models.py`: file containing the code of the models to train, and the implementation fo StainGAN and StainNet.
- `train.py`: file containing the train function

Training files:
- `dino_last_block_finetune.py`: code to finetune dino's last block
- `lora_finetune_dino.py`: code to train a LoRA adaptator for Dino 
- `stain_tools.py`: some experiments with Macenko
- `stainet_dino_last_block_features.py`: Code to normalize images with StainNet, and finetune dino's last block
- `staingan_training.py`: code to compute features using StainGAN and Dino, and train a classifier head 
