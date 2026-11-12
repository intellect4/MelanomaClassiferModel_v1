# -*- coding: utf-8 -*-
"""Melanoma Classifier Training Script for Google Colab (T4 GPU).

Run this in Google Colab to train the MultiScale Attention DenseNet-121 model
on your full 7,677-image dataset using NVIDIA T4 GPU acceleration in ~20-25 minutes.
"""

import os
import glob
import time
import zipfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 1. Mount Google Drive (Optional, for saving checkpoint permanently)
try:
    from google.colab import drive, files
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    SAVE_DIR = "/content/drive/MyDrive/MelanomaProject/checkpoints"
except Exception:
    print("Drive mounting skipped. Checkpoint will be saved in local session.")
    SAVE_DIR = "/content/checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_melanoma_model.pth")

# 2. Check GPU Acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Warning: GPU not active. Change runtime type to T4 GPU in Runtime menu!")

# 3. Locate or Unzip Dataset
DATASET_PATH = "/content/DataSet"

if not os.path.exists(DATASET_PATH):
    zip_candidates = [
        "/content/DataSet_224.zip",
        "/content/drive/MyDrive/DataSet_224.zip",
        "/content/drive/MyDrive/DataSet.zip"
    ]
    found_zip = None
    for z in zip_candidates:
        if os.path.exists(z):
            found_zip = z
            break

    if found_zip:
        print(f"Found archive: {found_zip}. Unzipping...")
        with zipfile.ZipFile(found_zip, 'r') as zf:
            zf.extractall("/content/")
        print("Unzipped successfully into /content/DataSet!")
    else:
        print("Waiting for DataSet_224.zip upload...")
        print("Please upload DataSet_224.zip using the Files panel on the left, or upload below:")
        try:
            uploaded = files.upload()
            for fn in uploaded.keys():
                if fn.endswith('.zip'):
                    with zipfile.ZipFile(fn, 'r') as zf:
                        zf.extractall("/content/")
                    print(f"Extracted {fn} successfully!")
        except Exception as e:
            print(f"Upload prompt: {e}")

# Verify paths
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Could not locate {DATASET_PATH}. Ensure DataSet_224.zip is uploaded to /content/.")

print(f"Dataset root verified: {DATASET_PATH}")
print(f"Checkpoint target   : {BEST_MODEL_PATH}")

# 4. MultiScaleAttentionMelanomaNet Architecture
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=1024, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        concated = torch.cat(x, 1)
        out = self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(concated))))))
        return out

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1}", _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size))

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_f = layer(features)
            features.append(new_f)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_f))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_f, out_f, 1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, 2))

class DenseNet121Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)

        self.denseblock1 = _DenseBlock(6, 64, 4, 32)
        self.transition1 = _Transition(256, 128)
        self.denseblock2 = _DenseBlock(12, 128, 4, 32)
        self.transition2 = _Transition(512, 256)
        self.denseblock3 = _DenseBlock(24, 256, 4, 32)
        self.transition3 = _Transition(1024, 512)
        self.denseblock4 = _DenseBlock(16, 512, 4, 32)
        self.norm5 = nn.BatchNorm2d(1024)

    def forward(self, x):
        out = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        out = self.denseblock1(out)
        out = self.transition1(out)
        out = self.denseblock2(out)
        out = self.transition2(out)
        out = self.denseblock3(out)
        out = self.transition3(out)
        out = self.denseblock4(out)
        return self.norm5(out)

class MultiScaleAttentionMelanomaNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = DenseNet121Features()
        self.channel_attn = ChannelAttention(1024, 16)
        self.spatial_attn = SpatialAttention(7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.channel_attn(feat) * feat
        feat = self.spatial_attn(feat) * feat
        pool = torch.cat([self.avg_pool(feat), self.max_pool(feat)], dim=1)
        return self.classifier(torch.flatten(pool, 1))

# 5. Dataset Loader
class HistologyColabDataset(Dataset):
    def __init__(self, file_list, is_train=True):
        self.file_list = file_list
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        img = Image.open(item["path"]).convert("RGB")
        img_np = np.array(img, dtype=np.float32)

        # Min-max normalization
        i_min, i_max = img_np.min(), img_np.max()
        norm_img = (img_np - i_min) / (i_max - i_min + 1e-8)

        # Data augmentation
        if self.is_train:
            if np.random.rand() > 0.5:
                norm_img = np.fliplr(norm_img)
            if np.random.rand() > 0.5:
                norm_img = np.flipud(norm_img)
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                norm_img = np.rot90(norm_img, k=k)

        # Standardize
        standardized = (norm_img - self.mean) / self.std
        tensor = torch.tensor(np.ascontiguousarray(standardized).transpose(2, 0, 1), dtype=torch.float32)
        return tensor, torch.tensor(item["label"], dtype=torch.long)

def index_images(base_dir, split_name):
    target_dir = os.path.join(base_dir, split_name)
    items = []
    if not os.path.exists(target_dir):
        for f in os.listdir(base_dir):
            if f.lower() == split_name.lower():
                target_dir = os.path.join(base_dir, f)
                break

    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                full_p = os.path.join(root, f)
                lbl = 0 if "primary" in full_p.lower() else 1
                items.append({"path": full_p, "label": lbl})
    return items

# Index Training & Validation splits
train_items = index_images(DATASET_PATH, "Training")
val_items = index_images(DATASET_PATH, "Validation")

print(f"Training Images Indexed  : {len(train_items)}")
print(f"Validation Images Indexed: {len(val_items)}")

train_loader = DataLoader(HistologyColabDataset(train_items, is_train=True), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(HistologyColabDataset(val_items, is_train=False), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 6. Initialize Model & Optimization
model = MultiScaleAttentionMelanomaNet(num_classes=2).to(device)

# Inverse class weighting
pri_c = sum(1 for d in train_items if d["label"] == 0)
met_c = sum(1 for d in train_items if d["label"] == 1)
total_c = pri_c + met_c
weights = torch.tensor([total_c / (2.0 * max(1, pri_c)), total_c / (2.0 * max(1, met_c))], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.08)
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 2e-5, 'weight_decay': 1e-3},
    {'params': model.channel_attn.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.spatial_attn.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 2e-4, 'weight_decay': 1e-4}
])

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

# 7. Training Execution Loop
epochs = 30
best_val_acc = 0.0
print(f"\n=======================================================")
print(f" Starting GPU Accelerated Fine-Tuning ({epochs} Epochs)")
print(f"=======================================================")

start_train_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        train_total += lbls.size(0)
        train_correct += preds.eq(lbls).sum().item()

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_total += lbls.size(0)
            val_correct += preds.eq(lbls).sum().item()

    scheduler.step()

    train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0.0
    val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
    epoch_sec = time.time() - epoch_start

    print(f"Epoch [{epoch+1:02d}/{epochs}] ({epoch_sec:.1f}s) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/max(1,val_total):.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  --> [NEW BEST] Checkpoint Saved: {val_acc:.2f}%")

total_min = (time.time() - start_train_time) / 60.0
print(f"\nTraining Complete in {total_min:.1f} minutes!")
print(f"Peak Validation Accuracy: {best_val_acc:.2f}%")
print(f"Checkpoint location: {BEST_MODEL_PATH}")

# Trigger direct download prompt in Colab browser
try:
    print("\nPreparing local download prompt...")
    files.download(BEST_MODEL_PATH)
    print("Download started! Save this file as 'best_melanoma_model.pth' in your PC's checkpoints/ folder.")
except Exception as e:
    print(f"Download prompt failed: {e}. You can download the file manually from: {BEST_MODEL_PATH}")
