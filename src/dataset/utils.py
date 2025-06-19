import os
import shutil

import time
import copy

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor, CenterCrop, Normalize
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from torchvision.transforms.functional import normalize
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split
from torchvision.transforms import v2
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from PIL import Image
from tempfile import TemporaryDirectory

annotations_file = "skin_cancer_data/HAM10000_metadata.csv"
images_dir = "skin_cancer_data/images"

label_map = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}


class HAM10000Segmentation(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", "_segmentation.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask


class ClearHair(torch.nn.Module):
    def forward(self, img):
        img_np = np.array(img)
        new_image, _ = dull_razor(img_np)
        return Image.fromarray(new_image)


class SegmentAndTransform:
    def __init__(self, seg_model, device, image_size=(224, 224)):
        self.seg_model = seg_model
        self.device = device
        self.image_size = image_size
        self.seg_model.eval()

    def __call__(self, img):
        img_resized = resize(to_tensor(img), self.image_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.seg_model(img_resized)
            mask = torch.sigmoid(output)[0, 0]
            mask_bin = (mask > 0.3).float()

        img_tensor = img_resized.squeeze(0) * mask_bin

        img_tensor = normalize(img_tensor,
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

        return img_tensor


class HAM10000(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
      df = pd.read_csv(annotations_file)[["image_id", "dx"]]
      df["image_id"] = df["image_id"] + ".jpg"
      df["dx"] = df["dx"].map(label_map)
      self.labels = df
      self.img_dir = img_dir
      self.transform = transform
      self.target_transform = target_transform

  def __len__(self):
      return len(self.labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
    image = Image.open(img_path).convert("RGB")
    label = self.labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


def dull_razor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)

    return inpainted, mask
