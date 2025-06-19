import shutil

import time

import pandas as pd
import numpy as np

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import normalize
from torch.utils.data import random_split, WeightedRandomSampler
from torchvision.transforms import v2
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset.utils import HAM10000Segmentation, HAM10000, ClearHair, SegmentAndTransform


def get_segmentation_dataloader(image_dir: str, mask_dir: str) -> tuple:
    transform_seg = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    dataset_seg = HAM10000Segmentation(image_dir, mask_dir, transform=transform_seg)
    train_dataset_seg, test_dataset_seg = torch.utils.data.random_split(dataset_seg, [0.8, 0.2])
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=64, shuffle=True)
    test_loader_seg = torch.utils.data.DataLoader(test_dataset_seg, batch_size=64, shuffle=True)

    return train_loader_seg, test_loader_seg


def get_classifier_dataloader(image_dir: str, mask_dir: str, model, device) -> tuple:
    transform_classification = v2.Compose([
        ClearHair(),
        SegmentAndTransform(model, device),
    ])
    dataset_cls = HAM10000(annotations_file="data/skin_cancer_data/HAM10000_metadata.csv",
                           img_dir=image_dir,
                           transform=transform_classification)
    train_dataset_cls, test_dataset_cls = torch.utils.data.random_split(dataset_cls, [0.8, 0.2])

    train_labels = [train_dataset_cls.dataset.labels.iloc[i, 1] for i in train_dataset_cls.indices]
    class_sample_counts = np.bincount(train_labels)
    class_weights = 1 / class_sample_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader_cls = torch.utils.data.DataLoader(train_dataset_cls, batch_size=64, sampler=sampler, drop_last=True, num_workers=4)
    test_loader_cls = torch.utils.data.DataLoader(test_dataset_cls, batch_size=64, drop_last=True, num_workers=4)

    return train_loader_cls, test_loader_cls

