import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models

from src.dataset.dataloader import get_classifier_dataloader
import segmentation_models_pytorch as smp

import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "skin_cancer_data")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_resnet34_segmentation.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 25
BATCH_SIZE = 64

seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
seg_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE))
seg_model = seg_model.to(DEVICE)

train_loader, test_loader = get_classifier_dataloader(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    model=seg_model,
    device=DEVICE
)
train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 classes
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_train, loss_test = [], []
    acc_train, acc_test = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        loss_train.append(epoch_loss)
        acc_train.append(epoch_acc.item())
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        model.eval()
        running_loss, running_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size
        loss_test.append(epoch_loss)
        acc_test.append(epoch_acc.item())

        print(f"Test  Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed:.0f}s. Best test acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, loss_train, loss_test, acc_train, acc_test


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    model, loss_train, loss_test, acc_train, acc_test = train_model(
        model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
    )

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "resnet_classifier_best.pth"))
