import os
import numpy as np
import torch
import torch.multiprocessing as mp

from evaluate import load
from transformers import (
    SwinForImageClassification,
    TrainingArguments,
    Trainer
)
from src.dataset.dataloader import get_classifier_dataloader
import segmentation_models_pytorch as smp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "skin_cancer_data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_resnet34_segmentation.pth")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def collate_fn(batch):
    images, labels = zip(*batch)
    pixel_values = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


metric = load("accuracy")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


def main():
    # Ensure "spawn" start method before creating DataLoaders
    mp.set_start_method("spawn", force=True)

    # Load segmentation model for preprocessing
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    seg_model.load_state_dict(
        torch.load(SEG_MODEL_PATH, map_location=DEVICE)
    )
    seg_model.to(DEVICE)

    # Create DataLoaders
    train_loader, test_loader = get_classifier_dataloader(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        model=seg_model,
        device=DEVICE
    )
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Label mapping
    label_map = {
        'akiec': 0, 'bcc': 1, 'bkl': 2,
        'df': 3, 'mel': 4, 'nv': 5,
        'vasc': 6
    }
    index_to_class = {v: k for k, v in label_map.items()}

    # Initialize Swin classifier
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=len(index_to_class),
        id2label={str(k): v for k, v in index_to_class.items()},
        label2id={v: str(k) for k, v in index_to_class.items()},
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="models/swin-classifier",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=25,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        gradient_checkpointing=True
    )
    # Optimize best model by accuracy
    training_args.metric_for_best_model = "accuracy"
    training_args.greater_is_better = True

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train & Save
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    torch.save(
        model.state_dict(),
        os.path.join(BASE_DIR, "models", "swin_classifier_best.pth")
    )


if __name__ == "__main__":
    main()
