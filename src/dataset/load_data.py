import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_datasets(data_dir="../data"):
    os.makedirs(data_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print("Downloading HAM10000 classification dataset...")
    api.dataset_download_files(
        "kmader/skin-cancer-mnist-ham10000",
        path=data_dir,
        unzip=False
    )

    print("Downloading HAM1000 segmentation dataset...")
    api.dataset_download_files(
        "surajghuwalewala/ham1000-segmentation-and-classification",
        path=data_dir,
        unzip=False
    )


def unzip_dataset(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    print(f"Removed archive {zip_path}")


def clean_up_and_merge(data_dir="../data"):
    skin_dir = os.path.join(data_dir, "skin_cancer_data")
    images_dir = os.path.join(skin_dir, "images")
    masks_dir = os.path.join(skin_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Merge classification images from part 1 and part 2
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        src_dir = os.path.join(skin_dir, part)
        if os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                shutil.move(
                    os.path.join(src_dir, fname),
                    os.path.join(images_dir, fname)
                )
            shutil.rmtree(src_dir)

    # Move segmentation masks into the masks directory
    seg_root = os.path.join(data_dir, "ham1000-segmentation-and-classification")
    seg_masks_src = os.path.join(seg_root, "masks")
    if os.path.isdir(seg_masks_src):
        for fname in os.listdir(seg_masks_src):
            shutil.move(
                os.path.join(seg_masks_src, fname),
                os.path.join(masks_dir, fname)
            )
        # Remove the segmentation dataset folder completely
        shutil.rmtree(seg_root)


if __name__ == "__main__":
    base_dir = "data"

    download_kaggle_datasets(base_dir)

    unzip_dataset(
        os.path.join(base_dir, "skin-cancer-mnist-ham10000.zip"),
        os.path.join(base_dir, "skin_cancer_data")
    )
    unzip_dataset(
        os.path.join(base_dir, "ham1000-segmentation-and-classification.zip"),
        os.path.join(base_dir, "ham1000-segmentation-and-classification")
    )

    clean_up_and_merge(base_dir)

    print("Data preparation complete.")
