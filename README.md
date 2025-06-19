**# Skin Cancer Segmentation & Classification

Below are instructions for running inference and training new models.

---

## Inference

To test the models without training, open and run the notebook `notebooks/Model_Inference.ipynb`:

1. Download the pre-trained segmentation weights (`unet_resnet34_segmentation.pth`) from Google Drive.
2. Create a folder named `models` at the root of this repository and place the downloaded file there
3. Create and activate a virtual environment, then install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Launch `notebooks/Model_Inference.ipynb` in Jupyter or JupyterLab.
## Training

If you need to train the models from scratch, follow these steps:

1. **Kaggle API token**

   * Go to your Kaggle account settings and generate a new API token (`kaggle.json`).
   * Create a directory for your token and set permissions:

     ```bash
     mkdir -p ~/.kaggle
     mv /path/to/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

2. **Download the dataset**

   ```bash
   python3 -m src.dataset.load_data
   ```

   This script uses the Kaggle CLI to download the HAM10000 dataset into `data/`.

3. **Train the segmentation model**

   ```bash
   python3 -m src.train_models.train_segmentation_model
   ```

   * A UNet with a ResNet-34 will be trained.
   * The resulting weights file `unet_resnet34_segmentation.pth` is saved to `models/`.
   * This step is mandatory before training other models for classification.

4. **Train the Swin classifier**

   ```bash
   python3 -m src.train_models.train_swin_classifier
   ```

   * This script trains a Swin-based image classifier.
   * Checkpoints will be saved under `models/` or another output folder specified in the code.

