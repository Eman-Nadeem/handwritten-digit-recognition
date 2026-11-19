# Handwritten Digit Prediction: CNN vs Vision Transformer

A deep learning project that trains a **Convolutional Neural Network (CNN)** and a small **attention-based Vision Transformer (ViT)** to recognize handwritten digits (0–9), and serves both through an interactive **Streamlit** web app. Upload a photo of a digit and see the prepared 8×8 input alongside each model's prediction.

Built for the **AI Engineering Bootcamp (PSEB-FSB-01), Module 04: Sunday Project**.

**Author:** Eman Nadeem
**GitHub:** [github.com/Eman-Nadeem](https://github.com/Eman-Nadeem/)
**LinkedIn:** [linkedin.com/in/Emaan-Nadeem](https://www.linkedin.com/in/Emaan-Nadeem)

---

## Features

- Trains and compares two architectures on the same stratified train/validation/test split
- CNN taking `[batch, 1, 8, 8]` input and producing 10 class scores
- ViT that splits each image into sixteen 2×2 patches (`[batch, 16, 4]`), applies a learned patch projection, learned positional embeddings and a 2-layer Transformer encoder (multi-head self-attention)
- Evaluation with test accuracy, macro F1, confusion matrices and a majority-class baseline
- Robust image preprocessing with Pillow: transparency handling, grayscale, conditional inversion, noise-tolerant cropping, centering with margin, resize to 8×8 and contrast-stretched 0–1 normalization
- Streamlit app that previews the original image and the processed 8×8 input, shows both predictions, and rejects blank, missing and unreadable images without crashing
- Saved model weights, so the app runs from a fresh launch with **no retraining**

---

## Results

Evaluated on the held-out test set (270 images), which was never used for training or model selection.

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| CNN | 98.15% | 0.9811 |
| ViT | 94.07% | 0.9408 |
| Majority-class baseline | 10.37% | n/a |

Both models are far above the baseline. The CNN performs better on this small 8×8 dataset, which is expected: convolutions are well suited to small images and limited training data, while Transformers usually need more data to show their strengths.

> **Note on real photos:** These scores are on the sklearn digits dataset. Photos of handwritten digits can be harder, because strokes, lighting, backgrounds and image size differ from the small, centered, high-contrast training images. Matching the pixel scale does not guarantee the same accuracy on real photos.

---

## Project Structure

```
.
├── app.py                    # Streamlit app (loads saved models, predicts)
├── prepare_data.ipynb        # Load digits, normalize, stratified split, save .npz
├── train_cnn.ipynb           # Train and validate the CNN, save weights
├── train_vit.ipynb           # Train and validate the ViT, save weights and metrics
├── evaluate_models.ipynb     # Test accuracy, macro F1, confusion matrices, baseline
├── prcocess_image.ipynb      # Step-by-step image preprocessing walkthrough
├── digit.png                 # Sample handwritten digit for testing
├── data/
│   └── digit_splits.npz      # Saved train/validation/test splits
├── models/
│   ├── cnn_model.pth         # Trained CNN weights
│   ├── vit_model.pth         # Trained ViT weights
│   └── vit_metrics.json      # ViT test accuracy and macro F1
└── README.md
```

---

## Installation

Requires Python 3.9 or newer (developed with Python 3.13).

```bash
pip install torch scikit-learn streamlit pillow numpy matplotlib
```

Optionally, use a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

---

## Usage

### Run the app (no training needed)

The trained weights are included in `models/`, so you can launch the app directly:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501) and upload a PNG or JPEG image containing **one handwritten digit**.

### Reproduce training and evaluation

Run the notebooks from the project folder in this order:

1. `prepare_data.ipynb` creates `data/digit_splits.npz`
2. `train_cnn.ipynb` saves `models/cnn_model.pth`
3. `train_vit.ipynb` saves `models/vit_model.pth` and `models/vit_metrics.json`
4. `evaluate_models.ipynb` reports accuracy, macro F1, confusion matrices and the baseline

`prcocess_image.ipynb` is an optional walkthrough of the preprocessing steps used by the app.

---

## How It Works

### 1. Data
- `sklearn.datasets.load_digits()` provides 1,797 images of size 8×8 with labels 0–9 and pixel values 0–16.
- Pixels are divided by 16 to give values in the 0–1 range.
- A stratified split gives 70% train (1,257), 15% validation (270) and 15% test (270). The same split is used for both models.

### 2. CNN
`Conv2d(1→16, 3×3, padding=1)` → ReLU → `MaxPool2d(2)` → Flatten → `Linear(256→64)` → ReLU → `Linear(64→10)`.
Trained with CrossEntropyLoss and Adam (lr = 0.001), using mini-batches of 32 for 30 epochs.

### 3. Vision Transformer
1. Split the 8×8 image into sixteen non-overlapping 2×2 patches → `[N, 16, 4]`
2. Patch embedding: trainable `Linear(4→16)`
3. Add learned positional embeddings `[1, 16, 16]`
4. `TransformerEncoder`: 2 layers, `d_model=16`, 4 attention heads
5. Mean-pool over patches → `Linear(16→10)`

Trained with the same loss, optimizer, batch size and epochs as the CNN.

### 4. Image preprocessing (app)
The app uses the same 0–1 pixel range as training:

1. Open the PNG/JPEG with Pillow; place transparent images on a white background
2. Convert to grayscale
3. Invert only if the background is bright, so strokes are bright on dark like the training data
4. Threshold out faint noise, then crop to the digit's bounding box (blank images are rejected)
5. Center on a square canvas with a small margin
6. Resize to 8×8 (Lanczos)
7. Scale to 0–1 and stretch contrast so the stroke reaches 1.0

### 5. Error handling
| Input | App behavior |
|---|---|
| No file uploaded | Shows a prompt to upload an image |
| Blank image | Rejected with a clear message |
| Corrupt or unreadable file | Shows an error message, no crash |

---

## Limitations

- Models are trained only on the small 8×8 sklearn digits, not on real handwriting photos, so accuracy on photos can be lower than on the test set.
- The two models may disagree on difficult images. The app shows both predictions and flags disagreement.
- Works best with a single, clearly written, dark-on-light (or light-on-dark) digit with good contrast.

## Possible Improvements

- Add controlled data augmentation (shifts, rotations, stroke thickness) to improve robustness on photos
- Collect a small photo test set to measure real-world accuracy
- Show prediction confidence for each model
- Add an optional FastAPI endpoint

---

## Contact

**Eman Nadeem**
GitHub: [Eman-Nadeem](https://github.com/Eman-Nadeem/)
LinkedIn: [Emaan-Nadeem](https://www.linkedin.com/in/Emaan-Nadeem)