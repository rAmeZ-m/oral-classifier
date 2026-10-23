# Oral Disease Classification with CNNs

Image classification of 6 oral conditions using PyTorch. I trained a custom residual CNN from scratch and compared it with two pretrained models (ResNet18, EfficientNet-B0), then deployed the best one as a web app.

**Live demo:** [https://oral-classifier-64cqrivizh5da32jvgqudd.streamlit.app/](https://oral-classifier-64cqrivizh5da32jvgqudd.streamlit.app/)
> If the app is asleep, click the wake-up button and wait about a minute.

**Kaggle notebook:** [https://www.kaggle.com/code/ramezmalak/oralvision](https://www.kaggle.com/code/ramezmalak/oralvision)

> ⚠️ Educational project only. Not a medical diagnosis tool.

<!-- 📸 IMAGE: screenshot of the deployed app with a prediction -->
![App screenshot](/images/app_screenshot.png)

---

## Dataset

[Oral Diseases (Kaggle)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases), 6 classes:

| Class | Images (original) |
|---|---|
| Calculus | 1296 |
| Caries | 219 |
| Gingivitis | 2349 |
| Hypodontia | 1251 |
| Mouth Ulcer | 265 |
| Tooth Discoloration | 183 |

**Data handling decisions**
- The dataset also ships *augmented* copies and a YOLO-annotated folder. I used **only the original images** and ignored the rest.
- Reason: augmented file names don't reveal which original they came from, so mixing them into a random split would put copies of the same image in train and test (**data leakage**) and inflate accuracy.
- Stratified split **70 / 15 / 15** (train / val / test), test set = 559 images.
- Class imbalance (about 13x between the largest and smallest class) handled with a **class-weighted loss** and **label smoothing (0.1)**.
- Augmentation is applied **at training time only** (flip, small rotation, brightness/contrast jitter).

---

## Models

| Model | Pretrained | Params | Epochs |
|---|---|---|---|
| Custom ResCNN (from scratch) | No | 4.9M | 60 |
| ResNet18 | ImageNet | ~11.2M | 25 |
| EfficientNet-B0 | ImageNet | ~4.0M | 25 |

The custom ResCNN uses a 7x7 stem, 4 residual blocks (64 → 512 channels), global average pooling and dropout.

Training setup: Adam, ReduceLROnPlateau, mixed precision, best checkpoint chosen by validation accuracy.

---

## Results (test set, 559 images)

| Model | Test Acc | Macro F1 | Weighted F1 |
|---|---|---|---|
| Custom ResCNN (from scratch) | 80.1% | 0.764 | 0.805 |
| ResNet18 | 82.6% | 0.830 | 0.828 |
| **EfficientNet-B0** | **85.7%** | **0.876** | **0.859** |

### Per-class F1

| Class | Test samples | ResCNN | ResNet18 | EfficientNet-B0 |
|---|---|---|---|---|
| Calculus | 130 | 0.716 | 0.709 | **0.750** |
| Caries | 22 | 0.491 | 0.704 | **0.809** |
| Gingivitis | 235 | 0.808 | 0.817 | **0.843** |
| Hypodontia | 126 | 0.937 | 0.965 | **0.976** |
| Tooth Discoloration | 19 | 0.732 | 0.821 | **0.895** |
| Ulcers | 27 | 0.900 | 0.963 | **0.982** |

**Takeaways**
- EfficientNet-B0 was best on every class, with fewer parameters than ResNet18.
- Transfer learning helped the minority classes the most (Caries F1: 0.49 → 0.81).

<!-- 📸 IMAGE: confusion matrix of EfficientNet-B0 -->
![Confusion matrix](/images/confusion_matrix_effnet.png)

---

## Grad-CAM

Grad-CAM heatmaps show which image regions drive the EfficientNet-B0 prediction.

<!-- 📸 IMAGE: Grad-CAM grid, one image per class -->
![Grad-CAM](/images/gradcam.png)

---

## Limitations

- **Label ambiguity (Calculus vs. Gingivitis):** this pair accounts for most errors (about 62 of 79 mistakes for EfficientNet-B0) across all three models. Inspecting misclassified images showed many photos with features of both conditions, while the dataset is single-label.
- **Some images in the "original" folders look augmented** (rotated, color-distorted), so a small amount of leakage between splits is possible.
- **Small test sets** for Caries (22) and Tooth Discoloration (19): their metrics move a lot with just a few images.
- **Watermarks:** some Caries images carry a stock-photo watermark, which a model could pick up as a shortcut.
- Not validated on external data or by a clinician.

---

## Run locally

```bash
git clone https://github.com/rAmeZ-m/oral-classifier.git
cd oral-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Repo contents

| File | Description |
|---|---|
| `app.py` | Streamlit app (inference on CPU) |
| `oral_effnet_b0.pth` | Trained EfficientNet-B0 weights |
| `class_names.json` | Class labels |
| `requirements.txt` | Dependencies |

## Tech stack

PyTorch, torchvision, scikit-learn, Matplotlib, Seaborn, Streamlit

## Author

**Ramez Malak Boushra**
[GitHub](https://github.com/rAmeZ-m) · [LinkedIn](https://linkedin.com/in/ramez-malak-318209254)
