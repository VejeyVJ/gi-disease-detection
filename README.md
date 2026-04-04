# CLAHE-Enhanced EfficientNet Framework for Gastrointestinal Disease Detection

A deep learning pipeline combining EfficientNetB0 with CBAM attention and CLAHE contrast enhancement for automated classification of gastrointestinal diseases from endoscopic images, achieving **96.2% validation accuracy** on the Kvasir v2 dataset.

## Overview

Endoscopic image interpretation is subjective and time-consuming (15–45 min per case). This project builds an automated 6-class GI disease classifier that enhances image contrast before classification, improving detection of subtle mucosal textures, lesion boundaries, and inflammation patterns.

### Pipeline

```
Endoscopic Image → CLAHE Enhancement (LAB L-channel) → Class-Wise Augmentation
→ EfficientNetB0 (ImageNet pretrained) → CBAM (Channel + Spatial Attention)
→ Global Average Pooling → Dense + Softmax → 6-Class Prediction
```

## Results

| Metric | Baseline (w/o CLAHE) | Enhanced (w/ CLAHE) |
|--------|---------------------|---------------------|
| Accuracy | 93.6% | 96.2% |
| F1 Score | 93.7% | 96.2% |

### Per-Class Sensitivity (Enhanced Model)

| Class | Sensitivity | Specificity |
|-------|------------|-------------|
| Dyed-lifted polyps | 0.903 | 0.988 |
| Dyed-resection margins | 0.913 | 0.988 |
| Esophagitis | 0.997 | 0.998 |
| Normal mucosa | 0.996 | 0.999 |
| Polyps | 0.939 | 0.993 |
| Ulcerative colitis | 0.954 | 0.990 |

## Key Components

- **CLAHE Preprocessing:** Applied on L-channel in LAB color space to enhance local contrast without altering color information, making subtle pathological features more visible
- **EfficientNetB0 Backbone:** Compound-scaled CNN with ImageNet transfer learning, modified for 128×128×3 input
- **CBAM Attention:** Sequential channel and spatial attention to focus on diagnostically relevant image regions
- **Class-Wise Augmentation:** Conservative augmentation for normal tissue, aggressive augmentation (flip, rotation, brightness) for pathological classes to handle class imbalance

## Project Structure

```
├── KVASIR_FINAL__1_.ipynb          # Full pipeline: preprocessing, model, training, evaluation
├── Gastrointestinal.pdf            # Research paper with methodology and results
├── IPA_VJ_FINAL_CHAPTER.docx       # Detailed project chapter
└── README.md
```

## Tech Stack

- **Framework:** TensorFlow/Keras
- **Preprocessing:** OpenCV (CLAHE), scikit-image
- **Dataset:** Kvasir v2 (6 classes, 70/30 stratified split)
- **Optimizer:** AdamW with ReduceLROnPlateau scheduler
- **Training:** 30 epochs, batch size 32, dropout 0.5

## Dataset

The [Kvasir v2](https://datasets.simula.no/kvasir/) dataset contains endoscopic images across 6 GI conditions: dyed-lifted polyps, dyed-resection margins, esophagitis, normal mucosa, polyps, and ulcerative colitis.

## Citation

```
Vejeya Prasad C. "A CLAHE Enhanced EfficientNet Framework for 
Gastrointestinal Disease Detection." SASTRA Deemed University, 2025.
```

## License

This project is part of academic coursework at SASTRA Deemed University, Thanjavur, India.
