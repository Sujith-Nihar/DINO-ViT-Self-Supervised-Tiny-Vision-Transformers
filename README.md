
# DINO-ViT: Self-Supervised Tiny Vision Transformers on CIFAR-100 and Tiny-ImageNet

This repository contains a complete implementation of the **DINO (Self-Distillation with No Labels)** self-supervised learning framework using a **ViT-Tiny** backbone.  
The project evaluates whether Vision Transformers, when trained with self-supervision, can still learn useful and transferable semantic representations even when:

- The dataset is small (CIFAR-100, Tiny-ImageNet)
- The model capacity is small (ViT-Tiny)
- Compute resources are limited

The work is divided into two fully independent experiment modules:

```
Dino_Tiny_VIT_CIFER_100/       → Self-Supervised DINO on CIFAR-100
Dino_Tiny_VIT_TinyImageNet/    → Self-Supervised DINO on Tiny-ImageNet + STL-10 Transfer
requirements.txt               → Environment dependencies
```

---

## 1. Project Overview

The central goal of this project is to investigate whether **self-supervised ViT models** exhibit the same emergent properties observed in the original DINO paper, even when trained under highly constrained conditions. These properties include:

- Strong feature separability in embedding space  
- Meaningful attention activation resembling object boundaries  
- Robust k-NN and linear probe performance  
- Transferability to other datasets such as STL-10  
- Few-shot learning capability  
- Feature stability across augmentations  

Unlike typical DINO implementations that use ImageNet and larger ViTs, our approach focuses on **TinyViT models** and **small datasets**, providing a more lightweight and computationally feasible SSL pipeline.

---

## 2. Detailed Methodology

### 2.1 Student–Teacher Architecture

The implementation follows the exact structure introduced in the DINO paper:

- A **student network** (ViT-Tiny) is optimized using gradient descent.
- A **teacher network** (same architecture) is updated using an exponential moving average (EMA) of the student parameters.
- The teacher uses a lower softmax temperature to produce sharper targets.
- A centering mechanism prevents representation collapse.
- The networks operate on multiple augmented views of the same image.

This framework encourages the student to match the teacher’s output distribution without requiring labels.

### 2.2 Multi-Crop Augmentation

We implement full DINO-style multi-crop augmentation, including:

- Two high-resolution global crops
- Several low-resolution local crops  
- Random resized cropping  
- Color jittering and grayscale  
- Gaussian blur and occasional solarization  
- Standard ImageNet normalization  

This ensures the model learns invariance to viewpoint, color, and scale changes.

### 2.3 Backbone and Projection Heads

- The backbone is **ViT-Tiny (patch16)** implemented using the `timm` library.
- The student and teacher both include a **DINO projection head**, an MLP producing high-dimensional logits.
- Final layer uses weight normalization to stabilize SSL training.

### 2.4 Optimization and Training Setup

- Optimizer: AdamW  
- Learning rate: cosine schedule  
- Weight decay: cosine schedule  
- Mixed precision training for efficiency  
- EMA momentum schedule for teacher update  
- No labels are used at any stage of SSL pretraining  

Each epoch processes all multi-crop batches, computes DINO loss, updates student, and updates the teacher via EMA.

---

## 3. Experiment Modules

### Module 1: CIFAR-100 Self-Supervised Learning  and STL-10 Transfer
Notebook: `Dino_Tiny_VIT_CIFER_100/DINO_Tiny_VIT_CIFER_100_FINAL.ipynb`

This module trains ViT-Tiny on CIFAR-100 without labels using the DINO framework.  
The implementation includes:

- CIFAR-100 dataloader with multi-crop augmentation
- Full DINO SSL training loop
- Student–Teacher EMA updates
- DINO loss with centering and temperature scaling
- Feature extraction from the trained backbone
- k-NN evaluation for unsupervised classification
- Linear probe evaluation for supervised downstream performance
- Visualization of attention from the final transformer layers

This module demonstrates that even on a small dataset, DINO-ViT learns semantically meaningful features.

---

### Module 2: Tiny-ImageNet Self-Supervised Learning and STL-10 Transfer  
Notebook: `Dino_Tiny_VIT_TinyImageNet/TinyImageNet.ipynb`

This module scales the DINO SSL pipeline to Tiny-ImageNet (100k images).  
It includes:

- Automated download and reorganization of Tiny-ImageNet (train/val)
- Multi-crop augmentations configured for 128x128 inputs
- DINO SSL training for up to 300 epochs
- ViT-Tiny backbone with DINO projection head
- k-NN evaluation on Tiny-ImageNet features
- Linear probe training and validation
- Transfer learning to STL-10 using:
  - Frozen feature extraction
  - k-NN evaluation
  - Few-shot classification (e.g., 5-shot)
- Transformer attention visualization on STL-10 images

This module shows that the model generalizes well across datasets and maintains stable semantic representations.

---

## 4. Evaluation Methods

### k-NN Classification  
Assesses the quality of learned representations without training additional layers.

### Linear Probe  
A single fully connected layer is trained on frozen features to test separability.

### STL-10 Transfer Learning  
Evaluates the DINO-ViT model’s generalization through:

- Feature extraction  
- k-NN transfer accuracy  
- Linear probe  
- Few-shot learning performance  

### Attention Map Visualization  
Displays how the final transformer blocks attend to different image regions, demonstrating emergent segmentation-like behavior.

---

## 5. Installation

From the project root directory:

```bash
pip install -r Dino_Tiny_VIT_TinyImageNet/requirements.txt
```

The environment includes PyTorch, torchvision, timm, scikit-learn, matplotlib, and other standard dependencies.

---

## 6. Running the Project

### CIFAR-100 Training
Open and run:
```
Dino_Tiny_VIT_CIFER_100/DINO_Tiny_VIT_CIFER_100_FINAL.ipynb
```

### Tiny-ImageNet Training
Open and run:
```
Dino_Tiny_VIT_TinyImageNet/TinyImageNet.ipynb
```

Ensure GPU acceleration is enabled in your execution environment.

---

## 7. Summary of Results

### CIFAR-100
- Learned features cluster by class even without labels  
- Linear probe achieves meaningful accuracy  
- Attention maps show object-focused regions  

### Tiny-ImageNet
- Higher k-NN and linear probe performance  
- Feature space shows clear separability  
- Transformer attention aligns with object boundaries  

### STL-10 Transfer
- Representations transfer effectively across datasets  
- Strong few-shot learning performance  
- Consistent attention behavior  

---

## 8. Contributions

### Sujith
- CIFAR-100 preprocessing  
- DINO-ViT training on CIFAR-100  
- CIFAR-100 k-NN and linear probe analysis  
- CIFAR-100 attention visualizations  
- Documentation for CIFAR-100 implementation
- Attention visualizations for CIFER-100 and STL-10 
- Final project documentation and analysis

### Hrishitaa
- Tiny-ImageNet and STL-10 preprocessing  
- DINO-ViT training on Tiny-ImageNet  
- Tiny-ImageNet k-NN and linear probe evaluations  
- STL-10 transfer learning and few-shot experiments  
- Attention visualizations for Tiny-ImageNet and STL-10  
- Final project documentation and analysis  

---

## 9. References

All referenced work, including DINO, Vision Transformers, SimCLR, MoCo, BYOL, SwAV, and DeiT, is cited in the project proposal and accompanying documents.

---

## 10. Conclusion

This project demonstrates that **self-supervised ViT models** are capable of learning strong, semantically meaningful representations even under highly constrained data and compute conditions.  
Our results show that TinyViTs, when trained with DINO, maintain:

- High-quality feature embeddings  
- Robust cross-dataset generalization  
- Clear semantic attention patterns  

This repository provides a complete and reproducible implementation of DINO-ViT for small-scale datasets.

