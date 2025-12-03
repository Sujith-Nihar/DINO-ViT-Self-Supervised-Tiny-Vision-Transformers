# DINO-ViT: Self-Supervised Tiny Vision Transformers on CIFAR-100 and Tiny-ImageNet

This project implements the DINO (Self-Distillation with No Labels) self-supervised learning framework using a lightweight ViT-Tiny backbone.  
The objective is to investigate whether Vision Transformers trained on small datasets with minimal computational resources can still exhibit the emergent semantic properties demonstrated in the original DINO paper.

Two complete self-supervised training pipelines are included:

- **DINO ViT-Tiny on CIFAR-100**
- **DINO ViT-Tiny on Tiny-ImageNet**

Both models are evaluated using:
- k-NN classification  
- Linear probing  
- Cross-dataset transfer to STL-10  
- 5-shot few-shot evaluation  
- Transformer attention visualizations  

---

## Project Structure

```
Dino_Tiny_VIT_CIFER_100/       → DINO SSL on CIFAR-100
Dino_Tiny_VIT_TinyImageNet/    → DINO SSL on Tiny-ImageNet + STL-10 Transfer Evaluation
requirements.txt               → Environment dependencies
```

---

# 1. Overview

The original DINO paper demonstrated that Vision Transformers trained without labels can learn highly semantic and transferable representations.  
This project evaluates if these properties persist when:

- The dataset is **50–100k images instead of ImageNet-1K**
- The model is a **5M–6M parameter ViT-Tiny** instead of ViT-S/B
- Training occurs on **a single A100/T4 GPU** instead of 8–32 GPUs
- The training duration is **100–300 epochs** instead of 300–800

Our results show that **self-supervised learning is still highly effective**, and a small ViT-Tiny can learn strong semantic representations even under extreme resource limitations.

---

# 2. Methodology

## 2.1 Student–Teacher Self-Distillation

DINO uses a dual-network setup:

- **Student ViT-Tiny**: updated via gradient descent  
- **Teacher ViT-Tiny**: updated as an exponential moving average (EMA) of the student  

The student must match the teacher’s probability distribution across multiple augmented views.  
Key elements include:

- Temperature sharpening  
- Centering to prevent collapse  
- EMA teacher updates  
- Multi-view consistency alignment  

---

## 2.2 Multi-Crop Augmentation

Following the original DINO formulation:

- **2 global crops** (high-resolution)
- **8–10 local crops** (low-resolution)
- Strong augmentations:
  - Color jittering  
  - Gaussian blur  
  - Grayscale  
  - Solarization  
  - Random resized crops  

This design enforces invariance across context, scale, and appearance.

---

## 2.3 Architecture

- Backbone: **ViT-Tiny (patch16)**  
- Projection head: **3-layer MLP with weight normalization**  
- Feature dimension: **192 or 384**  
- L2-normalized outputs for contrastive stability  

---

## 2.4 Optimization & Training

- Optimizer: **AdamW**
- LR schedule: **cosine decay**  
- Weight decay: **cosine**  
- EMA teacher with scheduled momentum  
- Mixed precision (**AMP**)  
- Single GPU (A100/T4) runtime  

This allows efficient SSL training even in low-resource environments.

---

# 3. Experiment Modules

## Module 1: CIFAR-100

Notebook: `DINO_Tiny_VIT_CIFER_100_FINAL.ipynb`

Tasks:

- SSL training on CIFAR-100  
- k-NN evaluation  
- Linear probe  
- STL-10 transfer evaluation  
- Attention visualization  

---

## Module 2: Tiny-ImageNet

Notebook: `TinyImageNet.ipynb`

Tasks:

- Dataset organization  
- SSL training on Tiny-ImageNet  
- k-NN evaluation  
- Linear probe  
- STL-10 transfer evaluation  
- 5-shot few-shot learning  
- Attention visualization  

---

# 4. Installation

```
pip install -r requirements.txt
```

GPU required for training.

---

# 5. Results

## 5.1 CIFAR-100 (DINO ViT-Tiny)

### Linear Probe
| Metric | Value |
|--------|--------|
| Train Accuracy | 61.00% |
| Test Accuracy | 61.00% |
| Loss | 1.2500 |

### k-NN (k = 5)
| Dataset | Accuracy |
|---------|----------|
| CIFAR-100 | 59.34% |

---

## 5.2 STL-10 Transfer (CIFAR-trained model)

### k-NN
| Dataset | Accuracy |
|---------|----------|
| STL-10 | 63.40% |

### Linear Probe
| Metric | Value |
|--------|--------|
| Train Accuracy | 63.85% |
| Test Accuracy | 64.12% |

### Few-Shot (5-shot)
| Metric | Value |
|--------|--------|
| Few-shot Accuracy | 56.97% |

---

## 5.3 Tiny-ImageNet (DINO ViT-Tiny)

### Linear Probe
| Metric | Value |
|--------|--------|
| Train Loss | 1.8018 |
| Train Accuracy | 55.73% |
| Val Loss | 1.8709 |
| Val Accuracy | 54.29% |

### k-NN (k = 5)
| Dataset | Accuracy |
|---------|----------|
| Tiny-ImageNet | 53.04% |

---

## 5.4 STL-10 Transfer (TinyImageNet-trained model)

### k-NN
| Dataset | Accuracy |
|---------|----------|
| STL-10 | 53.22% |

### Linear Probe
| Metric | Value |
|--------|--------|
| Train Accuracy | 61.73% |
| Test Accuracy | 55.48% |

### Few-Shot (5-shot)
| Metric | Value |
|--------|--------|
| Few-shot Accuracy | 50.34% |

---

# 6. Comparison with Original DINO Paper

## Original DINO Achievements

| Model | Dataset | Linear Probe | Notes |
|-------|---------|--------------|-------|
| ViT-S/16 | ImageNet-1K | ~77% | Trained with 300 epochs, multi-GPU |
| ViT-B/16 | ImageNet-1K | ~78–79% | Very large model |
| ResNet-50 | ImageNet-1K | 75.3% | SSL baseline |

Training setup in the original paper involved:

- 1.3M images  
- 8–32 GPUs  
- Large ViT-S/B models  
- 300–800 training epochs  

---

## Our Implementation vs. Original DINO

### Dataset
| Setting | Images |
|---------|--------|
| Original DINO | 1,300,000 |
| Ours (CIFAR-100) | 50,000 |
| Ours (Tiny-ImageNet) | 100,000 |

### Model Size
| Model | Params |
|--------|--------|
| ViT-B | 86M |
| ViT-S | 22M |
| **ViT-Tiny (ours)** | **5.7M** |

### Compute Resources
| Setting | GPUs | Epochs |
|---------|------|---------|
| Original DINO | 8–32 A100 | 300–800 |
| **Ours** | **1× A100/T4** | **100–300** |

---

# 7. Why SSL Worked Even With Minimal Resources

Despite using small models, small datasets, and minimal hardware, our implementation succeeded due to:

### 1. Multi-crop Augmentation  
Provided strong invariance signals.

### 2. ViT-Tiny’s Global Attention  
Even small ViTs exhibit strong global reasoning ability.

### 3. EMA Teacher Stability  
Smoothed training dynamics and prevented collapse.

### 4. Distribution-Level Matching  
DINO aligns probability distributions, not pixels—works well even on small datasets.

### 5. Effective Transfer  
CIFAR-trained model achieving **64.12%** on STL-10 indicates robust generalization.

### 6. Efficient Training Strategies  
AMP, cosine LR, and optimized dataloading enabled practical SSL training.

---

# 8. Discussion

- Token-level attention maps demonstrate semantic focus similar to the original DINO paper.  
- CIFAR-trained embeddings generalize **better** to STL-10 than Tiny-ImageNet-trained ones.  
- Few-shot results confirm non-trivial structure in the embedding space.  
- The quality of embeddings is competitive with supervised baselines despite being fully self-supervised.  

This project confirms that **self-supervised representation learning is feasible in low-resource environments**.

---

# 9. Contributions

### Sujith
- CIFAR-100 preprocessing  
- DINO training (CIFAR-100)  
- CIFAR → STL transfer  
- Linear probe, k-NN  
- CIFAR attention analysis  
- Documentation  

### Hrishitaa
- Tiny-ImageNet and STL preprocessing  
- DINO training (Tiny-ImageNet)  
- Tiny-ImageNet → STL transfer  
- Few-shot classification  
- Attention visualization  
- Final documentation  

---

# 10. References

References include DINO (Caron et al., 2021), ViT (Dosovitskiy et al., 2020), SimCLR, MoCo, BYOL, SwAV, and others as listed in the project proposal.

