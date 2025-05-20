<div align="center">
<h1> MEP-Seg: Multi-frequency Edge Enhancement and Prompt-guided Attention for Medical Image Segmentation </h1>
</div>

## 🎈 News

- [2025.2.19] Training and inference code released

## ⭐ Abstract

Medical image segmentation is crucial for clinical decision-making, treatment planning, and disease tracking. Nonetheless, there are many challenges in medical image segmentation, especially in the exploration of multi-scale and multi-frequency features for more effective edge detection and noise suppression. Meanwhile, it is also imperative to improve the adaptability and generalization ability of the model in medical images of different modalities. To this end, we propose MEP-Seg, which consists of two units: Edge Enhancement Unit (EEU) and Hybrid Prompt Unit (HPU). EEU enhances feature maps through multi-scale convolution and separates multi-frequency information. High-frequency components are used to capture the boundaries of salient objects, while low-frequency components help suppress noise brought by non-salient objects. In addition, the combination of multi-directional and fine-grained global-local offsets optimizes the adaptability of the model to irregular edges. HPU generates high-frequency and low-frequency prompt masks, and extracts transferable segmentation features applicable to a variety of medical cases through a prompt-guided cross-attention mechanism, thereby improving generalization performance. Evaluations on 7 public datasets show that MEP-Seg surpasses 11 existing state-of-the-art methods in segmentation accuracy.

## 🚀 Introduction

<div align="center">
    <img width="400" alt="image" src="asserts/challen.png?raw=true">
</div>

Medical images of different pathologies exhibit significant differences, with the complex edges and noise interference.

## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/network.png?raw=true">
</div>

Illustration of the overall architecture of MEP-Seg.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

```
conda create -n MEPSeg python=3.10
conda activate MEPSeg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: [link](https://challenge.isic-archive.com/data/#2018), [link](https://www.dropbox.com/scl/fi/epzcoqeyr1v9qlv/PH2Dataset.rar?rlkey=6mt2jlvwfkditkyg12xdei6ux&e=1), [link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), [link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [link](https://www.kaggle.com/datasets/balraj98/cvcclinicdb?resource=download), [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018), and [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the MEP-Seg

```
python train.py --datasets ISIC2018
```

### 3. Test the MEP-Seg

```
python test.py --datasets ISIC2018
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/Visualization_.png?raw=true">
</div>

<div align="center">
    Visualization results of twelve state-of-the-art methods and MEP-Seg for different lesions. The red circles indicate areas of incorrect predictions.
</div>

## 🎫 License

The content of this project itself is licensed under [LICENSE](https://github.com/ILoveICCV/RoMERPA-UNet/blob/main/LICENSE).
