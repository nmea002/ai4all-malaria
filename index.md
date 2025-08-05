---
layout: default
title: Malaria Cell Classification
---
# Malaria Blood Cell Image Classification with CNNs

Malaria continues to cause high mortality rates, especially in less developed countries. Early and accurate diagnosis is crucial to effective treatment, but manual methods are prone to errors and resource-limited. Our project examines how computer vision, specifically convolutional neural networks (CNNs), can facilitate automated malaria detection under diverse image conditions. We are especially focused on how well AI generalizes across datasets with differing characteristics like lighting, blur, and resolution. To reflect challenges encountered in real-world microscopy settings for malaria detection, we applied data augmentation techniques such as gaussian blur, brightness shifts, pixel masking, etc. These changes help us test if CNNs trained on clean images can still perform under challenging conditions. 

## Table of Contents
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Usage Guide](#usage-guide)
- [Results](#results)

## Dataset
We trained our model using the [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data) dataset on Kaggle. This dataset consists of microscopic images of blood cells categorized as **Parasitized** or **Uninfected**.

The images were originally sourced from the [NIH's malaria dataset repository](https://ceb.nlm.nih.gov/repositories/malaria-datasets/), which provides annotated data for developing machine learning models for malaria diagnosis.

## Approach
We built a convolutional neural network (CNN) using PyTorch to classify blood cell images as either parasitized or uninfected. We used a single copy of the dataset and performed an 80/20 split to create training and validation sets within in the program. Training and validations were resized to 128×128 pixels because it's close to the average size of the images. 

The model was trained for 5 epochs with real-time tracking of loss and accuracy across batches. Performance was assessed on both clean and augmented validation sets to evaluate the model’s ability to generalize under varying image conditions.

## Model Architecture
Our convolutional neural network (CNN) consists of two convolutional layers followed by fully connected layers:
- Conv Layer 1: 12 kernels of size 5×5 with ReLU activation
- Max Pooling: 2×2 to reduce dimensions
- Conv Layer 2: 24 kernels of size 5×5 with ReLU activation
- Max Pooling: 2×2 again to reduce dimensions of feature maps
- Flatten: Feature maps are flattened into a single vector
- Fully Connected Layer 1: 128 hidden units with ReLU
- Fully Connected Layer 2: Outputs 2 classes — Parasitized or Uninfected
  
The dimensions of the input images were 3×128×128, where 3 represents the color channels of an RGB image and 128×128 corresponds to the height and width. After two rounds of convolution and pooling, the feature maps were reduced to 24×29×29. These were then flattened and passed through fully connected layers for classification.

## Usage Guide
1. Install the necessary dependencies using:
```bash
pip install torchvision
```
2. Download the [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data) dataset from Kaggle.
3. After extracting the dataset, if you see a nested folder structure like:
```
cell_images/cell_images/Parasitized
```
Delete the inner `cell_images` folder so that your dataset path looks like this:
```
cell_images/
├── Parasitized/
└── Uninfected/
```
4. Place it inside your project directory, using the following structure:
```
your-project-folder/
├── cell_images/
│   ├── Parasitized/
│   └── Uninfected/
├── analysis.ipynb
```
5. Open and run all cells in the `analysis.ipynb` Jupyter notebook.

## Results

<div markdown = "1">
  
| Test Condition          | Accuracy (%) |
|-------------------------|--------------|
| No Augmentation         | 93.98        |
| Random Brightness       | 94.03        |
| Masked Pixels           | 83.13        |
| Low Resolution          | 94.54        |
| Gaussian Blur           | 94.43        |
| Rotation                | 94.10        |
| Affine Shift            | 92.65        |
| Perspective             | 92.76        |
| Combo (Rot., Bri., Blur)| 94.43        |

 </div>
