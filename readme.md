# CelebA Face Generation with DCGAN

## Project Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic human faces. The model is trained on the CelebA dataset, which contains over 200,000 celebrity face images. The main goal is to demonstrate the capabilities of GANs in generating high-quality, realistic face images from random noise vectors.

## Setup Instructions

### Using Conda (Recommended)
```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate celeba-gan
```


### Using Pip
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Pre-trained Model

1. Download the pre-trained generator model from [Google Drive Link]
2. Place the downloaded final_generator.pth file in the checkpoints/ directory

### How to Run

The project includes a demo script that showcases the model's face generation capabilities. You can run it in several ways:

1. Generate a grid of faces (default):
```python
python demo.py
```

2. Generate a single face:

```python
python demo.py --mode single
```

3. Generate a custom number of faces with a specific seed:
```python
python demo.py --mode grid --num_images 25 --seed 42
```


## Expected Output
After running the demo script:

1. Generated images will be saved in the results/ directory
2. For grid mode: You will see a grid of generated faces, each sized 64x64 pixels
3. For single mode: You will see one generated face image
4. File names include timestamps for easy tracking
5. Console output will show the path to the generated images

Example console output:
```txt
Using device: cuda
Successfully loaded pre-trained generator model
Generated image grid saved to: results/face_grid_20240214_143022.png
Demo completed successfully!
```


## Pre-trained Model
Download the pre-trained generator model from:
[[Google Drive Link to final_generator.pth](https://drive.google.com/file/d/1Be1mul1n4TR7oM037qDP3nApHkpHBiuF/view?usp=sharing)]
Place the downloaded file in the checkpoints/ directory before running the demo.

## Model Architecture
The DCGAN architecture consists of:

- Generator: Transforms random noise into 64x64 RGB images
- Discriminator: Classifies images as real or generated
- Training: Adversarial training on CelebA dataset for 30 epochs

## Dataset
The CelebA (CelebFaces Attributes) dataset used for training can be found at:CelebA Dataset[https://www.kaggle.com/datasets/jessicali9530/celeba-dataset]

## Acknowledgments

CelebA dataset: Large-scale CelebFaces Attributes (CelebA) Dataset
DCGAN paper: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
PyTorch implementation inspired by the official PyTorch tutorials
Training infrastructure based on NVIDIA's DCGAN implementation