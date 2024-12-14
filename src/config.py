import torch

# Model parameters
LATENT_DIM = 100
GEN_FEATURES = 64
DISC_FEATURES = 64
IMAGE_CHANNELS = 3
IMAGE_SIZE = 64

# Training parameters
BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005
BETA1 = 0.5

# Data parameters
DATA_ROOT = "data/img_align_celeba"
NUM_WORKERS = 2

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")