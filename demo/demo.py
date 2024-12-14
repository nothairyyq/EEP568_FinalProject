"""
Demo script for CelebA Face Generation using DCGAN
This script demonstrates how to load and use the pre-trained GAN model
to generate face images. It provides a simple interface for users to
generate either a single image or a grid of multiple images.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from src.model import Generator
from src.config import *

class FaceGenerationDemo:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.generator = self._load_generator()
        
    def _load_generator(self):
        try:
            # 初始化生成器模型
            generator = Generator(LATENT_DIM, GEN_FEATURES, IMAGE_CHANNELS).to(self.device)
            
            # load pre-trainied model
            model_path = "checkpoints/final_generator.pth" # or place the google driver link
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "\nError: Pre-trained model not found!\n"
                    "Please ensure you have:\n"
                    "1. Downloaded the model from the provided link in README.md\n"
                    "2. Placed the 'final_generator.pth' file in the 'checkpoints' directory"
                )
            
            # load model
            generator.load_state_dict(torch.load(model_path))
            generator.eval()
            print("Successfully loaded pre-trained generator model")
            return generator
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate_single_face(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        noise = torch.randn(1, LATENT_DIM, 1, 1, device=self.device)
        with torch.no_grad():  
            generated_image = self.generator(noise)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"generated_face_{timestamp}.png")
        
        save_image(generated_image, output_path, normalize=True)
        print(f"Generated image saved to: {output_path}")
        
        return output_path

    def generate_face_grid(self, num_images=16, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        noise = torch.randn(num_images, LATENT_DIM, 1, 1, device=self.device)
        
        with torch.no_grad():
            generated_images = self.generator(noise)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"face_grid_{timestamp}.png")
        
        save_image(generated_images, output_path, normalize=True, nrow=int(np.sqrt(num_images)))
        print(f"Generated image grid saved to: {output_path}")
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Face Generation Demo")
    parser.add_argument("--mode", choices=["single", "grid"], default="grid",
                      help="Generate either a single face or a grid of faces")
    parser.add_argument("--num_images", type=int, default=16,
                      help="Number of images to generate in grid mode")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    demo = FaceGenerationDemo()
    
    if args.mode == "single":
        output_path = demo.generate_single_face(seed=args.seed)
    else:
        output_path = demo.generate_face_grid(num_images=args.num_images, seed=args.seed)
    
    print("\nDemo completed successfully!")
    print(f"Generated images have been saved to: {output_path}")
    print("\nTry different options by using command line arguments:")
    print("--mode single : Generate a single face")
    print("--mode grid : Generate a grid of faces")
    print("--num_images N : Specify number of images in grid")
    print("--seed N : Set random seed for reproducible results")

if __name__ == "__main__":
    main()