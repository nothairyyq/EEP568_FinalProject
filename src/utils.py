import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


def get_dataloader(data_root, image_size, batch_size, num_workers):
    """Create data loader for CelebA dataset.
    
    This function sets up the data pipeline for the CelebA dataset, including
    image transformations and loading configurations.
    
    Args:
        data_root (str): Path to the root directory of CelebA dataset
        image_size (int): Target size for the images (both height and width)
        batch_size (int): Number of images per batch
        num_workers (int): Number of subprocesses for data loading
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the CelebA dataset
    """
    # Define the image transformations pipeline
    transform = transforms.Compose([
        # Resize images to the specified size while maintaining aspect ratio
        transforms.Resize(image_size),
        # Center crop to make images square
        transforms.CenterCrop(image_size),
        # Convert PIL images to tensors
        transforms.ToTensor(),
        # Normalize the images to [-1, 1] range, which is optimal for tanh activation
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        # Create the dataset using ImageFolder
        # ImageFolder expects a root directory with subdirectories for each class
        dataset = torchvision.datasets.ImageFolder(
            root=data_root,
            transform=transform
        )
        
        # Create and return the DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle data for each epoch
            num_workers=num_workers,
            pin_memory=True  # Faster data transfer to GPU
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} images")
        return dataloader
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def save_samples(generator, fixed_noise, epoch, output_dir):
    """Save generated samples during training.
    
    This function generates and saves sample images using the current state
    of the generator. It's useful for monitoring training progress and 
    creating final results.
    
    Args:
        generator (nn.Module): The trained/training generator model
        fixed_noise (torch.Tensor): Fixed noise vectors for consistent sampling
        epoch (int): Current training epoch number
        output_dir (str): Directory to save the generated images
        
    Returns:
        str: Path to the saved image file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set generator to evaluation mode
        generator.eval()
        
        with torch.no_grad():  # No need to track gradients for sampling
            # Generate fake images
            fake_images = generator(fixed_noise)
            
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output filename
            output_file = os.path.join(output_dir, f'fake_samples_epoch_{epoch}.png')
            
            # Save the generated images as a grid
            save_image(
                fake_images,
                output_file,
                normalize=True,
                nrow=8,  # Number of images per row in the grid
                padding=2
            )
            
            # Also save the latest samples separately for easy access
            latest_file = os.path.join(output_dir, 'latest_samples.png')
            save_image(
                fake_images,
                latest_file,
                normalize=True,
                nrow=8,
                padding=2
            )
            
            print(f"Samples saved to {output_file}")
            return output_file
            
    except Exception as e:
        print(f"Error saving samples: {str(e)}")
        raise
    finally:
        # Return generator to training mode
        generator.train()