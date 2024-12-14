import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from src.model import Generator, Discriminator
from src.utils import weights_init, get_dataloader, save_samples
from src.config import *

def setup_training():
    """
    Initialize all necessary components for training.
    This includes creating models, optimizers, and criterion.
    
    Returns:
        tuple: Contains all initialized components needed for training
    """
    # Create the generator and apply custom weights initialization
    netG = Generator(LATENT_DIM, GEN_FEATURES, IMAGE_CHANNELS).to(DEVICE)
    netG.apply(weights_init)
    print("Generator created:")
    print(netG)

    # Create the discriminator and apply custom weights initialization
    netD = Discriminator(IMAGE_CHANNELS, DISC_FEATURES).to(DEVICE)
    netD.apply(weights_init)
    print("\nDiscriminator created:")
    print(netD)

    # Initialize loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    return netG, netD, criterion, optimizerG, optimizerD, fixed_noise

def train_discriminator(netD, netG, real_batch, criterion, optimizerD, device):
    """
    Perform one training step for the discriminator.
    The discriminator is trained to distinguish between real and fake images.
    
    Returns:
        tuple: Contains discriminator loss and intermediate results
    """
    batch_size = real_batch.size(0)
    
    # Reset gradients
    netD.zero_grad()
    
    # Train with real batch
    label_real = torch.full((batch_size,), 1.0, device=device)
    output_real = netD(real_batch).view(-1)
    errD_real = criterion(output_real, label_real)
    errD_real.backward()
    D_x = output_real.mean().item()

    # Train with fake batch
    noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
    fake = netG(noise)
    label_fake = torch.full((batch_size,), 0.0, device=device)
    output_fake = netD(fake.detach()).view(-1)
    errD_fake = criterion(output_fake, label_fake)
    errD_fake.backward()
    D_G_z1 = output_fake.mean().item()
    
    # Compute total discriminator loss
    errD = errD_real + errD_fake
    optimizerD.step()
    
    return errD, D_x, D_G_z1

def train_generator(netD, netG, criterion, optimizerG, batch_size, device):
    """
    Perform one training step for the generator.
    The generator is trained to produce images that fool the discriminator.
    
    Returns:
        tuple: Contains generator loss and results
    """
    netG.zero_grad()
    
    # Generate fake images
    noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
    fake = netG(noise)
    
    # Since we just updated D, perform another forward pass of all-fake batch through D
    label = torch.full((batch_size,), 1.0, device=device)  # fake labels are real for generator cost
    output = netD(fake).view(-1)
    
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    
    optimizerG.step()
    
    return errG, D_G_z2

def save_training_progress(G_losses, D_losses, output_dir):
    """
    Save training metrics and plots.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def train():
    """
    Main training function that orchestrates the entire training process.
    """
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Setup all training components
    netG, netD, criterion, optimizerG, optimizerD, fixed_noise = setup_training()
    
    # Get the data loader
    dataloader = get_dataloader(DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

    # Lists to track progress
    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        # Progress bar for each epoch
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (real_batch, _) in pbar:
            real_batch = real_batch.to(DEVICE)
            
            # Train Discriminator
            errD, D_x, D_G_z1 = train_discriminator(
                netD, netG, real_batch, criterion, optimizerD, DEVICE
            )

            # Train Generator
            errG, D_G_z2 = train_generator(
                netD, netG, criterion, optimizerG, real_batch.size(0), DEVICE
            )

            # Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Update progress bar
            pbar.set_description(
                f"[{epoch}/{NUM_EPOCHS}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
            )

            # Save sample images periodically
            if i % 500 == 0:
                save_samples(netG, fixed_noise, epoch, "results")

        # Save model checkpoints after each epoch
        torch.save({
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'epoch': epoch,
        }, f'checkpoints/model_epoch_{epoch}.pth')

    # Save final training progress
    save_training_progress(G_losses, D_losses, "results")
    print("Training finished!")

if __name__ == "__main__":
    train()