"""
Variational Autoencoder (VAE) with CNN architecture
Based on the original VAE paper: https://arxiv.org/abs/1312.6114

This is Step 1: As-is SSL Comparison
Following the original implementation with CNN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_CLAMP = (-10.0, 10.0)


class VAE_CNN(nn.Module):
    """
    Variational Autoencoder with Convolutional architecture
    Adaptive for both small (28x28 MNIST) and large (224x224 ImageNet) images
    """
    
    def __init__(self, latent_dim=512, input_channels=3, image_size=224):
        """
        Args:
            latent_dim: Dimension of the latent space (default: 512)
            input_channels: Number of input channels (default: 3 for RGB)
            image_size: Input image size (default: 224)
        """
        super(VAE_CNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Choose architecture based on image size
        if image_size == 28:
            # MNIST-specific architecture (28x28)
            self._build_mnist_architecture()
        else:
            # ImageNet architecture (224x224)
            self._build_imagenet_architecture()
    
    def _build_mnist_architecture(self):
        """Build CNN architecture for 28x28 images (MNIST)"""
        # Encoder for 28x28 images
        self.encoder = nn.Sequential(
            # Input: 3 x 28 x 28
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),  # 32 x 14 x 14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the flattened dimension
        self.flatten_dim = 128 * 7 * 7  # 6272
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)
        
        # Decoder for 28x28 images
        self.decoder = nn.Sequential(
            # Input: 128 x 7 x 7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 x 14 x 14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3 x 28 x 28
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.spatial_size = 7
    
    def _build_imagenet_architecture(self):
        """Build CNN architecture for 224x224 images (ImageNet)"""
        # Encoder for 224x224 images
        self.encoder = nn.Sequential(
            # Input: 3 x 224 x 224
            nn.Conv2d(self.input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the flattened dimension
        self.flatten_dim = 512 * 7 * 7  # 25088
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(self.latent_dim, self.flatten_dim)
        
        # Decoder for 224x224 images
        self.decoder = nn.Sequential(
            # Input: 512 x 7 x 7
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=4, stride=2, padding=1),  # 3 x 224 x 224
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.spatial_size = 7
    
    def encode(self, x):
        """
        Encode input to latent space
        Returns mu and logvar for the latent distribution
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * sigma
        """
        logvar = torch.clamp(logvar, min=LOGVAR_CLAMP[0], max=LOGVAR_CLAMP[1])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction
        """
        h = self.decoder_input(z)
        # Reshape based on architecture
        if self.image_size == 28:
            h = h.view(h.size(0), 128, self.spatial_size, self.spatial_size)
        else:
            h = h.view(h.size(0), 512, self.spatial_size, self.spatial_size)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x):
        """
        Forward pass through VAE
        Returns reconstruction, input, mu, and logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, x, mu, logvar
    
    def get_features(self, x):
        """
        Extract features for downstream tasks (linear evaluation)
        Returns the latent representation (mu)
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon_x, x, mu, logvar, beta=1.0, logvar_clamp=LOGVAR_CLAMP):
    """
    VAE loss function: Reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    # Using BCE with logits for numerical stability
    # But since we use Sigmoid in decoder, we use MSE instead
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2).
    # Clamp log variance before exp(); the previous unbounded exp() path
    # produced fully-NaN Step 1 checkpoints on ImageNet.
    logvar = torch.clamp(logvar, min=logvar_clamp[0], max=logvar_clamp[1])
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Testing MNIST architecture (28x28)")
    print("=" * 60)
    model_mnist = VAE_CNN(latent_dim=20, image_size=28).to(device)
    
    # Test forward pass for MNIST
    x_mnist = torch.randn(4, 3, 28, 28).to(device)
    recon_mnist, _, mu_mnist, logvar_mnist = model_mnist(x_mnist)
    
    print(f"Input shape: {x_mnist.shape}")
    print(f"Reconstruction shape: {recon_mnist.shape}")
    print(f"Latent mu shape: {mu_mnist.shape}")
    print(f"Latent logvar shape: {logvar_mnist.shape}")
    
    # Test loss
    loss, recon_loss, kl_div = vae_loss(recon_mnist, x_mnist, mu_mnist, logvar_mnist)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_div.item():.4f}")
    
    # Test feature extraction
    features = model_mnist.get_features(x_mnist)
    print(f"Features shape: {features.shape}")
    print("✓ MNIST architecture test passed!\n")
    
    print("=" * 60)
    print("Testing ImageNet architecture (224x224)")
    print("=" * 60)
    model_imagenet = VAE_CNN(latent_dim=512, image_size=224).to(device)
    
    # Test forward pass for ImageNet
    x_imagenet = torch.randn(4, 3, 224, 224).to(device)
    recon_imagenet, _, mu_imagenet, logvar_imagenet = model_imagenet(x_imagenet)
    
    print(f"Input shape: {x_imagenet.shape}")
    print(f"Reconstruction shape: {recon_imagenet.shape}")
    print(f"Latent mu shape: {mu_imagenet.shape}")
    print(f"Latent logvar shape: {logvar_imagenet.shape}")
    
    # Test loss
    loss, recon_loss, kl_div = vae_loss(recon_imagenet, x_imagenet, mu_imagenet, logvar_imagenet)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_div.item():.4f}")
    
    # Test feature extraction
    features = model_imagenet.get_features(x_imagenet)
    print(f"Features shape: {features.shape}")
    print("✓ ImageNet architecture test passed!\n")
    
    print("=" * 60)
    print("✓ All model tests passed!")
    print("=" * 60)
