"""
Interface Principal - Model.py

Boîte noire contenant :
1. Le U-Net (réseau de neurones qui prédit le bruit)
2. Le processus forward diffusion (ajouter du bruit)

Usage pour les autres équipes (P2, P3, P4) :
    from model import DPMModel
    model = DPMModel(...)
    x_t = model.add_noise(x_0, t)
    noise_pred = model(x_t, t)
"""

import torch
import torch.nn as nn
from typing import Optional
from .diffusion import GaussianDiffusion
from .unet import UNet


class DPMModel(nn.Module):
    """
    Modèle complet DDPM = U-Net + Forward Diffusion Process.
    
    C'est la classe que l'équipe d'entraînement (P2) va utiliser.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # 1 pour MNIST, 3 pour RGB
        model_channels: int = 128,
        num_res_blocks: int = 2,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            in_channels : 1 pour MNIST, 3 pour images RGB
            model_channels : dimension de base du U-Net
            num_res_blocks : nombre de blocs résiduels par niveau
            timesteps : nombre d'étapes de diffusion
            beta_start, beta_end : schedule des variances du bruit
            device : torch.device (cuda/cpu)
        """
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.timesteps = timesteps
        
        # Le U-Net : prédit le bruit dans une image bruyante
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
        ).to(self.device)
        
        # Forward diffusion process : ajoute du bruit contrôlé
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=self.device,
        )
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Ajoute du bruit à l'image x_0 à l'étape t.
        C'est ce qu'utilise l'équipe P2 dans la boucle d'entraînement.
        
        Args:
            x_0 : image originale [B, C, H, W], valeurs dans [-1, 1] ou [0, 1]
            t : timesteps [B], valeurs entre 0 et timesteps-1
            noise : bruit gaussien [B, C, H, W] (généré si None)
            
        Returns:
            (x_t, noise) : image bruyante et le bruit qui a été ajouté
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        
        x_t = self.diffusion.q_sample(x_0, t, noise)
        return x_t, noise
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Prédication du bruit dans l'image x au timestep t.
        
        Args:
            x : image bruyante [B, C, H, W]
            t : timesteps [B]
            
        Returns:
            noise_pred : prédiction du bruit [B, C, H, W]
        """
        return self.unet(x, t)
    
    def get_timesteps(self) -> int:
        """Retourne le nombre total de timesteps."""
        return self.timesteps
    
    def to(self, device: torch.device) -> "DPMModel":
        """Déplacer le modèle vers un dispositif (CPU/GPU)."""
        self.device = device
        self.unet = self.unet.to(device)
        self.diffusion = self.diffusion.to(device)
        return self
    
    def get_optimizer_params(self) -> list:
        """Retourne les paramètres à optimiser (pour P2 - entraînement)."""
        return self.unet.parameters()

    @torch.no_grad()
    def sample(self, batch_size: int, image_size: int | tuple[int, int] = 32) -> torch.Tensor:
        """Génère un batch d'images à partir de bruit pur."""
        if isinstance(image_size, int):
            height = image_size
            width = image_size
        else:
            height, width = image_size

        shape = (batch_size, self.in_channels, height, width)
        return self.diffusion.p_sample_loop(self.unet, shape, device=self.device)


# Exemple d'utilisation pour les autres équipes
if __name__ == "__main__":
    # P2 - Entraînement
    print("=== Exemple P2 : Entraînement ===")
    model = DPMModel(in_channels=1, timesteps=1000)  # MNIST
    
    # Batch d'images
    x_0 = torch.randn(4, 1, 32, 32)  # [B=4, C=1, H=32, W=32]
    t = torch.randint(0, 1000, (4,))  # Timesteps aléatoires
    
    # Ajouter du bruit (forward diffusion)
    x_t, noise_true = model.add_noise(x_0, t)
    print(f"x_0 shape: {x_0.shape}")
    print(f"x_t shape (bruitée): {x_t.shape}")
    print(f"noise_true shape: {noise_true.shape}")
    
    # Prédire le bruit (c'est ce qu'on entraîne)
    noise_pred = model(x_t, t)
    print(f"noise_pred shape: {noise_pred.shape}")
    
    # Loss d'entraînement (P2 va implémenter ça)
    loss = torch.nn.functional.mse_loss(noise_pred, noise_true)
    print(f"MSE Loss: {loss.item():.4f}")
