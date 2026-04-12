import torch
from tqdm import tqdm
import os
from torchvision.utils import save_image, make_grid

class DDPMSampler:
    def __init__(self, model, T, betas, device):
        """
        Sampler for DDPM (Denoising Diffusion Probabilistic Models).
        
        Args:
            model: The U-Net model epsilon_theta(x_t, t).
            T: Total number of timesteps.
            betas: The beta schedule (tensor of shape (T,)).
            device: torch.device.
        """
        self.model = model
        self.T = T
        self.betas = betas.to(device)
        self.device = device
        
        # Precompute constants
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        
        # Precompute terms for sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """Extract coefficients from a at index t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Single step of reverse diffusion (sampling from p(x_{t-1} | x_t)).
        """
        # Predict noise epsilon_theta(x_t, t)
        epsilon_theta = self.model(x, t)
        
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)
        
        # Calculation of mean: Equation 11 in DDPM paper
        mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * epsilon_theta)
        
        if t[0] == 0:
            return mean
        else:
            # Add noise z ~ N(0, I)
            noise = torch.randn_like(x)
            variance = self._extract(self.posterior_variance, t, x.shape)
            # Re-parameterization trick for the reverse process
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, shape):
        """
        Complete sampling process (Algorithm 2 from DDPM paper).
        
        Args:
            shape: Shape of the batch to generate (batch_size, C, H, W).
        Returns:
            Generated images.
        """
        self.model.eval()
        batch_size = shape[0]
        # Start from pure noise x_T ~ N(0, I)
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.T)), desc='DDPM sampling', total=self.T):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
            
        self.model.train()
        # Clip images to [-1, 1] range as per common DDPM practice
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_with_steps(self, shape, n_steps=10):
        """
        Sample and return intermediate steps for visualization.
        """
        self.model.eval()
        batch_size = shape[0]
        x = torch.randn(shape, device=self.device)
        steps = []
        # Save every T // n_steps
        save_indices = torch.linspace(0, self.T - 1, n_steps).long()
        
        for i in tqdm(reversed(range(0, self.T)), desc='DDPM sampling steps', total=self.T):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
            if i in save_indices:
                steps.append(x.cpu().clamp(-1, 1))
                
        self.model.train()
        return steps

def save_image_grid(images, path, nrow=8):
    """Utility to save a grid of images."""
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)
