import torch
from tqdm import tqdm
import os
from torchvision.utils import save_image, make_grid

class DDIMSampler:
    def __init__(self, model, T, betas, device):
        """
        Sampler for DDIM (Denoising Diffusion Implicit Models).
        
        Args:
            model: The U-Net model epsilon_theta(x_t, t).
            T: Total number of timesteps the model was trained with.
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
        
    def _extract(self, a, t, x_shape):
        """Extract coefficients from a at index t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def sample(self, shape, ddim_steps=50, eta=0.0):
        """
        Complete sampling process using DDIM.
        
        Args:
            shape: Shape of the batch to generate (batch_size, C, H, W).
            ddim_steps: Number of sampling steps (can be much smaller than T).
            eta: Parameter for noise (0.0 for deterministic DDIM, 1.0 for DDPM-like).
        Returns:
            Generated images.
        """
        self.model.eval()
        batch_size = shape[0]
        
        # Select timesteps for DDIM sampling
        # We sample ddim_steps timesteps evenly spaced from 0 to T-1
        indices = torch.linspace(0, self.T - 1, ddim_steps).long().to(self.device)
        # We also need the previous timestep for each step
        indices_prev = torch.cat([torch.tensor([-1], device=self.device), indices[:-1]], dim=0)
        
        # Start from pure noise x_T ~ N(0, I)
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, ddim_steps)), desc='DDIM sampling', total=ddim_steps):
            t_idx = indices[i]
            t_prev_idx = indices_prev[i]
            
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Extract cumulative products for current and previous timesteps
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            
            if t_prev_idx == -1:
                alpha_cumprod_prev_t = torch.tensor(1.0, device=self.device)
            else:
                t_prev = torch.full((batch_size,), t_prev_idx, device=self.device, dtype=torch.long)
                alpha_cumprod_prev_t = self._extract(self.alphas_cumprod, t_prev, x.shape)
            
            # Predict noise epsilon_theta(x_t, t)
            epsilon_theta = self.model(x, t)
            
            # 1. Predict x_0 from x_t and epsilon_theta
            # Equation 12: pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * epsilon_theta) / torch.sqrt(alpha_cumprod_t)
            
            # 2. Compute the noise standard deviation sigma_t
            # Equation 16 in DDIM paper
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev_t))
            
            # 3. Compute direction pointing to x_t
            # Equation 12: sqrt(1 - alpha_bar_prev - sigma_t^2) * epsilon_theta
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t - sigma_t**2) * epsilon_theta
            
            # 4. Final step: x_{t-1} = sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * epsilon
            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = torch.sqrt(alpha_cumprod_prev_t) * pred_x0 + dir_xt + sigma_t * noise
            
        self.model.train()
        return x.clamp(-1, 1)

def save_image_grid(images, path, nrow=8):
    """Utility to save a grid of images."""
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)
