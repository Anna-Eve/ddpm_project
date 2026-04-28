import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from torchvision.utils import make_grid
from PIL import Image
import sys
sys.path.insert(0, ".")

# utilitaire interne
# def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
#     img = (tensor.clamp(-1, 1) + 1) / 2       # → [0, 1]
#     img = (img * 255).byte().cpu().numpy()     # → [0, 255]
#     if img.ndim == 3 and img.shape[0] == 3:
#         img = img.transpose(1, 2, 0)           # (3,H,W) → (H,W,3) pour RGB
#     elif img.ndim == 3 and img.shape[0] == 1:
#         img = img[0]                            # (1,H,W) → (H,W) pour gris
#     return img

def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)             # (1,3,64,64) → (3,64,64)
    img = (tensor.clamp(-1, 1) + 1) / 2
    img = (img * 255).byte().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)           # (3,H,W) → (H,W,3)
    elif img.ndim == 3 and img.shape[0] == 1:
        img = img[0]                            # (1,H,W) → (H,W)
    return img

# grille d'image générées
def save_image_grid(images: torch.Tensor,
                    save_path: str = "eval_P4/outputs/grid.png",
                    nrow: int = 8,
                    title: str = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    images_01 = (images.clamp(-1, 1) + 1) / 2      # → [0, 1]
    grid = make_grid(images_01, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()   # (C,H,W) → (H,W,C)

    plt.figure(figsize=(nrow * 1.5, (len(images) // nrow + 1) * 1.5))
    plt.imshow(grid_np.squeeze(), cmap="gray" if grid_np.shape[2] == 1 else None)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Grille sauvegardée → {save_path}")

# gif du processus de débruitage
def save_denoising_gif(frames: list,
                        save_path: str = "eval_P4/outputs/denoising.gif",
                        fps: int = 20,
                        n_frames_max: int = 100):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Sous-échantillonnage si trop de frames (ex: 1000 steps → 100 frames)
    if len(frames) > n_frames_max:
        indices = np.linspace(0, len(frames) - 1, n_frames_max, dtype=int)
        frames = [frames[i] for i in indices]

    gif_frames = []
    for f in frames:
        img = _to_uint8(f)
        # Agrandir pour que le GIF soit visible (MNIST 28×28 → 112×112)
        pil_img = Image.fromarray(img).resize((112, 112), Image.NEAREST)
        gif_frames.append(np.array(pil_img))

    imageio.mimsave(save_path, gif_frames, fps=fps, loop=0)
    print(f"[Visualize] GIF de débruitage sauvegardé → {save_path}  ({len(gif_frames)} frames, {fps} fps)")

# étapes clés du débruitage (image fixe)
def plot_denoising_steps(frames: list,
                          n_steps_shown: int = 10,
                          save_path: str = "eval_P4/outputs/denoising_steps.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    indices = np.linspace(0, len(frames) - 1, n_steps_shown, dtype=int)
    selected = [frames[i] for i in indices]

    fig, axes = plt.subplots(1, n_steps_shown, figsize=(n_steps_shown * 1.8, 2.5))
    for ax, frame, idx in zip(axes, selected, indices):
        img = _to_uint8(frame)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"t={len(frames)-1-idx}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Processus de débruitage — DDPM", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Étapes de débruitage sauvegardées → {save_path}")

# comparaison réel vs généré
def compare_real_vs_generated(real: torch.Tensor,
                               generated: torch.Tensor,
                               n: int = 8,
                               save_path: str = "eval_P4/outputs/real_vs_generated.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    real_01 = (real[:n].clamp(-1, 1) + 1) / 2
    gen_01  = (generated[:n].clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    for i in range(n):
        # axes[0, i].imshow(real_01[i].squeeze().cpu().numpy(), cmap="gray")
        axes[0, i].imshow(real_01[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis("off")
        # axes[1, i].imshow(gen_01[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1, i].imshow(gen_01[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Réel", fontsize=11, rotation=0, labelpad=40, va="center")
    axes[1, 0].set_ylabel("Généré", fontsize=11, rotation=0, labelpad=40, va="center")
    plt.suptitle("Images réelles vs images générées", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Comparaison sauvegardée → {save_path}")
