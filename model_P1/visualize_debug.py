"""
Outils de debug pour visualiser des images générées par le DDPM.

Ce fichier est volontairement séparé du code d'evaluation afin de servir
uniquement au debug rapide pendant le développement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import math

import matplotlib.pyplot as plt
import torch

from model import DPMModel


def _to_batch(images: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """Convertit une image ou une séquence d'images en batch [B, C, H, W]."""
    if isinstance(images, torch.Tensor):
        batch = images
    else:
        batch = torch.stack(list(images), dim=0)

    if batch.dim() == 3:
        batch = batch.unsqueeze(0)

    if batch.dim() != 4:
        raise ValueError("images doit avoir la forme [B, C, H, W], [C, H, W] ou une sequence de tenseurs.")

    return batch.detach().cpu()


def _unnormalize(batch: torch.Tensor) -> torch.Tensor:
    """Ramene des images normalisees dans [-1, 1] vers [0, 1]."""
    return ((batch + 1.0) / 2.0).clamp(0.0, 1.0)


def visualize_generated_images(
    images: Union[torch.Tensor, Sequence[torch.Tensor]],
    nrow: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    unnormalize: bool = True,
    figsize: Optional[tuple[int, int]] = None,
):
    """
    Affiche une grille d'images generees.

    Args:
        images: tenseur [B, C, H, W], [C, H, W] ou sequence de tenseurs.
        nrow: nombre d'images par ligne. Si None, une grille quasi carree est choisie.
        title: titre de la figure.
        save_path: chemin de sauvegarde optionnel.
        show: affiche la figure avec plt.show().
        unnormalize: remet les valeurs de [-1, 1] vers [0, 1].
        figsize: taille explicite de la figure.

    Returns:
        La figure matplotlib creee.
    """
    batch = _to_batch(images)

    if unnormalize:
        batch = _unnormalize(batch)

    batch = batch.clamp(0.0, 1.0)

    n_images = batch.shape[0]
    if n_images == 0:
        raise ValueError("Aucune image a afficher.")

    if nrow is None:
        nrow = max(1, int(math.sqrt(n_images)))

    ncol = min(nrow, n_images)
    nrow_fig = math.ceil(n_images / ncol)

    if figsize is None:
        figsize = (ncol * 2.5, nrow_fig * 2.5)

    fig, axes = plt.subplots(nrow_fig, ncol, figsize=figsize)

    if n_images == 1:
        axes_list = [axes]
    elif hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    for idx, ax in enumerate(axes_list):
        ax.axis("off")
        if idx >= n_images:
            continue

        image = batch[idx]
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image.permute(1, 2, 0))

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation debug des images generees par DDPM")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chemin vers un checkpoint .pth/.pt")
    parser.add_argument("--batch-size", type=int, default=8, help="Nombre d'images a generer")
    parser.add_argument("--image-size", type=int, default=32, help="Taille des images generees")
    parser.add_argument("--channels", type=int, default=1, help="Nombre de canaux de sortie")
    parser.add_argument("--timesteps", type=int, default=1000, help="Nombre d'etapes de diffusion")
    parser.add_argument("--no-show", action="store_true", help="N'ouvre pas la fenetre matplotlib")
    parser.add_argument("--save-path", type=str, default=None, help="Chemin de sauvegarde optionnel")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPMModel(in_channels=args.channels, timesteps=args.timesteps, device=device)

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=False)

    generated_images = model.sample(batch_size=args.batch_size, image_size=args.image_size)

    visualize_generated_images(
        generated_images,
        title="Debug DDPM - images generees",
        save_path=args.save_path,
        show=not args.no_show,
    )
