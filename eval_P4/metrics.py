import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# statistiques
def pixel_stats(images: torch.Tensor) -> dict:
    stats = {
        "mean": images.mean().item(),
        "std":  images.std().item(),
        "min":  images.min().item(),
        "max":  images.max().item(),
    }
    print(f"[Metrics] Pixel stats → mean={stats['mean']:.4f}, "
          f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    return stats

# courbe de loss
def plot_loss_curve(loss_history: list, save_path: str = "eval_P4/outputs/loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, linewidth=1.2, color="#4C72B0")
    plt.xlabel("Étape d'entraînement")
    plt.ylabel("Loss (MSE sur le bruit)")
    plt.title("Courbe d'entraînement — DDPM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Courbe de loss sauvegardée → {save_path}")

def plot_loss_curve_smoothed(loss_history: list, window: int = 50,
                              save_path: str = "eval_P4/outputs/loss_curve_smooth.png"): # moyenne glissante pour lisser la courbe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    loss_array = np.array(loss_history)
    # Moyenne glissante
    smoothed = np.convolve(loss_array, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(loss_array, alpha=0.3, color="#4C72B0", linewidth=0.8, label="Brut")
    plt.plot(range(window - 1, len(loss_array)), smoothed,
             color="#4C72B0", linewidth=2, label=f"Lissé (fenêtre={window})")
    plt.xlabel("Étape d'entraînement")
    plt.ylabel("Loss (MSE sur le bruit)")
    plt.title("Courbe d'entraînement lissée — DDPM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Courbe lissée sauvegardée → {save_path}")

# sauvegarde d'image pour le fid
def save_images_for_fid(images: torch.Tensor, folder: str):
    os.makedirs(folder, exist_ok=True)
    images_01 = (images.clamp(-1, 1) + 1) / 2  # remise en [0, 1]

    for i, img in enumerate(images_01):
        save_image(img, os.path.join(folder, f"img_{i:05d}.png"))

    print(f"[Metrics] {len(images)} images sauvegardées dans '{folder}'")
    print(f"[Metrics] Pour calculer le FID :")
    print(f"          python -m pytorch_fid eval_P4/fid_data/real/ {folder}")

# comparaison d'expériences
def compare_experiments(experiments: dict, save_path: str = "eval_P4/outputs/compare_exp.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    for label, loss_history in experiments.items():
        plt.plot(loss_history, linewidth=1.5, label=label)

    plt.xlabel("Étape d'entraînement")
    plt.ylabel("Loss")
    plt.title("Comparaison des expériences — DDPM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Comparaison sauvegardée → {save_path}")

# tests sur données simulées
if __name__ == "__main__":
    print("Test de metrics.py avec données simulées\n")

    # Simule un batch d'images générées
    fake_images = torch.randn(64, 1, 28, 28)
    pixel_stats(fake_images)

    # Simule une loss qui descend
    fake_loss = [1.0 * np.exp(-i / 500) + 0.05 * np.random.randn()
                 for i in range(2000)]
    plot_loss_curve(fake_loss, "eval_P4/outputs/loss_curve.png")
    plot_loss_curve_smoothed(fake_loss, window=50, save_path="eval_P4/outputs/loss_curve_smooth.png")

    # Simule deux expériences
    loss_exp1 = [0.8 * np.exp(-i / 400) + 0.03 * np.random.randn() for i in range(2000)]
    loss_exp2 = [0.9 * np.exp(-i / 700) + 0.03 * np.random.randn() for i in range(2000)]
    compare_experiments({"lr=1e-3": loss_exp1, "lr=1e-4": loss_exp2})

    # Sauvegarde images pour FID
    save_images_for_fid(fake_images, "eval_P4/fid_data/generated/")

    print("\nTous les tests ont passé.")