"""
Script pour générer des images avec le modèle entraîné.
"""
import torch
import sys
import os
import glob
import csv

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from model_P1.model import DPMModel
from sampling_P3.ddim import DDIMSampler
from eval_P4.visualize import save_image_grid, save_denoising_gif, plot_denoising_steps, compare_real_vs_generated
from eval_P4.metrics import pixel_stats, plot_loss_curve, save_images_for_fid
from data_P2.dataset import charger_f1

# Configuration
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_channels": 128,
    "num_res_blocks": 2,
    "channels": 3,
    "T": 1000,
    "image_size": 64,
    "num_samples": 16,  # Générer 16 images
}

def main():
    print(f"Génération sur : {CONFIG['device']}")

    # Charger le checkpoint
    checkpoint_dir = "checkpoints/"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        # cherche à la racine aussi
        checkpoints = glob.glob("*.pth")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✓ Chargement : {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=CONFIG["device"])

    # Créer le modèle
    model = DPMModel(
        in_channels=CONFIG["channels"],
        model_channels=CONFIG["model_channels"],
        num_res_blocks=CONFIG["num_res_blocks"],
        timesteps=CONFIG["T"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(CONFIG["device"])
    model.eval()

    betas = torch.linspace(0.0001, 0.02, CONFIG["T"])
    sampler = DDIMSampler(model, CONFIG["T"], betas, torch.device(CONFIG["device"]))

    os.makedirs("results", exist_ok=True)

    # 1. Générer les images
    print(f"\n Génération de {CONFIG['num_samples']} images...")
    shape = (CONFIG["num_samples"], CONFIG["channels"], CONFIG["image_size"], CONFIG["image_size"])
    samples = sampler.sample(shape, ddim_steps=200)

    # 2. Grille d'images
    save_image_grid(samples, "results/generated_samples.png", nrow=4, title="Images générées par DDPM")

    # 3. Métriques
    pixel_stats(samples)
    save_images_for_fid(samples, "results/fid_generated/")

    # 4. Courbe de loss
    if os.path.exists("logs/loss.csv"):
        with open("logs/loss.csv", "r") as f:
            rows = list(csv.DictReader(f))
            loss_history = [float(r["loss_moyenne"]) for r in rows]
        plot_loss_curve(loss_history, "results/loss_curve.png")
        print("✓ Courbe de loss sauvegardée")

    # 5. Comparaison réel vs généré (si les données F1 sont disponibles)
    if os.path.exists("./data_f1/train"):
        loader, _ = charger_f1(batch_size=8, image_size=CONFIG["image_size"])
        real, _ = next(iter(loader))
        compare_real_vs_generated(real, samples[:8], n=8, save_path="results/real_vs_generated.png")
        print("✓ Comparaison sauvegardée")
    else:
        print("⚠ data_f1/ introuvable — comparaison ignorée")

    print("\n✓ Terminé ! Regarde dans results/")

if __name__ == "__main__":
    main()