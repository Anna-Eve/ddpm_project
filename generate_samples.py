"""
Script pour générer des images avec le modèle entraîné.
"""

import torch
import sys
import os
import glob

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from model_P1.model import DPMModel
from sampling_P3.sample import DDPMSampler
from eval_P4.visualize import save_image_grid

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
    
    # Charger le checkpoint le plus récent
    checkpoint_dir = "checkpoints/"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé dans", checkpoint_dir)
        return
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✓ Chargement du checkpoint : {latest_checkpoint}")
    
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
    
    print(f"✓ Modèle chargé — {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Créer le sampler
    betas = torch.linspace(0.0001, 0.02, CONFIG["T"])
    sampler = DDPMSampler(model, CONFIG["T"], betas, torch.device(CONFIG["device"]))
    
    # Générer les images
    print(f"\n🎨 Génération de {CONFIG['num_samples']} images...")
    shape = (CONFIG["num_samples"], CONFIG["channels"], CONFIG["image_size"], CONFIG["image_size"])
    samples = sampler.sample(shape)
    
    # Sauvegarder la grille
    os.makedirs("results", exist_ok=True)
    save_path = "results/generated_samples.png"
    save_image_grid(samples, save_path, nrow=4, title="Images générées par DDPM")
    
    print(f"✓ Grille sauvegardée → {save_path}")
    print("\n✓ Génération terminée !")

if __name__ == "__main__":
    main()
