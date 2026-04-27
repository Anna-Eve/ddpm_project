
# Il fait 3 choses :
#   1. Charger les images 
#   2. Entraîner le modèle à reconnaître le bruit
#   3. Sauvegarder le modèle toutes les N époques


from pyexpat import model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_P1.model import DPMModel
from data_P2.dataset import charger_f1


# ÉTAPE 0 — Réglages entrainement


CONFIG = {
    "racine":       "./data_f1",   # chemin vers ton dataset
    "image_size":   64,            # 64x64 px — bon compromis qualité/vitesse
    "channels":     3,             # couleur RGB
    "T":            1000,          # étapes de bruitage
    "batch_size":   32,            # réduis à 16 si manque de RAM
    "model_channels": 128,         # dimension de base du U-Net
    "num_res_blocks": 2,           # nombre de blocs résiduels
    "epochs":       30,
    "lr":           5e-5,
    "save_every":   5,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints/",
    "log_file":       "logs/loss.csv",
}

print(f"Entraînement sur : {CONFIG['device']}")
print(f"Dataset          : {CONFIG['racine']}")




# ÉTAPE 2 — Boucle d'entraînement 

def entrainer(model, optimizer, loader, config, epoch_depart=0):
    """
    La boucle principale.
    À chaque itération :
      - on prend une image propre
      - on lui ajoute une quantité de farine aléatoire
      - on demande au modèle de deviner cette farine
      - on mesure l'écart (loss)
      - on corrige les poids du modèle (backprop)
    """
    device  = config["device"]
    loss_fn = nn.MSELoss()
 
    # Créer les dossiers de sauvegarde
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
 
    # Ouvrir le CSV en append si reprise, sinon recréer
    csv_mode = "a" if epoch_depart > 0 else "w"
    with open(config["log_file"], csv_mode, newline="") as f:
        if epoch_depart == 0:
            csv.writer(f).writerow(["epoch", "loss_moyenne"])
 
    model.to(torch.device(device))
    model.train()
 
    epochs_total = epoch_depart + config["epochs"]
 
    for epoch in range(epoch_depart, epochs_total):
        loss_totale = 0.0
        nb_batches  = 0
 
        for images, _ in loader:
            # _ = on ignore les labels des écuries, on n'en a pas besoin
 
            images = images.to(device)
 
            # Choisir une étape de bruit aléatoire pour chaque image du batch
            t = torch.randint(0, config["T"], (images.shape[0],)).to(device)
 
            # Ajouter du bruit à l'image (travail de P1)
            images_bruitees, vrai_bruit = model.add_noise(images, t)
 
            # Demander au modèle de deviner le bruit
            bruit_predit = model(images_bruitees, t)
 
            # Mesurer l'erreur entre le vrai bruit et le bruit prédit
            loss = loss_fn(bruit_predit, vrai_bruit)
 
            # Rétropropagation — corriger les poids
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — évite les explosions lors d'une reprise
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
            loss_totale += loss.item()
            nb_batches  += 1
 
        # Moyenne de la loss sur toute l'époque
        loss_moyenne = loss_totale / nb_batches
        print(f"Époque {epoch+1:3d}/{epochs_total} — Loss : {loss_moyenne:.5f}")
 
        # Sauvegarder dans le CSV
        with open(config["log_file"], "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(loss_moyenne, 6)])
 
        # Sauvegarder le modèle toutes les N époques
        if (epoch + 1) % config["save_every"] == 0:
            chemin = os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch":       epoch + 1,
                "state_dict":  model.state_dict(),   # clé cohérente partout
                "optim_state": optimizer.state_dict(),
                "loss":        loss_moyenne,
                "config":      config,
            }, chemin)
            print(f"  Save → {chemin}")
 
    print("\nEntraînement terminé !")
    return model


# ÉTAPE 3 — Reprendre un entraînement interrompu


def charger_checkpoint(chemin, config):
    """
    Si l'entraînement a planté ou si tu veux continuer,
    cette fonction recharge le dernier checkpoint sauvegardé.
    """
    checkpoint = torch.load(chemin, map_location=config["device"])

    model = DPMModel(
        in_channels=config["channels"],
        model_channels=config["model_channels"],
        num_res_blocks=config["num_res_blocks"],
        timesteps=config["T"],)
    

    key = "state_dict" if "state_dict" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[key])
    model.to(torch.device(config["device"]))
 
    # Recréer et recharger l'optimizer — garde le momentum Adam accumulé
    optimizer = torch.optim.Adam(model.get_optimizer_params(), lr=config["lr"])
    optimizer.load_state_dict(checkpoint["optim_state"])
 
    epoch_depart = checkpoint["epoch"]
    print(f"  Reprise à l'époque {epoch_depart}, loss précédente : {checkpoint['loss']:.5f}")
    return model, optimizer, epoch_depart




if __name__ == "__main__":

    # Argument optionnel pour passer un checkpoint en ligne de commande
    # Usage : python train.py --checkpoint checkpoints/model_epoch_10.pth
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre l'entraînement")
    args = parser.parse_args()
 
    # Charger les données
    loader, infos = charger_f1(
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        racine=CONFIG["racine"],
    )
    print(f"\nDataset chargé : {infos['n_images']} images")
    print(f"Écuries         : {infos['ecuries']}\n")
 
    # Reprise depuis checkpoint OU départ depuis zéro
    if args.checkpoint:
        model, optimizer, epoch_depart = charger_checkpoint(args.checkpoint, CONFIG)
    else:
        print("Départ depuis zéro\n")
        model = DPMModel(
            in_channels=CONFIG["channels"],
            model_channels=CONFIG["model_channels"],
            num_res_blocks=CONFIG["num_res_blocks"],
            timesteps=CONFIG["T"],
            beta_start=0.0001,
            beta_end=0.02,
        )
        optimizer    = torch.optim.Adam(model.get_optimizer_params(), lr=CONFIG["lr"])
        epoch_depart = 0
 
    print(f"Modèle créé — {sum(p.numel() for p in model.parameters()):,} paramètres\n")
 
    # Lancer l'entraînement
    entrainer(model, optimizer, loader, CONFIG, epoch_depart)