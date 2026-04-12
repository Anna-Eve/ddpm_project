
# Il fait 3 choses :
#   1. Charger les images 
#   2. Entraîner le modèle à reconnaître le bruit
#   3. Sauvegarder le modèle toutes les N époques


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import charger_f1

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
    "epochs":       50,
    "lr":           2e-4,
    "save_every":   5,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints/",
    "log_file":       "logs/loss.csv",
}

print(f"Entraînement sur : {CONFIG['device']}")
print(f"Dataset          : {CONFIG['racine']}")




# ÉTAPE 2 — Boucle d'entraînement 

def entrainer(model, loader, config):
    """
    La boucle principale.
    À chaque itération :
      - on prend une image propre
      - on lui ajoute une quantité de farine aléatoire
      - on demande au modèle de deviner cette farine
      - on mesure l'écart (loss)
      - on corrige les poids du modèle (backprop)
    """
    device    = config["device"]

    # Optimiseur — Adam est le standard, lr = vitesse d'apprentissage
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Loss — MSE = erreur quadratique moyenne entre vrai bruit et bruit prédit
    loss_fn = nn.MSELoss()

    # Créer les dossiers de sauvegarde
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Fichier de log pour suivre la loss
    with open(config["log_file"], "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss_moyenne"])

    model.to(torch.device(device))
    model.train()

    for epoch in range(config["epochs"]):
        loss_totale = 0.0
        nb_batches  = 0

        for images, _ in loader:
            # _ = on ignore les labels (0-9), on n'en a pas besoin

            images = images.to(device)                         # envoyer sur GPU si dispo

            # Choisir une étape de bruit aléatoire pour chaque image du batch
            t = torch.randint(0, config["T"], (images.shape[0],)).to(device)

            # Ajouter du bruit à l'image (travail de P1)
            images_bruitees, vrai_bruit = model.add_noise(images, t)

            # Demander au modèle de deviner le bruit (le modèle vient de P1)
            bruit_predit = model(images_bruitees, t)

            # Mesurer l'erreur entre le vrai bruit et le bruit prédit
            loss = loss_fn(bruit_predit, vrai_bruit)

            # Rétropropagation — corriger les poids
            optimizer.zero_grad()     # effacer les gradients précédents
            loss.backward()           # calculer les nouveaux gradients
            optimizer.step()          # mettre à jour les poids

            loss_totale += loss.item()
            nb_batches  += 1

        # Moyenne de la loss sur toute l'époque
        loss_moyenne = loss_totale / nb_batches
        print(f"Époque {epoch+1:3d}/{config['epochs']} — Loss : {loss_moyenne:.5f}")

        # Sauvegarder dans le CSV
        with open(config["log_file"], "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(loss_moyenne, 6)])

        # Sauvegarder le modèle toutes les N époques
        if (epoch + 1) % config["save_every"] == 0:
            chemin = os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch+1}.pth")
            
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
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
    

    model.load_state_dict(checkpoint["state_dict"])


    print(f"Checkpoint chargé — époque {checkpoint['epoch']}, loss {checkpoint['loss']:.5f}")
    return model




if __name__ == "__main__":

    # Charger les données
    loader, infos = charger_f1(
        batch_size=CONFIG["batch_size"],
        image_size=CONFIG["image_size"],
        racine=CONFIG["racine"],)

    print(f"\nDataset chargé : {infos['n_images']} images")
    print(f"Écuries         : {infos['ecuries']}\n")

    # Créer le modèle de P1
    model = DPMModel(
        in_channels=CONFIG["channels"],
        model_channels=CONFIG["model_channels"],
        num_res_blocks=CONFIG["num_res_blocks"],
        timesteps=CONFIG["T"],
        beta_start=0.0001,
        beta_end=0.02,
    )
 
    print(f"Modèle créé — {sum(p.numel() for p in model.parameters()):,} paramètres\n")
 
    # Lancer l'entraînement
    entrainer(model, loader, CONFIG)