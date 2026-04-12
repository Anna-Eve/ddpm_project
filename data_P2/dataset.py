
# Ce fichier gère TOUT ce qui touche aux données.
# Il supporte plusieurs datasets — change juste DATASET_CHOISI
# en bas du fichier pour switcher.

# Datasets disponibles :
#   - "mnist"         chiffres 28x28          (le plus léger)
#   - "fashion"       vêtements 28x28         (un peu plus dur)
#   - "cifar10"       photos couleur 32x32    (plus ambitieux)
#   - "data_f1"       image de F1             (voir bas du fichier) 


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os

 
ECURIES = [
    "alfaromeo", "alphatauri", "alpine", "astonmartin",
    "ferrari", "haas", "mclaren", "mercedes", "redbull", "williams"
]

# TRANSFORMATIONS — préparer les images pour le modèle
def get_transform(image_size, channels):
    """
    Prépare les images :
      - redimensionner à la bonne taille
      - convertir en tensor
      - normaliser entre -1 et 1 (obligatoire pour DDPM)
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    # Normalisation : 0.5 moyenne, 0.5 écart-type → valeurs entre -1 et 1
    if channels == 1:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

def get_transform_f1(image_size=64, mode="train"):
    """
    Augmentation agressive pour compenser le petit dataset F1.
    images x ces transformations = ~10 000 variations vues
    par le modèle au fil de l'entraînement.

        Deux modes :
      - "train" : augmentation activée
      - "eval"  : pas d'augmentation (pour la génération)
    """

    if mode == "eval":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.3,    # les couleurs des livrées changent
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # léger décalage
            scale=(0.9, 1.1)         # léger zoom
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



# DATASETS PRÊTS À L'EMPLOI
def charger_mnist(batch_size, image_size=28):
    """
    MNIST — 60 000 chiffres manuscrits en niveaux de gris.
    Le plus léger, idéal pour tester que tout fonctionne.
    """
    transform = get_transform(image_size, channels=1)
    dataset   = datasets.MNIST(
        root="./data/raw", train=True,
        download=True, transform=transform
    )
    return _make_loader(dataset, batch_size), {"channels": 1, "image_size": image_size}


def charger_fashion_mnist(batch_size, image_size=28):
    """
    Fashion-MNIST — 60 000 images de vêtements (t-shirts, chaussures...).
    Même format que MNIST mais plus visuellement intéressant.
    10 classes : t-shirt, pantalon, pull, robe, manteau,
                 sandale, chemise, basket, sac, bottine
    """
    transform = get_transform(image_size, channels=1)
    dataset   = datasets.FashionMNIST(
        root="./data/raw", train=True,
        download=True, transform=transform
    )
    return _make_loader(dataset, batch_size), {"channels": 1, "image_size": image_size}


def charger_cifar10(batch_size, image_size=32):
    """
    CIFAR-10 — 60 000 photos couleur (avions, voitures, chats...).
    Plus lourd : nécessite un U-Net plus grand et plus de temps.
    À tenter seulement si MNIST/Fashion marchent bien.
    """
    transform = get_transform(image_size, channels=3)
    dataset   = datasets.CIFAR10(
        root="./data/raw", train=True,
        download=True, transform=transform
    )
    return _make_loader(dataset, batch_size), {"channels": 3, "image_size": image_size}



# DATASET F1
class F1Dataset(Dataset):
    """
    Lit le dossier data_f1/train/ (ou test/) avec ses sous-dossiers
    d'écuries, et applique l'augmentation à chaque image.
 
    On utilise ImageFolder de torchvision qui gère automatiquement
    la structure sous-dossier → label.
    """
 
    def __init__(self, racine="./data_f1", split="train",
                 image_size=64, mode="train"):
 
        self.transform = get_transform_f1(image_size, mode)
        dossier        = os.path.join(racine, split)
 
        if not os.path.exists(dossier):
            raise ValueError(
                f"Dossier introuvable : '{dossier}'\n"
                f"Vérifie que data_f1/{split}/ existe bien."
            )
 
        # ImageFolder lit automatiquement les sous-dossiers
        # chaque sous-dossier = une classe (écurie)
        self.dataset = datasets.ImageFolder(
            root=dossier,
            transform=self.transform
        )
 
        print(f"F1Dataset [{split}] : {len(self.dataset)} images")
        print(f"  Écuries trouvées : {self.dataset.classes}")
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # On retourne le label aussi — utile pour P4 si elle veut
        # faire de la génération conditionnelle par écurie
        return image, label

def charger_f1(batch_size=8, image_size=64, racine="./data_f1"):
    """
    Charge le split train avec augmentation.
    Retourne (loader, infos).
    """
    dataset = F1Dataset(
        racine=racine,
        split="train",
        image_size=image_size,
        mode="train"
    )
 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
 
    infos = {
        "channels":   3,
        "image_size": image_size,
        "n_images":   len(dataset),
        "n_ecuries":  len(ECURIES),
        "ecuries":    ECURIES,
    }
 
    return loader, infos
 
 
def charger_f1_test(batch_size=32, image_size=64, racine="./data_f1"):
    """
    Charge le split test SANS augmentation — pour évaluation (P4).
    """
    dataset = F1Dataset(
        racine=racine,
        split="test",
        image_size=image_size,
        mode="eval"
    )
 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
 
    return loader

# UTILITAIRE INTERNE

def _make_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,      # évite les batches incomplets en fin d'époque
    )


# ============================================================
# FONCTION PRINCIPALE — change juste cette ligne pour switcher
# ============================================================

DATASET_CHOISI = "custom"    # "mnist" | "fashion" | "cifar10" | "custom"

def charger_dataset(batch_size):
    """
    Point d'entrée unique pour train.py.
    Retourne (loader, infos) où infos contient channels et image_size.
    """
    if DATASET_CHOISI == "mnist":
        return charger_mnist(batch_size)

    elif DATASET_CHOISI == "fashion":
        return charger_fashion_mnist(batch_size)

    elif DATASET_CHOISI == "cifar10":
        return charger_cifar10(batch_size)

    elif DATASET_CHOISI == "custom":
        # Mets le chemin vers ton dossier d'images ici
        return charger_f1_test(
            batch_size=batch_size,
            image_size=64,
        )

    else:
        raise ValueError(f"Dataset inconnu : '{DATASET_CHOISI}'")


# TEST RAPIDE
if __name__ == "__main__":
    print(f"Test du dataset : {DATASET_CHOISI}")
    loader, infos = charger_dataset(batch_size=32)

    print(f"Infos : {infos}")
    print(f"Nombre de batches : {len(loader)}")

    # Vérifier la forme d'un batch
    images, labels = next(iter(loader))
    print(f"Forme d'un batch : {images.shape}")
    print(f"Min / Max des pixels : {images.min():.2f} / {images.max():.2f}")
    print("OK — le dataset fonctionne correctement")