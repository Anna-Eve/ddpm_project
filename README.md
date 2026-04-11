# ddpm_project


ddpm-projet/
├── model/               ← P1
│   ├── unet.py          # le cerveau du modèle
│   └── diffusion.py     # ajouter/enlever la farine
├── data/                ← P2
│   ├── dataset.py       # charger MNIST
│   └── train.py         # la boucle d'entraînement
├── sampling/            ← P3
│   ├── sample.py        # générer des images depuis du bruit
│   └── ddim.py          # version rapide (50 étapes)
├── eval/                ← P4
│   ├── metrics.py       # noter la qualité des images
│   └── visualize.py     # faire les grilles d'images / GIFs
├── train.py             # script principal — tout le monde
├── config.py            # tous les réglages (lr, T, batch…)
├── requirements.txt     # liste des librairies à installer
├── checkpoints/         # sauvegardes du modèle (auto)
├── samples/             # images générées (auto)
├── logs/                # courbes de loss (auto)
└── data/raw/            # MNIST téléchargé (auto)

# dependances
```shell
pip install torch torchvision matplotlib pytorch-fid imageio pillow
```
# Personne 4 
Mon rôle c'est l'évaluation et la visualisation — je suis la personne qui va mesurer si le modèle fonctionne bien et préparer la démo finale.
J'ai créé deux fichiers dans le dossier eval/ :
metrics.py qui contient les fonctions pour mesurer la qualité des images générées — des stats de base sur les pixels, les courbes de loss pour voir si l'entraînement se passe bien, et la préparation pour calculer le FID qui est la métrique standard pour les modèles génératifs.
visualize.py qui contient tout ce qui est visuel — les grilles d'images générées, la comparaison entre les vraies images MNIST et celles du modèle, et surtout le GIF de débruitage qui va montrer en temps réel le passage du bruit pur vers une image réaliste — c'est ça notre démo finale.
Pour l'instant j'ai tout testé avec des données simulées, ça fonctionne. Mes fonctions sont prêtes à recevoir les vraies données dès que P2 et P3 avancent — j'ai juste à brancher leurs sorties à la place du bruit aléatoire.
La prochaine étape c'est de lancer les premières vraies expériences dès qu'on a un modèle qui tourne, et de comparer différents hyperparamètres pour trouver la meilleure configuration

## metrics.py 
Fonctions pour mesurer la qualité des images générées par le modèle DDPM.
| Fonction | Ce qu'elle fait |
|---|---|
| `pixel_stats(images)` | Stats de base sur les pixels générés |
| `plot_loss_curve(loss_history)` | Courbe de loss brute |
| `plot_loss_curve_smoothed(...)` | Même chose mais lissée |
| `compare_experiments({...})` | Compare plusieurs runs sur un graphique |
| `save_images_for_fid(images, folder)` | Prépare les images pour calculer le FID |

## visualize.py
Fonctions pour visualiser les images générées par le modèle DDPM.
| Fonction | Ce qu'elle fait |
|---|---|
| `save_image_grid(images)` | Grille de toutes les images générées |
| `save_denoising_gif(frames)` | GIF bruit → image (ta démo finale) |
| `plot_denoising_steps(frames)` | Bande horizontale des étapes clés (parfait pour les slides) |
| `compare_real_vs_generated(real, gen)` | Deux lignes côte à côte : MNIST vs généré |
