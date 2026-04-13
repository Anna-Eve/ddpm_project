# ddpm_project
```
DDPM_PROJECT/
├── model_P1/                  ← P1
│   └── model.py               # le cerveau du modèle (U-Net)
├── data_P2/                   ← P2
│   ├── dataset.py             # charger les images F1
│   └── train.py               # la boucle d'entraînement
├── sampling_P3/               ← P3
│   ├── sample.py              # générer des images depuis du bruit (1000 étapes)
│   └── ddim.py                # version rapide (50 étapes)
├── eval_P4/                   ← P4
│   ├── metrics.py             # mesurer la qualité des images
│   └── visualize.py           # faire les grilles d'images / GIFs
├── generate_samples.py        # script principal de génération — P4
├── train.py                   # script d'entraînement — tout le monde
├── model_epoch_50.pth         # checkpoint du modèle entraîné
├── logs/                      # courbes de loss (auto)
│   └── loss.csv
└── results/                   # images générées (auto)
    ├── generated_samples.png
    ├── loss_curve.png
    └── real_vs_generated.png
```

## Dépendances
```shell
pip install torch torchvision matplotlib pytorch-fid imageio pillow
```

## Personne 4
Mon rôle c'est l'évaluation et la visualisation — je suis la personne qui mesure si le modèle fonctionne bien et qui prépare la démo finale.
J'ai créé deux fichiers dans le dossier `eval_P4/` et un script principal `generate_samples.py` :
`metrics.py` contient les fonctions pour mesurer la qualité des images générées — des stats de base sur les pixels, les courbes de loss pour voir si l'entraînement se passe bien, et la préparation pour calculer le FID qui est la métrique standard pour les modèles génératifs.
`visualize.py` contient tout ce qui est visuel — les grilles d'images générées, la comparaison entre les vraies images F1 et celles du modèle, et le GIF de débruitage qui montre en temps réel le passage du bruit pur vers une image réaliste.
`generate_samples.py` orchestre tout — il charge le modèle de P1, appelle le sampler de P3, et utilise mes fonctions de `metrics.py` et `visualize.py` pour produire les résultats finaux dans `results/`.

## `metrics.py`
Fonctions pour mesurer la qualité des images générées par le modèle DDPM.

| Fonction | Ce qu'elle fait |
|---|---|
| `pixel_stats(images)` | Stats de base sur les pixels générés |
| `plot_loss_curve(loss_history)` | Courbe de loss brute |
| `plot_loss_curve_smoothed(...)` | Même chose mais lissée |
| `compare_experiments({...})` | Compare plusieurs runs sur un graphique |
| `save_images_for_fid(images, folder)` | Prépare les images pour calculer le FID |

## `visualize.py`
Fonctions pour visualiser les images générées par le modèle DDPM.

| Fonction | Ce qu'elle fait |
|---|---|
| `save_image_grid(images)` | Grille de toutes les images générées |
| `save_denoising_gif(frames)` | GIF bruit → image (démo finale) |
| `plot_denoising_steps(frames)` | Bande horizontale des étapes clés (parfait pour les slides) |
| `compare_real_vs_generated(real, gen)` | Deux lignes côte à côte : F1 réel vs généré |
```