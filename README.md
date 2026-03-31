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