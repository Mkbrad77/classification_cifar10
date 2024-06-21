# CIFAR Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Classification of CIFAR-10 images using various algorithms

## Project Organization

```
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cifar10_classification
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── __init__.py
│   ├── modeling
│   │   ├── grid_search.py
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── plots.py
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs
│   ├── docs
│   │   ├── experiments.md
│   │   ├── getting-started.md
│   │   └── index.md
│   ├── mkdocs.yml
│   ├── model_results_2best.xlsx
│   ├── model_results_best.xlsx
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│   ├── best_model_show.ipynb    <- Tutorial for importing the best-trained models
│   ├── best_model_show.ipynb <-- ici je présente les performances de mes meilleurs projets
│   ├── experimentation.ipynb
│   ├── initial_data_exploration.ipynb
│   ├── model_results2.xlsx
│   ├── model_results.xlsx
|   ├── test_best_model.ipynb <-- Ici possibilité de tester n'importe quelle image hors base de donnée CIFAR10 (ici comme exemple je prend une image de voiture téléchargé sur google)
│   └── Tuto_librairie.ipynb     <- Tutorial on using the library
├── pyproject.toml     <- Project configuration file with package metadata for ML_&_reconnaissance_de_forme
│                         and configuration for tools like black
├── README.md          <- The top-level README for developers using this project.
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── setup.cfg          <- Configuration file for flake8
└── setup.py           <- Makes the project pip installable (pip install -e .) so src can be imported

```
## Configuration initiale du projet

Après avoir cloné ce dépôt, suivez les étapes ci-dessous pour configurer Git LFS et télécharger les fichiers volumineux nécessaires.

### 1. Cloner le dépôt

1. **Cloner le dépôt** :
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/your-username/classification_cifar10.git
    cd classification_cifar10
    ```

2. **Initialiser Git LFS** :
    ```bash
    git lfs install
    ```

3. **Télécharger les fichiers volumineux** :
    ```bash
    git lfs pull
    ```
    Si ça marche pas Vous devrez placer manuellement les fichiers volumineux dans le répertoire approprié: dans le répertoire `models` pour les models '.plk'. 


### Utilisation du Makefile
Ce projet utilise un Makefile pour automatiser les tâches courantes. Voici comment utiliser les différentes commandes du Makefile :

1. **Configurer l'environnement** :
La création d'un environnement virtuel permet d'isoler les dépendances spécifiques à ce projet des autres projets et des paquets installés globalement sur votre système. Cela évite les conflits de versions et assure que votre projet utilise exactement les versions de paquets spécifiées dans `requirements.txt`.

```bash
make setup_env
```

2. **Installer les dépendances** :

```bash
make install_deps
```

3. **Préparer les données** :

```bash
make prepare_data
```

4. **Extraire les caractéristiques** :

```bash
make extract_features
```

5. **Entraîner le modèle** :

```bash
make train_model
```

6. **Évaluer le modèle** :
ICi pour le moment le modèle le plus performant, SVM avec des features extraites de HOG, est chargé et les résultats sont affichés.

```bash
make evaluate_model
```

7. **Effectuer une recherche en grille (grid search)**
Ici, certains modèles ont été mis en commentaire. Seule la régression logistique avec les features extraites de HOG (prend 0.89 min) est exécutée et son résultat est fourni.

```bash
make grid_search
```

8. **Nettoyer les fichiers générés** :

```bash
make clean


--------

