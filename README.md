# CIFAR Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Classification of CIFAR-10 images using various algorithms

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for ML_&_reconnaissance_de_forme
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ML_&_reconnaissance_de_forme                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes ML_&_reconnaissance_de_forme a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
## Configuration initiale du projet

Après avoir cloné ce dépôt, suivez les étapes ci-dessous pour configurer Git LFS et télécharger les fichiers volumineux nécessaires.

### 1. Cloner le dépôt

1. **Cloner le dépôt** :
    ```bash
    git clone https://github.com/votre-nom-utilisateur/classification_cifar10.git
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

4. **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
--------

