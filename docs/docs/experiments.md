# Expérimentations

## Analyse de la base de données
La base de données CIFAR-10 contient 60 000 images de 32x32 pixels réparties en 10 classes. Les classes sont: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

## Pré-processing
Les images sont normalisées pour avoir des valeurs de pixels entre 0 et 1. Les données sont divisées en ensembles d'entraînement, de validation et de test avec un ratio de 80/20 pour l'entraînement et la validation.

## Choix des hyper-paramètres

### Split train/valid/test
- Train: 40 000 images (80% du total d'entraînement)
- Validation: 10 000 images (20% du total d'entraînement)
- Test: 10 000 images

### Taille des mini-batchs
La taille des mini-batchs est fixée à 64 images.

### Learning rate
Le learning rate utilisé pour la descente de gradient stochastique est de 0.01.

### Kernels utilisés
- Pour SVM: Kernel linéaire
- Pour HOG: Cellules de 8x8 pixels

### Époques
Le nombre d'époques pour l'entraînement des modèles est de 50.

