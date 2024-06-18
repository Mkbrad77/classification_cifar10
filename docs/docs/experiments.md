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

