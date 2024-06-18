# plots.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
def plot_sample_images(images, labels, label_names, num=10):
    fig, axes = plt.subplots(1, num, figsize=(15, 15))
    for i in range(num):
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(label_names[labels[i]])
        axes[i].axis('off')
    plt.savefig('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/raw/sample_images.png')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/raw/confusion_matrix.png')
    plt.show()

# Visualiser quelques images pour comprendre la structure des données
def show_images(images, labels, label_names, num_images=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(label_names[labels[i]])
        plt.axis('off')
    plt.show()

# Visualiser les performances du modèle
def plot_roc_curve(y_true, y_score, n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/raw/roc_curve.png')
    plt.show()

if __name__ == "__main__":
    from dataset import prepare_data
    from features import extract_features
    from modeling.train import train_classifier

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    X_train_flat = extract_features(X_train, method='flatten')
    X_val_flat = extract_features(X_val, method='flatten')

    model, y_pred = train_classifier(X_train_flat, y_train, X_val_flat, y_val, model_type='logistic')
    #y_pred = model.predict(X_val_flat)
    plot_confusion_matrix(y_val, y_pred, classes=[str(i) for i in range(10)])