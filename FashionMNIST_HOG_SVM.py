import kagglehub
from pathlib import Path
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

import gzip
import shutil
import idx2numpy
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data_path = Path("/run/media/eduardo-velasco/Data/aDoctorado/Articulos/CITCA2026")

X_train = idx2numpy.convert_from_file(
   str(data_path / "train-images-idx3-ubyte")
)

y_train = idx2numpy.convert_from_file(
    str(data_path / "train-labels-idx1-ubyte")
)

X_test = idx2numpy.convert_from_file(
    str(data_path / "t10k-images-idx3-ubyte")
)

y_test = idx2numpy.convert_from_file(
    str(data_path / "t10k-labels-idx1-ubyte")
)

print("Este es el codigo FashionMNIST_HOG_SVM.py")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_train[i],
        cmap="gray"
    )

    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

def extract_hog_features(
    X,
    orientations=9,
    # pixels_per_cell=(4, 4), 
    pixels_per_cell=(7,7), 
    cells_per_block=(2, 2)
):
    features = []
    for image in X:
        hog_features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block
        )
        features.append(hog_features)
    return np.array(features)

print("Cargando características HOG...")
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("X_train:", X_train.shape)
print("X_train_hog:", X_train_hog.shape)

print("X_test:", X_test.shape)
print("X_test_hog:", X_test_hog.shape)

# Valores HOG dentro de un rango razonable
print("Mínimo:", X_train_hog.min())
print("Máximo:", X_train_hog.max())
print("Media:", X_train_hog.mean())
print("Desviación:", X_train_hog.std())

# print(X_train_hog[0]) # Matriz de características HOG de la primera imagen
print("Cargando accuracy")

# Sección de SVM (Support Vector Machine) para clasificación de dígitos
svm = SVC(
    kernel="rbf",
    C=10
)

# entrenar
# svm.fit(X_train_hog, y_train)

# y_pred = svm.predict(X_test_hog)

# accuracy = accuracy_score(y_test, y_pred)

# # Accuracy
# print("Accuracy:", accuracy)

# print(
#     classification_report(y_test,y_pred)
# )

# 5,000 imágenes
N = 5000
X_train_small = X_train_hog[:N]
y_train_small = y_train[:N]

# SVM con conjunto reducido
svm.fit(X_train_small, y_train_small)

y_pred = svm.predict(X_test_hog)

print(
    accuracy_score(y_test, y_pred)
)

# Prueba con diferentes tamaños
training_sizes = [
    1000,
    5000,
    10000,
    30000,
    60000
]

for N in training_sizes:

    print(f"\nEntrenando con {N} imágenes...")

    svm = SVC(
        kernel="rbf",
        C=10
    )

    svm.fit(
        X_train_hog[:N],
        y_train[:N]
    )

    y_pred = svm.predict(X_test_hog)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("Accuracy:", accuracy)