import kagglehub
from pathlib import Path
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", path)

path = Path(path)

#for file in path.iterdir():
#    print(file.name)


# Datos de entrenamiento
X_train = idx2numpy.convert_from_file(
    str(path / "train-images-idx3-ubyte" / "train-images-idx3-ubyte")
)

y_train = idx2numpy.convert_from_file(
    str(path / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte")
)

X_test = idx2numpy.convert_from_file(
    str(path / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte")
)

y_test = idx2numpy.convert_from_file(
    str(path / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte")
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
# plt.show()

# Visualización de una sola imagen y su etiqueta correspondiente
# plt.figure(figsize=(5, 5))
# plt.imshow(X_train[0], cmap="gray")
# plt.title(f"Dígito: {y_train[0]}")
# plt.colorbar()
# plt.show()

# print(X_train[0])

classes, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(classes, counts)
plt.xlabel("Clase")
plt.ylabel("Número de imágenes")
plt.title("Distribución de clases - MNIST")
plt.xticks(classes)

# plt.show()

# Fashion MNIST dataset (disponible en github: https://github.com/zalandoresearch/fashion-mnist)
# Es necesrio descomprimir los archivos .gz para poder leerlos con idx2numpy
data_path = Path("/run/media/eduardo-velasco/Data/aDoctorado/Articulos/CITCA2026")

# X_train_fashion = idx2numpy.convert_from_file(
#     str(data_path / "train-images-idx3-ubyte.gz")
# )

X_train_fashion = idx2numpy.convert_from_file(
   str(data_path / "train-images-idx3-ubyte")
)

y_train_fashion = idx2numpy.convert_from_file(
    str(data_path / "train-labels-idx1-ubyte")
)

X_test_fashion = idx2numpy.convert_from_file(
    str(data_path / "t10k-images-idx3-ubyte")
)

y_test_fashion = idx2numpy.convert_from_file(
    str(data_path / "t10k-labels-idx1-ubyte")
)

print(X_train_fashion.shape)
print(y_train_fashion.shape)

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
        X_train_fashion[i],
        cmap="gray"
    )

    plt.title(class_names[y_train_fashion[i]])
    plt.axis("off")

plt.tight_layout()
# plt.show()

# HOG (Histogram of Oriented Gradients) feature extraction
imagen = X_train[0]

features, hog_image = hog(
    imagen,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    visualize=True
)
print("HOG features shape")
print(features.shape)
print(hog_image.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(imagen, cmap="gray")
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(hog_image, cmap="gray")
axes[1].set_title("Representación HOG")
axes[1].axis("off")

plt.tight_layout()

#plt.show()

# Hiperparámetros de HOG
hog(
    imagen,
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    visualize=True
)

X_hog = []

print("Cargando")
for imagen in X_train:
    features = hog(
        imagen,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2)
    )
    X_hog.append(features)
X_hog = np.array(X_hog)
print(X_hog.shape)

N = 1000
X_sample = X_train[:N]
y_sample = y_train[:N]
X_hog = []
for imagen in X_sample:
    features = hog(
        imagen,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2)
    )
    X_hog.append(features)
X_hog = np.array(X_hog)
print(X_hog.shape)

