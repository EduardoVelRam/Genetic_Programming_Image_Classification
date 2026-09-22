from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import kagglehub
from deap import base, creator, gp, tools, algorithms
import operator
import random
import numpy as np

mnist_path = Path(
    kagglehub.dataset_download("hojjatk/mnist-dataset")
)

# print(mnist_path)

def load_mnist(path):

    X_train = idx2numpy.convert_from_file(str(path / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"))

    y_train = idx2numpy.convert_from_file(str(path / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"))

    X_test = idx2numpy.convert_from_file(str(path / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte"))

    y_test = idx2numpy.convert_from_file(str(path / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte"))

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist(mnist_path)

print("Este es el código MNIST_HOG_GP_10classes.py")

def extract_hog_features(
    X,
    orientations=9,
    #pixels_per_cell=(4, 4), 
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




# Se seleccionan 100 imágenes de cada clase 

rng = np.random.default_rng(42)

indices_train = []

for clase in range(10):
    indices_clase = np.where(y_train == clase)[0]
    seleccion = rng.choice(indices_clase, size=100, replace=False)
    indices_train.extend(seleccion)

indices_train = np.array(indices_train)
rng.shuffle(indices_train)

X_gp = X_train_hog[indices_train]
y_gp = y_train[indices_train]

print("X_gp:", X_gp.shape)
print("y_gp:", y_gp.shape)

print(np.bincount(y_gp))

# ============================================================
# Configuración de Programación Genética
# ============================================================

# Fitness: minimizar el error
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Individuo GP
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Número de características HOG
n_features = X_gp.shape[1]

# Conjunto de primitivas
pset = gp.PrimitiveSet("MAIN", n_features)

# Operaciones aritméticas protegidas
def protected_div(left, right):
    return left / right if abs(right) > 1e-6 else 1.0

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)

# Crear toolbox
toolbox = base.Toolbox()

toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=1,
    max_=3
)

toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.expr
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register(
    "compile",
    gp.compile,
    pset=pset
)

# Operadores evolutivos
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

# Un árbol por clase, para un total de 10 árboles. Cada árbol se entrenará para clasificar una clase frente a todas las demás (One-vs-Rest).
def evaluate_ovr(individual, X, y, target_class):
    
    func = toolbox.compile(expr=individual)
    predictions = []

    for sample in X:
        score = func(*sample)
        if score > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    predictions = np.array(predictions)
    y_binary = np.where(y == target_class, 1, -1)
    balanced_acc = balanced_accuracy_score(y_binary, predictions)
    error = 1 - balanced_acc

    return (error,)

# Función para entrenar un GP
def train_gp_ovr(target_class, X, y):
    
    toolbox.register("evaluate", evaluate_ovr, X=X, y=y, target_class=target_class)

    population = toolbox.population(n=100)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(
        lambda ind: ind.fitness.values[0]
    )

    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=30,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    return hof[0], logbook

# Se entrenan los 10 modelos GP.
models = {}
logs = {}

for clase in range(10):

    print(f"Entrenando GP para clase {clase}...")

    best_tree, logbook = train_gp_ovr(
        clase,
        X_gp,
        y_gp
    )

    models[clase] = best_tree
    logs[clase] = logbook

    print(
        f"Fitness clase {clase}: "
        f"{best_tree.fitness.values[0]:.4f}"
    )

# comentar desde qui
# Función para predecir usando los 10 modelos GP entrenados
# def predict_ovr(models, X):
#     predictions = []

#     for sample in X:
#         scores = []
#         for clase in range(10):
#             func = toolbox.compile(
#                 expr=models[clase]
#             )

#             score = func(*sample)
#             scores.append(score)

#         predicted_class = np.argmax(scores)
#         predictions.append(predicted_class)

#     return np.array(predictions)

# # Evaluación sobre datos
# y_pred = predict_ovr(
#     models,
#     X_test_hog
# )

# # Calcular la precisión y el informe de clasificación
# accuracy = accuracy_score(
#     y_test,
#     y_pred
# )

# print("Accuracy:", accuracy)

# print(
#     classification_report(
#         y_test,
#         y_pred
#     )
# )
# descomentar hasta aqui

#
compiled_models = {
    clase: toolbox.compile(expr=models[clase])
    for clase in range(10)
}

# 
def predict_ovr(compiled_models, X):

    predictions = []

    for sample in X:
        scores = []
        for clase in range(10):
            score = compiled_models[clase](*sample)
            scores.append(score)
        predictions.append(
            np.argmax(scores)
        )

    return np.array(predictions)

y_pred = predict_ovr(
    compiled_models,
    X_test_hog
)

# Calcular la precisión y el informe de clasificación
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracyf:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)