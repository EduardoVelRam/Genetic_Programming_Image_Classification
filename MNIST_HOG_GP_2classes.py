from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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

print("Este es el código MNIST_HOG_GP_2classes.py")
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

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

print("X_train:", X_train.shape)
print("X_train_hog:", X_train_hog.shape)

# Sección de SVM (Support Vector Machine) para clasificación de dígitos
svm = SVC(
    kernel="rbf",
    C=10
)

# 5,000 imágenes
#N = 5000
#X_train_small = X_train_hog[:N]
#y_train_small = y_train[:N]

# Seleccionar únicamente las clases 0 y 1
mask = (y_train == 0) | (y_train == 1)

X_binary = X_train_hog[mask]
y_binary = y_train[mask]

# Seleccionar 500 muestras de cada clase
rng = np.random.default_rng(42)

idx_0 = np.where(y_binary == 0)[0]
idx_1 = np.where(y_binary == 1)[0]

idx_0 = rng.choice(idx_0, size=500, replace=False)
idx_1 = rng.choice(idx_1, size=500, replace=False)

indices = np.concatenate([idx_0, idx_1])
rng.shuffle(indices)

X_gp = X_binary[indices]
y_gp = y_binary[indices]

print("X_gp:", X_gp.shape)
print("y_gp:", y_gp.shape)

print("Clase 0:", np.sum(y_gp == 0))
print("Clase 1:", np.sum(y_gp == 1))

# Representación que utilizará el arbol: 0: -1; 1: +1
y_gp_binary = np.where(y_gp == 1, 1, -1)


######################################### Creación de GP en DEAP
# Funcion de fitness y el individuo
creator.create(
    "FitnessMin",
    base.Fitness,
    weights=(-1.0,)
)

creator.create(
    "Individual",
    gp.PrimitiveTree,
    fitness=creator.FitnessMin
)

# Como hay 324 características HOG, el arbol tendrá 324 entradas.
pset = gp.PrimitiveSet("MAIN", 324)

for i in range(324):
    pset.renameArguments(**{f"ARG{i}": f"x{i}"})

# De momento, se utilizarán las sigueintes operaciones: suma, resta, multiplicación y división protegida
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
def protected_div(left, right):
    if abs(right) < 1e-6:
        return 1.0
    return left / right
pset.addPrimitive(protected_div, 2)

# Se agregan constantes aleatorias entre -1 y 1
def random_constant():
    return random.uniform(-1, 1)
pset.addEphemeralConstant("rand", random_constant)

# Se crea la población
toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# Se define la función de evaluación
def evaluate(individual, X, y):
    func = toolbox.compile(expr=individual)
    predictions = []
    for sample in X:
        score = func(*sample)
        if score > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    predictions = np.array(predictions)
    error = np.mean(predictions != y)

    return (error,)

# Se registra la función 
toolbox.register(
    "evaluate",
    evaluate,
    X=X_gp,
    y=y_gp_binary
)

# Se registran los operadores evolutivos
# Se usarán selección por torneo, crossover y mutación 
toolbox.register("select", tools.selTournament,tournsize=3)

toolbox.register("mate", gp.cxOnePoint)

toolbox.register("expr_mut",gp.genFull, min_=0, max_=2)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Se limita la altura del árbol a 7 para evitar el sobreajuste y mantener la interpretabilidad
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

toolbox.decorate("mutate",gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

# Ejecución de la evolución genética
random.seed(42)
np.random.seed(42)

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
    verbose=True
)

best_individual = hof[0]

print("Mejor árbol:")
print(best_individual)

print("\nFitness:")
print(best_individual.fitness.values[0])


# Evaluación sobre datos no vistos (conjunto de prueba)
# Seleccionar clases 0 y 1 del conjunto de prueba
mask_test = (y_test == 0) | (y_test == 1)

X_test_binary = X_test_hog[mask_test]
y_test_binary = y_test[mask_test]

# Convertir etiquetas a -1 y +1
y_test_binary = np.where(y_test_binary == 1, 1, -1)

print("X_test_binary:", X_test_binary.shape)
print("Clase 0:", np.sum(y_test_binary == -1))
print("Clase 1:", np.sum(y_test_binary == 1))

# Se evalúa el arbol evoluvcionado
best_func = toolbox.compile(expr=best_individual)

predictions_test = []

for sample in X_test_binary:
    score = best_func(*sample)

    if score > 0:
        predictions_test.append(1)
    else:
        predictions_test.append(-1)

predictions_test = np.array(predictions_test)

accuracy_test = np.mean(predictions_test == y_test_binary)

print("Accuracy en entrenamiento:", 1 - best_individual.fitness.values[0])
print("Accuracy en prueba:", accuracy_test)