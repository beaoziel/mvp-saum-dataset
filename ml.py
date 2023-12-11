# configuração para não exibir os warnings
import warnings

warnings.filterwarnings("ignore")

# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor 

# Informa o Path do arquivo
path = "ObesityDataSet.csv"

# Realiza a leitura
df = pd.read_csv(path, delimiter=",")

# Como o modelo de machine learning possui features que contêm valores não numéricos (como strings) e o modelo espera entradas numéricas.
# Com o label encoder, vamos realizar essa conversão.
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["family_history_with_overweight"] = label_encoder.fit_transform(df["family_history_with_overweight"])
df["FAVC"] = label_encoder.fit_transform(df["FAVC"])
df["CAEC"] = label_encoder.fit_transform(df["CAEC"])
df["SMOKE"] = label_encoder.fit_transform(df["SMOKE"])
df["SCC"] = label_encoder.fit_transform(df["SCC"])
df["FAF"] = label_encoder.fit_transform(df["FAF"])
df["CALC"] = label_encoder.fit_transform(df["CALC"])
df["MTRANS"] = label_encoder.fit_transform(df["MTRANS"])
df["NObeyesdad"] = label_encoder.fit_transform(df["NObeyesdad"])

#Vamos ver os dados do dataframe.
print(df.head())


test_size = 0.20  # tamanho do conjunto de teste, 20% do dataset
seed = 7  # semente aleatória

# Separação em conjuntos de treino e teste
array = df.values #definição do array como os valores do dataframe definidos anteriormente
X = array[:, 0:16]  # atributos
y = array[:, 16]  # classes do modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=True, random_state=seed, stratify=y
)  # holdout com estratificação

# Parâmetros e partições da validação cruzada
scoring = "accuracy"
num_particoes = 10
kfold = StratifiedKFold(
    n_splits=num_particoes, shuffle=True, random_state=seed)  # validação cruzada com estratificação

np.random.seed(7)  # definindo uma semente global

# Lista que armazenará os modelos
models = []

# Criando os modelos e adicionando-os na lista de modelos
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))

# Listas para armazenar os resultados
results = []
names = []

# Avaliação dos modelos
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot de comparação dos modelos
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Comparação dos Modelos")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Inicio de uso de pipelines
np.random.seed(7)  # definindo uma semente global para este bloco

# Listas para armazenar os armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
names = []


# Criando os elementos do pipeline

# Algoritmos que serão utilizados
knn = ("KNN", KNeighborsClassifier())
cart = ("CART", DecisionTreeClassifier())
naive_bayes = ("NB", GaussianNB())
svm = ("SVM", SVC())

# Transformações que serão utilizadas
standard_scaler = ("StandardScaler", StandardScaler())
min_max_scaler = ("MinMaxScaler", MinMaxScaler())


# Montando os pipelines

# Dataset original
pipelines.append(("KNN-orig", Pipeline([knn])))
pipelines.append(("CART-orig", Pipeline([cart])))
pipelines.append(("NB-orig", Pipeline([naive_bayes])))
pipelines.append(("SVM-orig", Pipeline([svm])))

# Dataset Padronizado
pipelines.append(("KNN-padr", Pipeline([standard_scaler, knn])))
pipelines.append(("CART-padr", Pipeline([standard_scaler, cart])))
pipelines.append(("NB-padr", Pipeline([standard_scaler, naive_bayes])))
pipelines.append(("SVM-padr", Pipeline([standard_scaler, svm])))

# Dataset Normalizado
pipelines.append(("KNN-norm", Pipeline([min_max_scaler, knn])))
pipelines.append(("CART-norm", Pipeline([min_max_scaler, cart])))
pipelines.append(("NB-norm", Pipeline([min_max_scaler, naive_bayes])))
pipelines.append(("SVM-norm", Pipeline([min_max_scaler, svm])))

# Executando os pipelines
for name, model in pipelines:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.3f (%.3f)" % (
        name,
        cv_results.mean(),
        cv_results.std(),
    )  # formatando para 3 casas decimais
    print(msg)

# Boxplot de comparação dos modelos
fig = plt.figure(figsize=(25, 6))
fig.suptitle("Comparação dos Modelos - Dataset orginal, padronizado e normalizado")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=90)
plt.show()

# Tuning do CART

np.random.seed(7)  # definindo uma semente global para este bloco

pipelines = []

# Definindo os componentes do pipeline
cart = ("CART", DecisionTreeClassifier())
standard_scaler = ("StandardScaler", StandardScaler())
min_max_scaler = ("MinMaxScaler", MinMaxScaler())

pipelines.append(("cart-orig", Pipeline(steps=[cart])))
pipelines.append(("cart-padr", Pipeline(steps=[standard_scaler, cart])))
pipelines.append(("cart-norm", Pipeline(steps=[min_max_scaler, cart])))

param_grid = {
    "max_features": ["auto", "sqrt", "log2"],
    "ccp_alpha": [
        0.1,
        0.01,
        0.001,
    ],  # represents the non-negative complexity parameter (alpha) in the pruning formula
    "max_depth": [5, 6, 7, 8, 9],  # The maximum depth of the tree.
    "criterion": ["gini", "entropy"],  # The function to measure the quality of a split.
    "min_samples_split": [2, 5, 10, 17],
    "min_samples_leaf": [1, 2, 4, 6],
}

# Prepara e executa o GridSearchCV
for name, model in pipelines:
    tree_clas = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
    grid.fit(X_train, y_train)
    # imprime a melhor configuração
    print(
        "Sem tratamento de missings: %s - Melhor: %f usando %s"
        % (name, grid.best_score_, grid.best_params_)
    )


# Avaliação do modelo com o conjunto de testes 

# Preparação do modelo
scaler = StandardScaler().fit(X_train)  # ajuste do scaler com o conjunto de treino
rescaledX = scaler.transform(X_train)  # aplicação da padronização no conjunto de treino
model = DecisionTreeClassifier( #aplicação dos parametros cart-norm
    ccp_alpha=0.001,
    criterion="entropy",
    max_depth=9,
    max_features="sqrt",
    min_samples_leaf=2,
    min_samples_split=5,
)
model.fit(rescaledX, y_train)

# Estimativa da acurácia no conjunto de teste
rescaledTestX = scaler.transform(
    X_test
)  # aplicação da padronização no conjunto de teste
predictions = model.predict(rescaledTestX)
print(accuracy_score(y_test, predictions))

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# simulando
data = {
    "Gender": ["Female"],
    "Age": [53],
    "Height": [1.53],
    "Weight": [80],
    "family_history_with_overweight": ["Yes"],
    "FAVC": ["no"],
    "FCVC": [1],
    "NCP": [4],
    "CAEC": ["Frequently"],
    "SMOKE": ["yes"],
    "CH2O": [1],
    "SCC": ["no"],
    "FAF": [2],
    "TUE": [2],
    "CALC": ["Frequently"],
    "MTRANS": ["Automobile"]
}

#Realizando a conversão dos dados com label encoder, ja definido anteriormente
data["Gender"] = label_encoder.fit_transform(data["Gender"])
data["family_history_with_overweight"] = label_encoder.fit_transform(
    data["family_history_with_overweight"])
data["FAVC"] = label_encoder.fit_transform(data["FAVC"])
data["CAEC"] = label_encoder.fit_transform(data["CAEC"])
data["SMOKE"] = label_encoder.fit_transform(data["SMOKE"])
data["SCC"] = label_encoder.fit_transform(data["SCC"])
data["FAF"] = label_encoder.fit_transform(data["FAF"])
data["CALC"] = label_encoder.fit_transform(data["CALC"])
data["MTRANS"] = label_encoder.fit_transform(data["MTRANS"])

atributos = [ "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC", #Frequent consumption of high caloric food | yes no
    "FCVC", #Frequency of consumption of vegetables | 1 3
    "NCP", #Number of main meals | 1 4
    "CAEC", #Consumption of food between meals | Sometimes Frequently
    "SMOKE", #Smoker or not
    "CH2O", #Consumption of water daily | 1 3
    "SCC", #Calories consumption monitoring | yes no
    "FAF", #Physical activity frequency | 0 3
    "TUE", #Time using technology devices | 0 2
    "CALC", #Consumption of alcohol | Sometimes no
    "MTRANS"] #Transportation used Public_T Automobile

input = pd.DataFrame(data, columns=atributos)
regressor.score(X_test, y_test)
predictions = regressor.predict(X_test)
array_entrada = input.values
X_entrada = array_entrada[:,0:16].astype(float)

# Padronização nos dados de entrada usando o scaler utilizado em X
rescaledEntradaX = scaler.transform(X_entrada)

print(rescaledEntradaX)

# Predição de classes dos dados de entrada
saidas = model.predict(rescaledEntradaX)
print(saidas)

#0 - insuficiencia
#1 - normal weight
#2 - obesidade tipo 1
#3 - Obesity_Type_II
#4 - Obesidade tupo 3
#5 - Overweight_Level_I
#6 - Overweight_Level_II