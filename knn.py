import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]

y = dataset["Y"]
x = dataset.drop('Y', axis=1)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
X_norm = normalizador.fit_transform(x)

seed=0
np.random.seed(seed)


#Definindo valores que serão testados no KNN:
valores_K = np.array([4, 5, 6])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1, 2])
valores_weights = np.array(['uniform', 'distance'])
valores_grid = {'n_neighbors': valores_K, 'metric': calculo_distancia, 'p': valores_p, 'weights': valores_weights}

#Criacao do modelo
modelo = KNeighborsClassifier()

#Criando os grids
gridKNN = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 4)
gridKNN.fit(X_norm, y)

#Imprimindo os melhores parâmetros:
print("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
print("Melhor weights: ", gridKNN.best_estimator_.weights)
print("Melhor distância: ", gridKNN.best_estimator_.metric)
print("Melhor valor p: ", gridKNN.best_estimator_.p)
print("Melhor acurácia: ", gridKNN.best_score_)

table = (pd.concat([pd.DataFrame(gridKNN.cv_results_["params"]),pd.DataFrame(gridKNN.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

