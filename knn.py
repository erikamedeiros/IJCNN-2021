import pandas as pd
arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

y = arq["Y"]
x = arq.drop('Y', axis=1)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
X_norm = normalizador.fit_transform(x)

seed=1
np.random.seed(seed)

modelo = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski', p=1)

#se quiser usar o StratifiedKFold, onde tiver a variavel kfold troca por skfold
#skfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=2)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
resultado = cross_val_score(modelo,X_norm,y, cv = kfold, n_jobs=-1)

print(resultado.mean())

#Definindo valores que serão testados no KNN:
valores_K = np.array([2, 3, 5, 7, 11])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1, 2, 3, 4])
valores_weights = np.array(['uniform', 'distance'])
valores_grid = {'n_neighbors': valores_K, 'metric': calculo_distancia, 'p': valores_p, 'weights': valores_weights}

#Criacao do modelo
modelo = KNeighborsClassifier()

#Criando os grids
gridKNN = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 4)
gridKNN.fit(X_norm, y)

#Imprimindo os melhores parâmetros:
print("Melhor acurácia: ", gridKNN.best_score_)
print("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
print("Melhor weights: ", gridKNN.best_estimator_.weights)
print("Melhor distância: ", gridKNN.best_estimator_.metric)
print("Melhor valor p: ", gridKNN.best_estimator_.p)
