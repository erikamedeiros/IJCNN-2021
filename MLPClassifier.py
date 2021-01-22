import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

seed = 0

arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]
classe = arq["Y"]
previsores = arq.drop('Y', axis=1)
normalizador = MinMaxScaler(feature_range = (0, 1))
previsores = normalizador.fit_transform(previsores)

#Definindo valores que serão testados no KNN:
valores_hidden_layer_sizes = np.array([3, 4, 5])
valores_activation = np.array(['identity', 'logistic', 'tanh', 'relu'])
valores_solver = np.array(['lbfgs', 'sgd', 'adam'])
valores_batch_size = np.array([40, 50])

valores_grid = {'hidden_layer_sizes': valores_hidden_layer_sizes, 
                'activation': valores_activation, 
                'solver': valores_solver,
                'batch_size': valores_batch_size}

#criação do modelo
modelo = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state = seed)
 
gridMLP = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 4)
gridMLP.fit(previsores, classe)

print('Melhor hiden_layer_size: ', gridMLP.best_estimator_.hidden_layer_sizes)
print('Melhor activation: ', gridMLP.best_estimator_.activation)
print('Melhor solver: ', gridMLP.best_estimator_.solver)
print('Melhor batch_size: ', gridMLP.best_estimator_.batch_size)
print('Melhor acuracia: ', gridMLP.best_score_)

table = (pd.concat([pd.DataFrame(gridMLP.cv_results_["params"]),pd.DataFrame(gridMLP.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

