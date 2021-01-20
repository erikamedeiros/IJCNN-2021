import pandas as pd
import numpy as np

arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

y = arq["Y"]
x = arq.drop('Y', axis=1)

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
X_norm = normalizador.fit_transform(x)

seed=0
np.random.seed(seed)

#Fazendo um refinamento dos parametros
import numpy as np
from sklearn.model_selection import GridSearchCV

#Definindo valores que serão testados em SVM:
c = np.array([0.4, 0.5, 0.6])
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([1, 2, 3])
gamma = ['auto', 'scale']

valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'gamma': gamma}

#Criacao do Modelo
modelo = SVC(random_state=seed)

#Criando os grids
gridSVM = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=4, n_jobs=-1)
gridSVM.fit(X_norm,y)

#imprimindo os melhores parâmetros
print ("Melhor valor constante: ", gridSVM.best_estimator_.C)
print("Melhor kernel: ", gridSVM.best_estimator_.kernel)
print("Melhor degree: ", gridSVM.best_estimator_.degree)
print("Melhor gama: ", gridSVM.best_estimator_.gamma)
print("Melhor Acurácia: ", gridSVM.best_score_)

table = (pd.concat([pd.DataFrame(gridSVM.cv_results_["params"]),pd.DataFrame(gridSVM.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

#Fazendo uma testagem com os hiperparametros encontrados
modelo = SVC(C=0.5, kernel='poly', degree=2, gamma='scale', random_state=seed) #Olhei o tunning pra ter esses paramentros

kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
resultado = cross_val_score(modelo,X_norm,y, cv = kfold, n_jobs=-1)

print(resultado.mean())


