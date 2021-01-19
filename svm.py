import pandas as pd

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

seed=1

#criacao do modelo
modelo = SVC(C=0.9, kernel='poly', degree=2, gamma='scale', random_state=seed) #Olhei o tunning pra ter esses paramentros

#se quiser usar o StratifiedKFold, onde tiver a variavel kfold troca por skfold
#skfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=2)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
resultado = cross_val_score(modelo,X_norm,y, cv = kfold, n_jobs=-1)

print(resultado.mean())


#Fazendo um refinamento dos parametros
import numpy as np
from sklearn.model_selection import GridSearchCV

#Definindo valores que serão testados em SVM:
c = np.array([1.0, 0.99, 0.96, 1.1, 0.95, 1.05, 1.1, 1.2, 2.0, 0.9, 0.8, 0.7])
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([1, 2, 3, 4, 5])
gamma = ['auto', 'scale']
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'gamma': gamma}

#Criacao do Modelo
modelo = SVC(random_state=seed)

#Criando os grids
gridSVM = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=3, n_jobs=-1)
gridSVM.fit(X_norm,y)

#imprimindo os melhores parâmetros
print ("Melhor valor constante: ", gridSVM.best_estimator_.C)
print("Melhor kernel: ", gridSVM.best_estimator_.kernel)
print("Melhor grau polinômio: ", gridSVM.best_estimator_.degree)
print("Melhor gama: ", gridSVM.best_estimator_.gamma)
print("Acurácia: ", gridSVM.best_score_)


