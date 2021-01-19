import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
seed = 0

arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]
classe = arq["Y"]
previsores = arq.drop('Y', axis=1)
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.20, random_state=seed)

classificador = MLPClassifier(verbose = True, max_iter = 1500,
                              tol = 0.000010, solver='lbfgs',
                              hidden_layer_sizes=(3), activation = 'relu',
                              random_state = 1)
 
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
resultado = cross_val_score(classificador,previsores,classe, cv = kfold, n_jobs=-1)
print(resultado.mean())

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


