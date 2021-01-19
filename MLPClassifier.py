import pandas as pd


arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

classe = arq["Y"]
previsores = arq.drop('Y', axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.20, random_state=0)

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True, max_iter = 1500,
                              tol = 0.000010, solver='lbfgs',
                              hidden_layer_sizes=(100), activation = 'relu',
                              random_state = 1)
 

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
print('Precisao:', precisao)


