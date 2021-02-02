#imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]
X = dataset.drop('Y', axis=1)
y = dataset["Y"]


#tunign KNN
def tuning_KNN(x,y):
    #Criacao do grid para o KNN
    valores_n_neighbors = np.array([4, 5, 6])
    valores_weights = np.array(['uniform', 'distance'])
    valores_p = np.array([1, 2])
    valores_metric = ['minkowski','chebyshev']
    valores_grid = {'n_neighbors': valores_n_neighbors, 'weights': valores_weights, 'p': valores_p, 'metric': valores_metric}
    
    #chamdno o modelo
    modelo = KNeighborsClassifier()
    #Criando os grids
    gridKNN = GridSearchCV(estimator= modelo, param_grid = valores_grid, cv = 2, n_jobs=-1)
    gridKNN.fit(x, y)
    return gridKNN

#tunign SVM
def tuning_SVM(x,y):
    #Criacao do grid para o SVM
    valores_c = np.array([0.4, 0.5])
    valores_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    valores_degree = np.array([1, 2, 3])
    valores_gamma = ['auto', 'scale']
    valores_grid = {'C': valores_c, 'kernel': valores_kernel, 'degree': valores_degree, 'gamma': valores_gamma}
    
    #chamando o modelo
    modelo = SVC(random_state=0)
    #Criando os grids
    gridSVC = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=2, n_jobs=-1)
    gridSVC.fit(x,y)
    return gridSVC

#tunign SVM
def tuning_MLP(x,y):
    #Criacao do grid para o MLP
    valores_hidden_layer_sizes = np.array([3, 4, 5])
    valores_activation = np.array(['identity', 'logistic', 'tanh', 'relu'])
    valores_solver = np.array(['lbfgs', 'sgd', 'adam'])
    valores_batch_size = np.array([40, 50])
    
    valores_grid = {'hidden_layer_sizes': valores_hidden_layer_sizes, 
                    'activation': valores_activation, 
                    'solver': valores_solver,
                    'batch_size': valores_batch_size}
    
    #chamando o modelo
    modelo = MLPClassifier(max_iter = 4000, tol= 0.000010, random_state=0)
    #Criando os grids
    gridMLP = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=2, n_jobs=-1)
    gridMLP.fit(x,y)
    return gridMLP

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = KFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

#Listas para armazenamento
#Dados de treino
results_knn=list()
results_svc=list()
results_mlp=list()

#Dados de validação
results_val_KNN=list()
results_val_SVC=list()
results_val_MLP=list()

#Dados de teste
accuracy_predict_KNN=list()
accuracy_predict_SVC=list()
accuracy_predict_MLP=list()

for seed in range(1, 31):
    #Criando uma base de validacao   
    #Divisão da base em treino (80%) e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=True, stratify = y, random_state=seed)
    
    #normalização do x_treino e x_teste
    normalizador = MinMaxScaler(feature_range = (0, 1))
    x_treino = normalizador.fit_transform(x_treino)
    x_teste = normalizador.fit_transform(x_teste)
   
    #chamando o tuning dos classificadores
    KNN_model = tuning_KNN(x_treino, y_treino)
    SVC_model = tuning_SVM(x_treino, y_treino)
    MLP_Model = tuning_MLP(x_treino, y_treino)

    
    # preparar o modelo para ser validado no kfold
    scores_knn = evaluate_model(x_treino, y_treino, KNN_model)
    scores_svc = evaluate_model(x_treino, y_treino, SVC_model)
    scores_mlp = evaluate_model(x_treino, y_treino, MLP_Model)
    results_knn.append(mean(scores_knn))
    results_svc.append(mean(scores_svc))
    results_mlp.append(mean(scores_mlp))
    ############################################################
    
    #Acurácia nos dados de validação
    results_val_KNN.append(KNN_model.best_score_)
    results_val_SVC.append(SVC_model.best_score_)
    results_val_MLP.append(MLP_Model.best_score_)
    #################################################################
    
    #Make predictions
    KNN_model.fit(x_treino, y_treino)
    predict_KNN = KNN_model.predict(x_teste)
    accuracy_predict_KNN.append(accuracy_score(y_teste, predict_KNN))
    
    SVC_model.fit(x_treino, y_treino)
    predict_SVC = SVC_model.predict(x_teste)
    accuracy_predict_SVC.append(accuracy_score(y_teste, predict_SVC))
    
    MLP_Model.fit(x_treino, y_treino)
    predict_MLP = MLP_Model.predict(x_teste)
    accuracy_predict_MLP.append(accuracy_score(y_teste, predict_MLP))
    ####################################################################
    
#Dados de treino
print ("Média da acurácia do KNN nos dadaos de treino {:.4f}%.".format(mean(results_knn)*100))
print("Desvio padrao do KNN no dados de treino: ", np.std(results_knn))
print ("Média da acurácia do SVM nos dadaos de treino {:.4f}%.".format(mean(results_svc)*100))
print("Desvio padrao do SVM no dados de treino: ", np.std(results_svc))
print ("Média da acurácia do MLP nos dadaos de treino {:.4f}%.".format(mean(results_mlp)*100))
print("Desvio padrao do MLP no dados de treino: ", np.std(results_mlp))

#Dados de validacao 
print("--------------------------------------------------------------")
print ("A acurácia da validacao do KNN foi de {:.4f}%.".format(mean(results_val_KNN)*100))
print("Desvio padrao nos dados de validação do KNN: ", np.std(results_val_KNN))
print ("A acurácia da validacao do KNN foi de {:.4f}%.".format(mean(results_val_SVC)*100))
print("Desvio padrao nos dados de validação do SVM: ", np.std(results_val_SVC))
print ("A acurácia da validacao do KNN foi de {:.4f}%.".format(mean(results_val_MLP)*100))
print("Desvio padrao nos dados de validação do MLP: ", np.std(results_val_MLP))

#Dados de teste
print("--------------------------------------------------------------")
print ("A acurácia da predição do KNN foi de {:.4f}%.".format(mean(accuracy_predict_KNN)*100))
print("Desvio padrao na predição do KNN: ", np.std(accuracy_predict_KNN))
print ("A acurácia da predição do SVM foi de {:.4f}%.".format(mean(accuracy_predict_SVC)*100))
print("Desvio padrao na predição do SVM: ", np.std(accuracy_predict_SVC))
print ("A acurácia da predição do MLP foi de {:.4f}%.".format(mean(accuracy_predict_MLP)*100))
print("Desvio padrao na predição do SVM: ", np.std(accuracy_predict_MLP))

