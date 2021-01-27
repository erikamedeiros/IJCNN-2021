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
from sklearn.model_selection import cross_val_predict


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
    
    #Imprimindo os melhores parâmetros:
    #print("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
    #print("Melhor weights: ", gridKNN.best_estimator_.weights)
    #print("Melhor distância: ", gridKNN.best_estimator_.metric)
    #print("Melhor valor p: ", gridKNN.best_estimator_.p)
    #print("Melhor acurácia: ", gridKNN.best_score_)
    
    n = gridKNN.best_estimator_.n_neighbors
    w = gridKNN.best_estimator_.weights
    p = gridKNN.best_estimator_.p
    m = gridKNN.best_estimator_.metric

    #table = (pd.concat([pd.DataFrame(gridKNN.cv_results_["params"]),pd.DataFrame(gridKNN.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    return n, w, p, m

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
    gridSVM = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=2, n_jobs=-1)
    gridSVM.fit(x,y)
    
    #Imprimindo os melhores parâmetros:
    #print("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
    #print("Melhor weights: ", gridKNN.best_estimator_.weights)
    #print("Melhor distância: ", gridKNN.best_estimator_.metric)
    #print("Melhor valor p: ", gridKNN.best_estimator_.p)
    #print("Melhor acurácia: ", gridKNN.best_score_)
    
    c = gridSVM.best_estimator_.C
    k = gridSVM.best_estimator_.kernel
    d = gridSVM.best_estimator_.degree
    g = gridSVM.best_estimator_.gamma
    
    #table = (pd.concat([pd.DataFrame(gridKNN.cv_results_["params"]),pd.DataFrame(gridKNN.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    return c, k, d, g

#tunign SVM
def tuning_MLP(x,y):
    #Criacao do grid para o SVM
    valores_hidden_layer_sizes = np.array([3, 4, 5])
    valores_activation = np.array(['identity', 'logistic', 'tanh', 'relu'])
    valores_solver = np.array(['lbfgs', 'sgd', 'adam'])
    valores_batch_size = np.array([40, 50])
    
    valores_grid = {'hidden_layer_sizes': valores_hidden_layer_sizes, 
                    'activation': valores_activation, 
                    'solver': valores_solver,
                    'batch_size': valores_batch_size}
    
    #chamando o modelo
    modelo = MLPClassifier(max_iter = 1, tol= 0.000010, random_state = 0)
    #Criando os grids
    gridSVM = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=2, n_jobs=-1)
    gridSVM.fit(x,y)
    
    h = gridSVM.best_estimator_.hidden_layer_sizes
    a = gridSVM.best_estimator_.activation
    s = gridSVM.best_estimator_.solver
    b = gridSVM.best_estimator_.batch_size
    
    #table = (pd.concat([pd.DataFrame(gridKNN.cv_results_["params"]),pd.DataFrame(gridKNN.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
    return h, a, s, b

#Avaliacao do modelo em kfold de 10
def evaluate_model(x, y, model):
	# prepare the cross-validation procedure
	cv = KFold(10, shuffle=True, random_state=0)
	# evaluate model
	scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

results_knn=list()
results_svm=list()
results_mlp=list()
accuracy_predict_KNN=list()

for seed in range(1, 31):
    
    #Divisão da base em treino e teste (20%)
    x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    normalizador = MinMaxScaler(feature_range = (0, 1))
    x_treino_norm = normalizador.fit_transform(x_treino)

    #chamando o tuning do KNN
    n, w, p, m = tuning_KNN(x_treino_norm, y_treino)
    c, k, d, g = tuning_SVM(x_treino_norm, y_treino)
#    h, a, s, b = tuning_MLP(x_treino_norm, y_treino)

    # create model com os parametros tunados
    KNN_model = KNeighborsClassifier(n_neighbors=n, weights=w, p=p, metric=m)
    SVC_model = SVC(C=c, kernel=k, degree=d, gamma=g, random_state=seed)
    '''
    MLP_Model = MLPClassifier(verbose = True, max_iter = 4000,
                                        tol= 0.000010, 
                                        hidden_layer_sizes=h,
                                        solver=s,
                                        activation = a, 
                                        batch_size = b,
                                        random_state=0)
    '''
    
    # preparar o modelo para ser validado no kfold
    scores_knn = evaluate_model(x_treino_norm, y_treino, KNN_model)
    scores_svm = evaluate_model(x_treino_norm, y_treino, SVC_model)
#    scores_mlp = evaluate_model(x_treino_norm, y_treino, MLP_Model)
    #print('>%d mean=%.4f se=%.3f' % (seed, mean(scores_knn), sem(scores_knn)))
    results_knn.append(mean(scores_knn))
    results_svm.append(mean(scores_svm))
#    results_mlp.append(mean(scores_mlp))
    
    #Make predictions
    KNN_model.fit(x_treino_norm, y_treino)
    predict = KNN_model.predict(x_teste)
    accuracy_predict_KNN.append(accuracy_score(y_teste, predict) * 100)
    
  
    
#print("Media da acurácia do KNN: {:.2f}%."..format(mean(results_knn)))
print ("Média da acurácia do KNN nos dadaos de treino {:.2f}%.".format(mean(results_knn)*100))
print("Media do desvio padrao do KNN: ", np.std(results_knn))
print ("Média da acurácia do SVM nos dadaos de treino {:.2f}%.".format(mean(results_svm)*100))
print("Media do desvio padrao do SVM: ", np.std(results_svm))
#print ("Média da acurácia do MLP nos dadaos de treino {:.2f}%.".format(mean(results_mlp)*100))
#print("Media do desvio padrao do MLP: ", np.std(results_mlp))

print ("A acurácia da predição do KNN foi de {:.2f}%.".format(mean(accuracy_predict_KNN)))

