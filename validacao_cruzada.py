#Referencia
# https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/#:~:text=Repeated%20k%2Dfold%20cross%2Dvalidation%20provides%20a%20way%20to%20improve,all%20folds%20from%20all%20runs

# compare the number of repeats for repeated k-fold cross-validation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import sem
from numpy import mean
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot

seed=0

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]
X = dataset.drop('Y', axis=1)
y = dataset["Y"]

normalizador = MinMaxScaler(feature_range = (0, 1))
X = normalizador.fit_transform(X)

# create model
KNN_model = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=1)
SVC_model = SVC(C=0.8, kernel='poly', degree=2, gamma='scale', random_state=seed)
MLPClassifier_Model = MLPClassifier(verbose = True, max_iter = 4000,
                              tol= 0.000010, solver='lbfgs',
                              hidden_layer_sizes=(4), activation = 'relu', random_state=seed)

# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats, model):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# configurations to test do KNN
def test(model):
    results = list()
    scores = evaluate_model(X, y, 30, model)
    # summarize
    print('>%d mean=%.4f se=%.3f' % (30, mean(scores), sem(scores)))
    # store
    results.append(scores)

print('KNN')
test(KNN_model)

print('SVC')
test(SVC_model)

print('MLPClassifier')
test(MLPClassifier_Model)



