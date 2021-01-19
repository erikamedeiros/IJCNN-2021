# https://machinelearningmastery.com/clustering-algorithms-with-python/

import numpy as np #para manipular os vetores
import pandas as pd #para abrir arquivos
from sklearn.cluster import KMeans #para usar o KMeans
from sklearn.preprocessing import MinMaxScaler #para normalizar
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot

#loading file
dataset = pd.read_csv('dataset.csv', sep=';') 
del dataset["imgName"]
y = dataset["Y"]
del dataset["Y"]

#creating csv file array
dataset = np.array(dataset)


#creating a modelo
model = KMeans(n_clusters=4, random_state=1)
# fit the modelo
model.fit(dataset)

# assign a cluster to each example
yhat = model.predict(dataset)

# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(dataset[row_ix, 0], dataset[row_ix, 1])
# show the plot
pyplot.show()

clusters =  model.predict(dataset)

#printing the same result through the target variable
from sklearn.metrics import accuracy_score
print(accuracy_score(y, clusters))

