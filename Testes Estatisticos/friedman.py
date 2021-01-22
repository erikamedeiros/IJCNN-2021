# Friedman test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare

import matplotlib.pyplot as plt
import pandas as pd


# seed the random number generator
seed(0)
# generate three independent samples
dataset = pd.read_csv('dados_testes.csv')  
data1 = dataset["KNN"]
data2 = dataset["SVM"]
data3 = dataset["MLPClassifier"]

# compare samples
stat, p = friedmanchisquare(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

from autorank import autorank, create_report, plot_stats
results = autorank(dataset)
create_report(results)
plot_stats(results)

