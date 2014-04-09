from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from numpy.linalg import lstsq
import numpy as np
from pylab import plot, title, show , legend
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model

datafile = open('nomissingbycol.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
for row in datacsv:
    data.append([float(row[i]) for i in range(len(row)-1)])
    val.append(float(row[len(row)-1]))
clf = linear_model.LinearRegression()
clf.fit(data, list(val))
print clf.score(data, list(val))
#matplotlib ploting
title('Residuals')
plot(val,[clf.predict(data[i]) - val[i] for i in range(len(data))], 'o')

show()

val, _ = stats.boxcox(np.array([x+1 for x in val]))
clf = linear_model.LinearRegression()
clf.fit(data, list(val))
print clf.score(data, list(val))
#matplotlib ploting
title('BoxCox Residuals')
plot(val,[clf.predict(data[i]) - val[i] for i in range(len(data))], 'o')

show()



datafile = open('imputed.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
for row in datacsv:
    data.append([float(row[i]) for i in range(len(row)-1)])
    val.append(float(row[len(row)-1]))
clf = linear_model.LinearRegression()
clf.fit(data, list(val))
print clf.score(data, list(val))
#matplotlib ploting
title('Imputed Residuals')
plot(val,[clf.predict(data[i]) - val[i] for i in range(len(data))], 'o')

show()

val, _ = stats.boxcox(np.array([x+1 for x in val]))
clf = linear_model.LinearRegression()
clf.fit(data, list(val))
print clf.score(data, list(val))
#matplotlib ploting
title('Imputed BoxCox Residuals')
plot(val,[clf.predict(data[i]) - val[i] for i in range(len(data))], 'o')

show()