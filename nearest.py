from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from scipy.linalg import lstsq
from pylab import plot, title, show , legend
import csv
from numpy import argsort,sqrt
import numpy as np



def knn_search(x, D, K):
	""" find K nearest neighbours of data among D """
	ndata = D.shape[1]
	K = K if K < ndata else ndata
	# euclidean distances from the other points
	sqd = sqrt(((D - x[:,:ndata])**2).sum(axis=0))
	idx = argsort(sqd) # sorting
	# return the indexes of K nearest neighbours
	return idx[:K]



datafile = open('nomissingbycol.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
firstval = []
for row in datacsv:
	data.append([float(i) for i in row[:-1]])
	val.append(float(row[-1]))
	firstval.append(float(row[0]))

residuals = []

for i in range(len(data)):
	datacpy = data[:]
	datacpy.remove(data[i])

	knn = knn_search(np.array(data[i], ndmin=2), np.array(datacpy), 1)

	print(knn)

	neighbor = knn[0]

	# print(neighbor)
	prediction = val[neighbor]

	error = prediction - val[i]

	residuals.append(error)

plot(val, residuals, 'o')

show()
