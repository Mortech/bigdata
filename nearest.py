from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from scipy.linalg import lstsq
from pylab import plot, title, show , legend
import csv
from numpy import argsort,sqrt
import numpy as np
from sklearn import neighbors, datasets


def knn_search(x, D, K):
	""" find K nearest neighbours of data among D """
	ndata = D.shape[1]
	K = K if K < ndata else ndata
	# euclidean distances from the other points
	sqd = sqrt(((D - x[:,:ndata])**2).sum(axis=0))
	idx = argsort(sqd) # sorting
	# return the indexes of K nearest neighbours
	return idx[:K]



# datafile = open('nomissingbycol.data', 'r')
datafile = open('imputed.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
firstval = []
for row in datacsv:
	data.append([float(i) for i in row[:-1]])
	val.append(float(row[-1]))
	firstval.append(float(row[0]))

residuals = []

tot = 0
length = 0

for i in range(len(data)):
	datacpy = data[:]
	datacpy.pop(i)

	# knn = knn_search(np.array(data[i], ndmin=2), np.array(datacpy), 1)

	# print(knn)

	# neighbor = knn[0]

	# print(neighbor)
	clf = neighbors.KNeighborsClassifier(3, weights="distance")

	valcpy = val[:]
	valcpy.pop(i)
	# clf.fit(np.array(datacpy), np.array(val[:i-1].extend(val[i+1:])))
	clf.fit(np.array(datacpy), np.array(valcpy))



	# prediction = val[neighbor]

	prediction = clf.predict(np.array(data[i]))

	error = prediction - val[i]
	tot += error**2
	length += 1

	residuals.append(error)


print str(float(tot) / float(length))
plot(val, residuals, 'o')

show()

val, _ = stats.boxcox(np.array([x+1 for x in val]))
val = list(val)

residuals = []

tot = 0
length = 0

for i in range(len(data)):
	datacpy = data[:]
	datacpy.pop(i)

	# knn = knn_search(np.array(data[i], ndmin=2), np.array(datacpy), 1)

	# print(knn)

	# neighbor = knn[0]

	# print(neighbor)
	clf = neighbors.KNeighborsClassifier(3, weights="distance")

	valcpy = val[:]
	valcpy.pop(i)
	# clf.fit(np.array(datacpy), np.array(val[:i-1].extend(val[i+1:])))
	clf.fit(np.array(datacpy), np.array(valcpy))



	# prediction = val[neighbor]

	prediction = clf.predict(np.array(data[i]))

	error = prediction - val[i]
	tot += error**2
	length += 1

	residuals.append(error)


print str(float(tot) / float(length))
plot(val, residuals, 'o')

show()


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

tot = 0
length = 0

for i in range(len(data)):
	datacpy = data[:]
	datacpy.pop(i)

	# knn = knn_search(np.array(data[i], ndmin=2), np.array(datacpy), 1)

	# print(knn)

	# neighbor = knn[0]

	# print(neighbor)
	clf = neighbors.KNeighborsClassifier(3, weights="distance")

	valcpy = val[:]
	valcpy.pop(i)
	# clf.fit(np.array(datacpy), np.array(val[:i-1].extend(val[i+1:])))
	clf.fit(np.array(datacpy), np.array(valcpy))



	# prediction = val[neighbor]

	prediction = clf.predict(np.array(data[i]))

	error = prediction - val[i]
	tot += error**2
	length += 1

	residuals.append(error)


print str(float(tot) / float(length))
plot(val, residuals, 'o')

show()

val, _ = stats.boxcox(np.array([x+1 for x in val]))
val = list(val)

residuals = []

tot = 0
length = 0

for i in range(len(data)):
	datacpy = data[:]
	datacpy.pop(i)

	# knn = knn_search(np.array(data[i], ndmin=2), np.array(datacpy), 1)

	# print(knn)

	# neighbor = knn[0]

	# print(neighbor)
	clf = neighbors.KNeighborsClassifier(3, weights="distance")

	valcpy = val[:]
	valcpy.pop(i)
	# clf.fit(np.array(datacpy), np.array(val[:i-1].extend(val[i+1:])))
	clf.fit(np.array(datacpy), np.array(valcpy))



	# prediction = val[neighbor]

	prediction = clf.predict(np.array(data[i]))

	error = prediction - val[i]
	tot += error**2
	length += 1

	residuals.append(error)


print str(float(tot) / float(length))
plot(val, residuals, 'o')

show()