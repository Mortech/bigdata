from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from scipy.linalg import lstsq
from pylab import plot, title, show , legend
import csv
from numpy import argsort,sqrt
import numpy as np
from sklearn import neighbors, datasets

def impute(training, values, testing):
	clf = neighbors.KNeighborsClassifier(3, weights="distance")
	clf.fit(np.array(training), np.array(values))
	return clf.predict(np.array(testing))

def training(data2, data, col):
	ret_val = []
	for x in range(len(data)):
		if data[x][col] != '?':
			ret_val.append(data2[x])
	return ret_val

def valuer(data2, data, col):
	ret_val = []
	for x in range(len(data)):
		if data[x][col] != '?':
			ret_val.append(float(data[x][col]))
	return ret_val


datafile = open('explanatory.data', 'r')
datacsv = csv.reader(datafile)

datafile2 = open('nomissingbycol.data', 'r')
datacsv2 = csv.reader(datafile2)


data2 = []
for row in datacsv2:
	data2.append([float(i) for i in row[:]])


data = []

for row in datacsv:
	# new_row = []
	# for element in row:
	# 	if element == '?':
	# 		new_row.append(impute(hsagdflg))
	# 	else:
	data.append([i for i in row[:]])

out_data = []

for x in range(len(data)):
	new_row = []
	for y in range(len(data[0])):
		if(data[x][y] == '?'):
			new_row.append(float(impute(training(data2, data, y), valuer(data2, data, y), data2[x])))
		else:
			new_row.append(float(data[x][y]))
	out_data.append(new_row)

writefile = open('imputed.data', 'w')
writer = csv.writer(writefile)

for row in out_data:
    writer.writerow(row)
