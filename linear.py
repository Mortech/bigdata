from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from numpy.linalg import lstsq
from pylab import plot, title, show , legend
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
"""
datafile = open('communities.data', 'r')
data = csv.reader(datafile)
writefile = open('explanatory.data', 'w')
writer = csv.writer(writefile)
for row in data:
    writer.writerow(row[5:])


datafile = open('explanatory.data', 'r')
data = csv.reader(datafile)
writefile = open('nomissingbyrow.data', 'w')
writer = csv.writer(writefile)
for row in data:
    write = True
    for i in range(len(row)):
        if row[i] == '?':
            write = False
    if write:
        writer.writerow(row)


datafile = open('explanatory.data', 'r')
datacsv = csv.reader(datafile)
writefile = open('nomissingbycol.data', 'w')
writer = csv.writer(writefile)
data = []
stuff = None
for row in datacsv:
    data.append(row)
    if stuff is None:
        stuff = [i for i in range(len(row))]
    for i in range(len(row)):
        if row[i] == '?':
            stuff[i] = -1
for row in data:
    newdat = []
    for i in stuff:
        if i != -1:
            newdat.append(row[i])
    writer.writerow(newdat)




"""
datafile = open('nomissingbycol.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
firstval = []
for row in datacsv:
    data.append([float(row[i]) for i in range(len(row)-1)])
    val.append(float(row[len(row)-1]))
    firstval.append(float(row[0]))
clf = linear_model.LinearRegression()
clf.fit(data, val)
print('Coefficients: \n', clf.coef_)
#matplotlib ploting
title('Residuals')
plot(val,[val[i] - clf.predict(data[i]) for i in range(len(data))], 'o')

show()