from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from scipy.linalg import lstsq
from pylab import plot, title, show , legend
import csv
"""
datafile = open('communities.data', 'r')
data = csv.reader(datafile)
writefile = open('explanatory.data', 'w')
writer = csv.writer(writefile)
for row in data:
    writer.writerow(row[5:])


datafile = open('explanatory.data', 'r')
data = csv.reader(datafile)
writefile = open('nomissing.data', 'w')
writer = csv.writer(writefile)
for row in data:
    write = True
    for i in range(len(row)):
        if row[i] == '?':
            write = False
    if write:
        writer.writerow(row)
"""

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
        if row[i] is '?':
            stuff[i] = -1
for row in data:
    newdat = []
    for i in stuff:
        if stuff is not -1:
            newdat.append(row[i])
    writer.writerow(newdat)




"""
datafile = open('nomissing.data', 'r')
datacsv = csv.reader(datafile)
data = []
val = []
for row in datacsv:
    data.append(row[:-1])
    val.append(row[-1])

ret = lstsq(data, val)

print ret
"""


"""
#Linear regression example
# This is a very simple example of using two scipy tools 
# for linear regression, polyfit and stats.linregress

#Sample data creation
#number of points 
n=50
t=linspace(-5,5,n)
#parameters
a=0.8
b=-4
x=polyval([a,b],t)
#add some noise
xn=x+randn(n)

#Linear regressison -polyfit - polyfit can be used other orders polys
(ar,br)=polyfit(t,xn,1)
xr=polyval([ar,br],t)
#compute the mean square error
err=sqrt(sum((xr-xn)**2)/n)

print('Linear regression using polyfit')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, ms error= %.3f' % (a,b,ar,br,err))

#matplotlib ploting
title('Linear Regression Example')
plot(t,x,'g.--')
plot(t,xn,'k.')
plot(t,xr,'r.-')
legend(['original','plus noise', 'regression'])

show()

#Linear regression using stats.linregress
(a_s,b_s,r,tt,stderr)=stats.linregress(t,xn)
print('Linear regression using stats.linregress')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, std error= %.3f' % (a,b,a_s,b_s,stderr))
"""