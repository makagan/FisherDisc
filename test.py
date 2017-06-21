import numpy as np

from Fisher import Fisher
from BinnedFisher import BinnedFisher


#make random two classes
class0 = np.random.normal(1,1, (50,5) )
class1 = np.random.normal(-2,0.5, (50,5) )

X = np.vstack( (class0, class1) )
y = np.array( [0 for i in range(50)] + [1 for i in range(50)] )


f = Fisher()

f.fit(X, y, tol = 0.1)

print f.transform(X)


#after fit, can update tolerance
f.update_tol( tol = 0.01 )

print f.transform(X)



#add additional variable to binning, for BinnedFisher
v = np.array( [[0.25 for i in range(25)]+[0.75 for i in range(25)]+[0.25 for i in range(25)]+[0.75 for i in range(25)]] )


X = np.hstack( (v.T, X) )

bf = BinnedFisher( bins = [0.0,0.5,1.0] )

bf.fit(X,y, tol=[0.01, 0.01])

print bf.transform(X)



