import sys
import time
import cPickle

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from BinnedFisher import BinnedFisher

usePartialData=True

print "### Loading Data ###"
## if not usePartialData:
##     sig = np.loadtxt('data/signal.txt', delimiter=',')
##     bkg = np.loadtxt('data/qcd.txt', delimiter=',')

##     alldata =  np.concatenate( (sig, bkg), axis=0) 
##     #np.random.shuffle(alldata)

##     #alldata = alldata[0::10, :]
##     outfile = file('alldata.pkl', 'wb')
##     cPickle.dump(alldata, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
##     outfile.close()
##     sys.exit(0)

normImage = False

makePlot = False

alldata = cPickle.load( file('data/alldata_'+('norm_' if normImage else '')+ 'TRAIN.pkl', 'r') )


print "### Building Model ###"

X = alldata[:,1:]
y = alldata[:,0]

bfish = BinnedFisher( bins=[0.25, 0.5, 0.75, float('inf')] )

#bfish.fit(X, y, tol=[4.0e-3, 1.0e-3, 0.5e-3]) #old
if normImage:
    #bfish.fit(X, y, tol=[2.0e-3, 0.6e-3, 0.2e-3]) #good for normed, 10k per bin per label
    bfish.fit(X, y, tol=[1.0e-3, 0.75e-4, 0.1e-3]) #good for normed, 10k per bin per label

else:
    #bfish.fit(X, y, tol=[9.5e0, 11e0, 3.0e0]) #good-ish for non-normed, 10k per bin per label
    bfish.fit(X, y, tol=[2.5e0, 2.5e-1, 0.3e0]) #good-ish for non-normed, 10k per bin per label
    
outfile = file('trained_'+('norm_' if normImage else '')+ 'DR_fisher.pkl', 'wb')
cPickle.dump(bfish, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
outfile.close()




if makePlot:
    for ifish in range(len(bfish.comp)):
        fish = bfish.comp[ifish][::-1]

        print 'fish', ifish,'singular values:'
        print  bfish.fish[ifish].singular_vals
        
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        elem = fish.reshape(25,25)
        vmin = np.min(elem)
        vmax = np.max(elem)
    
        elem /= np.max(  [ abs(vmin), abs(vmax)] ) 
        vmin = np.min(elem)
        vmax = np.max(elem)
        
        cm_bi = colors.LinearSegmentedColormap.from_list('bi', 
                            [(0,'red'), (abs(vmin)/(vmax-vmin), 'white'),(1,'blue')])
        ret = ax.imshow(elem,
                        cmap=cm_bi,
                        interpolation='nearest',
                        origin='lower') #extent=[low, high, low, high],
        ax.set_title("Fisher "+str(ifish), size='xx-large')

    plt.show()






## t =  bfish.transform(X)

## s, bns = np.histogram(t[y==1], normed=True)
## b, bns = np.histogram(t[y==0], bins=bns, normed=True)

## x_cen = [ 0.5*(bns[i]+bns[i+1]) for i in range(len(bns)-1)]

## plt.figure()
## plt.plot(x_cen, s, color='g', linewidth=3)
## plt.plot(x_cen, b, color='b', linewidth=3)
## plt.show()




