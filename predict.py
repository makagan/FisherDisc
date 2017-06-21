import sys
import time
import cPickle

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotting

from BinnedFisher import BinnedFisher

def predict( normImage = True, saveFile = False, makePlot = True):

    bfish = cPickle.load( file('trained_'+('norm_' if normImage else '')+ 'DR_fisher.pkl', 'r') )
    
    testfile = file('data/alldata_'+('norm_' if normImage else '')+ 'TEST.pkl', 'r')
    data = cPickle.load( testfile )
    spec = cPickle.load( testfile )
    testfile.close()

    drbin = (data[:,1]>=0.5)*(data[:,1]<0.75)
    data = data[drbin,:]
    spec = spec[drbin,:]
    
    X = data[:,1:]
    y = data[:,0]
    dr = data[:,1]
    tau21 = spec[:,2]

    bfish.update_tol(tol=[1.0e-3, 0.75e-6, 0.1e-3]) # normed
    #bfish.update_tol(tol=[2.5e0, 2.5e-1, 0.3e0]) # non-normed
    
    t =  bfish.transform(X, return_ll=False)
    
    print t




    if saveFile:
        out_arr = np.hstack( (np.array([y]).T, spec, np.array([dr]).T, np.array([t]).T) )
        print "out shape=", out_arr.shape
        np.savetxt('TEST_predict/TEST_'+('norm_' if normImage else '')+ 'DR_Fisher.txt', out_arr, delimiter=',')

    #sys.exit(0)




    if makePlot:
        s, bns = np.histogram(t[y==1], normed=True)
        b, bns = np.histogram(t[y==0], bins=bns, normed=True)
        
        x_cen = [ 0.5*(bns[i]+bns[i+1]) for i in range(len(bns)-1)]
        
        plt.figure()
        plt.plot(x_cen, s, color='g', linewidth=3)
        plt.plot(x_cen, b, color='b', linewidth=3)
        #plt.show()
    
        Sigs = [ t[y==1], tau21[y==1] ]
        Bkgs = [ t[y==0], tau21[y==0] ]
        Labs = ["Fisher","Tau21"]
        cut_type=['g','l']

        plotting.ROC(Sigs, Bkgs, Labs, cut_type=cut_type)

        for ifish in range(len(bfish.comp)):
            fish = bfish.fish[ifish].w_[0][::-1]
        
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


if __name__=="__main__":
	predict()
