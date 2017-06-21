import sys
import time

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


from Fisher import Fisher

__all__ = ['BinnedFisher']


#####################################################################################################################
#NOTE TO SELF:
# np.inner(A,B) sums over last indices, i.e. = A[i,j]*B[k,j]
# so if you want to do A*B, you should do np.inner(A, B.T)
# Also, np.inner is faster than np.dot
#####################################################################################################################


class BinnedFisher(BaseEstimator, ClassifierMixin, TransformerMixin):

    
    def __init__(self, norm_covariance = True, n_components=None, priors=None, bins = [-float('inf'), float('inf')] ):
        
        self.nbins = len(bins)-1
        self.bins = np.sort(bins)
        self.bin_trained = [False for i in range(self.nbins)]

        self.fish = [ Fisher(norm_covariance, n_components, priors) for i in range(self.nbins) ]


    def fit(self, X, y, tol=[1.0e-4], store_covariance=False,  do_smooth_reg=False, cov_class=None, cov_power=1, entries_per_ll_bin = 10):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(tol)==1:
            tol = [tol for i in range(self.nbins)]
        elif len(tol) != self.nbins:
            print "tol must have length 1 or nbins. exiting"
            sys.exit(2)

        self.tol = tol
        self.do_smooth_reg = do_smooth_reg
        self.cov_class = cov_class
        self.cov_power = cov_power
        self.entries_per_ll_bin = entries_per_ll_bin
        self.comp = []
        self.ll_sig = []
        self.ll_bkg = []

        self.ll_bin_edges = []


        for i in range(self.nbins):
            print "Starting fit for bin", i
            ts = time.time()
            
            low, high = self.bins[i], self.bins[i+1]

            the_entries = (X[:,0] >=low) * (X[:,0] <high)

            if len(the_entries) == 0:
                print ("Warning: no entries in bin=[%d , %d], not training!", low, high)
                continue

            Xi = X[ the_entries, 1:]

            yi = y[ the_entries ]

            self.fish[i].fit(Xi, yi, tol=self.tol[i], 
                            do_smooth_reg=self.do_smooth_reg, 
                            cov_class=self.cov_class,
                            cov_power=self.cov_power, store_covariance=True)
            
            self.comp.append( self.fish[i].w_[0] ) # should be normed in fisher.py with linalg.norm(self.fish[i].w_[0])


            #now making log-likelihood
            sigs = np.sort(self._transform_bin(Xi[ yi==1 ], bin_number=i, override=True).flatten())
            bkgs = np.sort(self._transform_bin(Xi[ yi==0 ], bin_number=i, override=True).flatten())
            
            
            self.ll_bin_edges.append( bkgs[0::self.entries_per_ll_bin] )
            self.ll_bin_edges[i][0]  = np.minimum(bkgs[0], sigs[0])
            self.ll_bin_edges[i][-1] = np.maximum(bkgs[-1], sigs[-1])

            sig_hist, temp = np.histogram( sigs, bins=self.ll_bin_edges[i], normed=True)
            bkg_hist, temp = np.histogram( bkgs, bins=self.ll_bin_edges[i], normed=True)

            #set any zero entries to 1e-4 so that log(sig_hist) is always well defined
            sig_hist[ sig_hist==0 ] = np.repeat(1e-4, np.count_nonzero(sig_hist==0))

            self.ll_sig.append( np.log(sig_hist) )
            self.ll_bkg.append( np.log(bkg_hist) )

            self.bin_trained[i] = True

            print 'Fitting bin %d took %d seconds' % (i, time.time() - ts)


        return self

    def update_tol(self, tol):
        if len(tol)==1:
            tol = [tol for i in range(self.nbins)]
        elif len(tol) != self.nbins:
            print "tol must have length 1 or nbins. exiting"
            sys.exit(2)

        for i in range(self.nbins):
             self.fish[i].update_tol(tol[i])

        return self

    def transform(self, X, return_ll=False):
        t, l = self._transform(X)
        if return_ll:
            return l
        else:
            return t
        

    def _transform(self, X):
        
        X = np.asarray(X)

        out  = np.zeros( X.shape[0] ) # default transformed value is 0
        llout = np.ones( X.shape[0] ) # default ll value is 1 (i.e. equal prob)

        for i in range(self.nbins):
            if not self.bin_trained[i]:
                print ("bin %d not trained! Can't transform before running fit!", i)
                sys.exit(2)
            
            low, high = self.bins[i], self.bins[i+1]
            the_entries = (X[:,0] >=low) * (X[:,0] <high)

            Xi = X[ the_entries, 1:]

            out[ the_entries ] = self._transform_bin(Xi, i) #self.fish[i].transform(Xi)

            llout[ the_entries ] = self._eval_ll_bin( out[ the_entries ].flatten(), i)

        return out, llout
    

    def _transform_bin(self, Xi, bin_number, override=False):
        '''
        ony works after binned var stripped off
        '''

        if bin_number < 0 or bin_number > (self.nbins-1):
            print ("bin number must be between 0 and %d", self.nbins-1)
            sys.exit(2)

        if not override and not self.bin_trained[bin_number]:
            print ("bin %d not trained! Can't transform before running fit!", i)
            sys.exit(2)
        
        Xi = np.asarray(Xi)

        out  = self.fish[bin_number].transform(Xi)
        
        return out

    
    def _eval_ll_bin(self, Ti, bin_number):
        '''
        only works on transformed data
        '''

        if bin_number < 0 or bin_number > (self.nbins-1):
            print ("bin number must be between 0 and %d", self.nbins-1)
            sys.exit(2)

        if not self.bin_trained[bin_number]:
            print ("bin %d not trained! Can't transform before running fit!", i)
            sys.exit(2)

        Ti = np.asarray(Ti)

        # anything not found, gets value of 1
        llout = np.ones( Ti.shape[0] )

        for i in range(len(self.ll_bin_edges[bin_number]) - 1):
            the_entries = (Ti >= self.ll_bin_edges[bin_number][i]) * (Ti < self.ll_bin_edges[bin_number][i+1])            
            llout[ the_entries ] = np.repeat( self.ll_sig[bin_number][i] / self.ll_bkg[bin_number][i], np.count_nonzero(the_entries))

        return llout


        
