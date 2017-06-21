import matplotlib.pyplot as plt

import numpy as np
import scipy as sc


def ROC( signal, background, label, cut_start=None, cut_end=None, cut_type=None):

    ## s = np.array( signal )
    ## b = np.array( background )
    ## l = np.array( label )

    ## if len(s.shape)==1:
    ##     s = np.array( [signal] )
    ## if len(b.shape)==1:
    ##     b = np.array( [background] )
    ## if len(l.shape)==0:
    ##     l = np.array( [label] )

    fig = plt.figure()

    if cut_type==None:
        cut_type=['g' for ic in range(len(signal)) ]
    
    for ivar in range(len(signal)):
        s_sort = np.sort( signal[ivar] )
        b_sort = np.sort( background[ivar] )

        #c_start=(0.0 if cut_start==None else cut_start)
        #c_end=  (1.0 if cut_end==None else cut_end)

        c_start=np.min( (s_sort[0], b_sort[0]) )
        c_end=  np.max( (s_sort[len(s_sort)-1], b_sort[len(b_sort)-1]) )
        
        if c_start==-float('inf'):
            c_start = -2*c_end

        print label[ivar], "min(", s_sort[0],  b_sort[0],  ")=", c_start
        print label[ivar], "max(", s_sort[-1], b_sort[-1], ")=", c_end
        
        s_eff=[]
        b_rej=[]

        n_points = 1000
        c_delta = (1.0*c_end - 1.0*c_start) / (1.0*n_points)
        for i in range(1000):
            cut = c_start + i*1.0*c_delta
            if cut_type[ivar]=='g':
                s_eff.append( 1.0*np.count_nonzero( s_sort > cut ) / (1.0*len(s_sort))  )
                b_count = np.count_nonzero( b_sort > cut )
            elif cut_type[ivar]=='l':
                s_eff.append( 1.0*np.count_nonzero( s_sort < cut ) / (1.0*len(s_sort))  )
                b_count = np.count_nonzero( b_sort < cut )
            b_rej.append(  (1.0*len(b_sort)) / (1.0 if b_count==0 else (1.0*b_count))  )

        #print s_eff
        plt.plot(s_eff,b_rej)

    plt.legend(label, loc='lower left', prop={'size':6})
    plt.yscale('log')
    #plt.show() 


    return


def Eff_vs_Var( disc, var, label, bins, cuts= None, eff_target=0.7  ):

    fig = plt.figure()

    
    bin_error=[]
    bin_center=[]
    for ibin in range(len(bins)-1):
        ierror = (bins[ibin+1] - bins[ibin])/2.0
        bin_error.append( ierror )
        bin_center.append( bins[ibin] +  ierror )

    for isamp in range(len(disc)):

        idisc = np.array(disc[isamp])
        ivar  = np.array(var[isamp])

        if cuts == None:
            cut_val = Get_Cut_Value(idisc, eff_target)
            #cut_val = np.sort(idisc)[ int((1.0-eff_target)*len(idisc)) ]
        else:
            cut_val = cuts[isamp]
        
        #sort_indices = np.argsort(disc[isamp])

        eff = []
        yerr = []
        for ibin in range(len(bins)-1):
            idisc_ibin = idisc[  (ivar>=bins[ibin]) * (ivar<bins[ibin+1]) ]
            n_tot = len(idisc_ibin)
            n_pass = np.count_nonzero( idisc_ibin > cut_val   )

            eff.append( (1.0*n_pass) / (1.0*n_tot) )
            yerr.append( (1.0/(1.0*n_tot)) * np.sqrt( n_pass * (1.0 - (1.0*n_pass) / (1.0*n_tot)))  )

            print bin_center[ibin], n_pass, n_tot, eff[ibin], yerr[ibin]
    
        plt.errorbar( bin_center, eff, xerr = bin_error, yerr = yerr)
            
    plt.legend(label, loc='best', prop={'size':6})

    return


def Get_Cut_Value(disc, eff_target):
    return np.sort(disc)[ int((1.0-eff_target)*len(disc)) ]
