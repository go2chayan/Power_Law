# HW2: Plot rank vs. frequency on a log-log scale, 
# and fit parameters for the generalized power law. 
#
# Coded by Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

# Read the data and create a  count dictionary
def getcountdic(dataFile):
    dic,totcount={},0
    print 'please wait while it finishes reading the data ...'
    with open(dataFile) as inp:
        for line in inp:
            for aword in line.strip().split(' '):
                if aword.lower() in dic:
                    dic[aword.lower()]+=1
                else:
                    dic[aword.lower()]=1
                totcount += 1
    return dic,totcount

## Calculate the ccdf(c) = p(X>x) from data
#def dataccdf(thresh,data):
#    N = float(len(data))
#    thresh = np.sort(thresh,kind='heapsort')
#    data = np.sort(data,kind='heapsort')
#    # I am using binery search for faster counting
#    datccdf = (N - np.searchsorted(data,thresh))
#    datccdf = datccdf/N
#    #print max(datccdf),min(datccdf),N, max(np.searchsorted(data,thresh))
#    return datccdf
    
# Calculate the ccdf using model
def modelccdf(thresh,alpha,xmin):
    return zeta(alpha,thresh)/float(zeta(alpha,xmin))
    
def main():
    # Count words
    dic,totcount=getcountdic('training.eng')    
    x = np.array(dic.values()) # frequencies of each word
    print 'Total Tokens:',totcount,'Total Types:', len(x)
    
    # We shall do this for various bin sizes
    for subfignumber, totalbin in enumerate([1000,10000,len(x)]):
        ##############################################################
        # Figure 1: Parameter estimation using Least Square Regression 
        ##############################################################
        # with various bin sizes
        # Estimate pmf from histogram
        px,xbins = np.histogram(x,bins=totalbin)
        xbins = xbins[:-1]
        # normalizing px in a way so that it doesn't contain 0
        # this is necessary because later we'll take log of px
        px = (px+1)/np.sum(px+1.0)        
        
        # Least square regression for log(px) vs log(xbins)
        slope_lsr,intercept_lsr = np.polyfit(np.log10(xbins),np.log10(px),1)
        print 'binsize:',totalbin,'slope(LS):',slope_lsr,'intercept(LS):',intercept_lsr        
        y_lsr = 10.**(np.log10(xbins)*slope_lsr+intercept_lsr)
        # Show a plot of pdf of x vs x
        plt.figure(1)
        plt.subplot(3,1,subfignumber+1)
        plt.loglog(xbins,px,'bo')
        plt.hold('on')
        plt.loglog(xbins,y_lsr,'r-')
        plt.hold('off')
        plt.xlim([min(xbins),max(xbins)])
        plt.ylabel('PMF of X, p(X=x)')  
        if subfignumber==2:
            plt.xlabel('Word Frequencies, x')          
        else:
            plt.tick_params(bottom='off',labelbottom='off')
        plt.title('p(X) vs x in log-log scale, alpha='+str(-1*slope_lsr)+\
        ', bins='+str(totalbin))
        plt.legend(['data','least square line'])
    
        # Use of Kolmogorov-Smirnov test to select a good xmin
        currKS = np.inf
        best_xmin = 100.0
        for xmin in [1.0,5.0,10.0,50.0,100.0,500,1e3,5e3,1e4,5e4,1e5]:
            # Now we are calculating the parameter using MLE
            alpha = 1.0+np.size(x[x>=xmin])*(1.0/(np.sum(np.log(x[x>=xmin]/(xmin-0.5)))))

            # Calculate the ccdf for KS test below
            px,xbins = np.histogram(x[x>=xmin],bins=totalbin)
            xbins = xbins[:-1]
            px = px/float(np.sum(px))        
            dataccdf = 1 - np.cumsum(px)

            # Kolmogorov-Smirnov test. Choose the minimum KS value
            KS = np.max(np.abs(modelccdf(xbins,alpha,xmin) - dataccdf))
            if KS < currKS:
                currKS = KS
                best_xmin = xmin
            print 'alpha',alpha,'xmin',xmin,'--> KS:',KS

        # Calculate the ccdf again from the data using best xmin
        px,xbins = np.histogram(x[x>=best_xmin],bins=totalbin)
        xbins = xbins[:-1]
        px = px/float(np.sum(px))        
        dataccdf = 1 - np.cumsum(px)
        
        # Calculate alpha again using the best xmin
        alpha = 1.0 +np.size(x[x>=best_xmin])*(1.0/(np.sum(np.log(x[x>=best_xmin]/(best_xmin-0.5)))))

        # We need slope and intercept to draw the line
        slope_mle = -1.0*alpha
        intercept_mle = -1*np.log(zeta(alpha,best_xmin))
        y_mle = 10.**(np.log10(xbins)*slope_mle + intercept_mle)
        print 'binsize:',totalbin,'alpha',alpha
        print 'slope(MLE):',slope_mle,'intercept(MLE):',intercept_mle
        
        ####################################################################
        # Figure 2: Plot of p(x) vs x for best xmin from Kolmogorov-Smirnov
        ####################################################################  
        plt.figure(2)
        plt.subplot(3,1,subfignumber+1)
        plt.loglog(xbins,px,'bo')
        plt.hold('on')
        plt.loglog(xbins,y_mle,'r-')
        plt.hold('off')
        plt.xlim([min(xbins),max(xbins)])
        plt.ylabel('PMF of X, p(X=x)') 
        if subfignumber==2:
            plt.xlabel('Word Frequencies, x')
        else:
            plt.tick_params(bottom='off',labelbottom='off')
        plt.title('alpha='+str(alpha)+', bins='+str(totalbin)+', xmin='+str(best_xmin))
        plt.legend(['data','best fit line using MLE'],loc='upper right')

        ####################################################################
        # Figure 3: Plot of CCDF vs x for best xmin from Kolmogorov-Smirnov
        ####################################################################    
        plt.figure(3)
        plt.subplot(3,1,subfignumber+1)
        plt.loglog(xbins,dataccdf,'bo')
        plt.hold('on')
        sx = modelccdf(xbins,alpha,best_xmin)
        plt.loglog(xbins,sx,'r-')
        plt.hold('off')
        plt.xlim([min(xbins),max(xbins)])
        plt.ylabel('CCDF of X, p(X>x)') 
        if subfignumber==2:
            plt.xlabel('Word Frequencies, x')
        else:
            plt.tick_params(bottom='off',labelbottom='off')
        plt.title('alpha='+str(alpha)+', bins='+str(totalbin)+', xmin='+str(best_xmin))
        plt.legend(['data','best fit model of CCDF using MLE'],loc='lower left')

    ##################################
    # Figure 3: Best fit in my opinion
    ##################################
    best_xmin=50
    px,xbins = np.histogram(x[x>=best_xmin],bins=totalbin)
    xbins = xbins[:-1]
    px = px/float(np.sum(px))        
    dataccdf = 1 - np.cumsum(px)
    alpha = 1.0 +np.size(x[x>=best_xmin])*(1.0/(np.sum(np.log(x[x>=best_xmin]/(best_xmin-0.5)))))
    
    plt.figure(4)
    plt.subplot(211)
    slope_mle = -1.0*alpha
    intercept_mle = -1*np.log(zeta(alpha,best_xmin))
    y_mle = 10.**(np.log10(xbins)*slope_mle + intercept_mle)
    plt.loglog(xbins,px,'bo')
    plt.hold('on')
    plt.loglog(xbins,y_mle,'r-')
    plt.hold('off')
    plt.xlim([min(xbins),max(xbins)])
    plt.ylabel('PMF of X, p(X=x)') 
    plt.xlabel('Word Frequencies, x')
    plt.title('Best fit in my opinion. alpha='+str(alpha)+', bins='+str(totalbin)+', xmin='+str(best_xmin))
    plt.legend(['data','best fit line using MLE'],loc='upper right')

    # Show a plot of CCDF of x vs x
    plt.subplot(212)
    plt.loglog(xbins,dataccdf,'bo')
    plt.hold('on')
    sx = modelccdf(xbins,alpha,best_xmin)
    plt.loglog(xbins,sx,'r-')
    plt.hold('off')
    plt.xlim([min(xbins),max(xbins)])
    plt.ylabel('CCDF of X, p(X>x)') 
    if subfignumber==2:
        plt.xlabel('Word Frequencies, x')
    else:
        plt.tick_params(bottom='off',labelbottom='off')
    plt.title('Best fit in my opinion. alpha='+str(alpha)+', bins='+str(totalbin)+', xmin='+str(best_xmin))
    plt.legend(['data','best fit model of CCDF using MLE'],loc='lower left')
    
    plt.show()
    
    
if __name__=='__main__':
    main()
