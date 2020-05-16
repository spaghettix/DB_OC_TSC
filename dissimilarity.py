


__author__ = 'Stefano Mauceri'

__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numba
import numpy as np
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf
from scipy.stats import wasserstein_distance as wsd
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel



# =============================================================================
# DISSIMILARITY MEASURES
# =============================================================================



@numba.njit
def hist1d(x, binx):
    return np.histogram(x, binx)[0]



class dissimilarity(object):
    
 
    
    def __init__(self):
        pass



    def autocorrelation(self, a, b, **kwargs):
        lags = int(a.shape[0]) - 1
        coeff = np.geomspace(1, 0.001, lags)
        try:
            return self.euclidean(acf(a, nlags=lags)[1:] * coeff, acf(b, nlags=lags)[1:] * coeff)
        except:
            return 1E5
        


    def chebyshev(self, a, b, **kwargs):
        return np.abs(a-b).max()
    
    
 
    def cityblock(self, a, b, **kwargs):
        return np.abs(a-b).sum()



    def cosine(self, a, b, **kwargs):
        return 1 - (np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))



    def DTW(self, a, b, **kwargs):
        # Dynamic Time Warping
        length = a.shape[0]
        C = np.zeros((length + 1, length + 1))
        C[0, 1:] = np.inf
        C[1:, 0] = np.inf
        C[1:,1:] = ssd.cdist(a.reshape(-1,1), b.reshape(-1,1), 'euclidean')
        for i in range(1, length+1):
            for j in range(1, length+1):
                C[i, j] += min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
        return C[length, length]



    def EDR(self, a, b, eps, **kwargs):
        # Edit Distance on Real Sequences
        return (np.abs(a-b) > eps).sum()



    def euclidean(self, a, b, **kwargs):
        return np.sqrt(np.square(np.abs(a-b)).sum())



    def kullback_leibler(self, a, b, **kwargs):
        ab = np.concatenate([a, b])
        bins = np.linspace(ab.min(), ab.max(), 10)
        a, b = hist1d(a, bins) + 1, hist1d(b, bins) + 1
        return entropy(a, b) + entropy(b, a)



    def gaussian(self, a, b, **kwargs):
        # gamma = 1 / ts_length
        return rbf_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=None) * -1



    def sigmoid(self, a, b, **kwargs):
        # gamma = 1 / ts_length
        return sigmoid_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=None, coef0=0) * -1



    def wasserstein(self, a, b, **kwargs):
        return wsd(a, b)



# =============================================================================
# END
# =============================================================================


