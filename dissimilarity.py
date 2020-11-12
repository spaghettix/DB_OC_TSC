


__author__ = 'Stefano Mauceri'

__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numba
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import acf
from scipy.stats import wasserstein_distance as wsd
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel



# =============================================================================
# DISSIMILARITY MEASURES
# =============================================================================



def C(a1, a2, b, c):
    if a2 <= a1 <= b or a2 >= a1 >= b:
        return c
    else:
        return c + min(abs(a1-a2), abs(a1-b))



@numba.njit
def hist1d(x, binx):
    return np.histogram(x, binx)[0]



class dissimilarity(object):
    
 
    
    def __init__(self):
        pass



    def autocorrelation(self, a, b, **kwargs):
        # This fails if a time series is a constant.
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
        C[1:,1:] = cdist(a.reshape(-1,1), b.reshape(-1,1), 'euclidean')
        for i in range(1, length+1):
            for j in range(1, length+1):
                C[i, j] += min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
        return C[length, length]



    def EDR(self, a, b, eps, **kwargs):
        # Edit Distance on Real Sequences
        length_a, length_b = a.size, b.size
        C = np.full(shape=(length_a + 1, length_b + 1), fill_value=np.inf)
        C[:, 0] = np.arange(length_a + 1)
        C[0, :] = np.arange(length_b + 1)
        for i in range(1, length_a + 1):
            for j in range(1, length_b + 1):
                if np.abs(a[i-1] - b[j-1]) < eps:
                    C[i, j] = min(C[i, j - 1] + 1, C[i - 1, j] + 1, C[i - 1, j - 1] + 0)
                else:
                    C[i, j] = min(C[i, j - 1] + 1, C[i - 1, j] + 1, C[i - 1, j - 1] + 1)
        return C[length_a, length_b]



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



    def MSM(self, a, b, c_penalty, **kwargs):
        length_a, length_b = a.size, b.size
        D = np.zeros(shape=(length_a, length_b))
        D[0,0] = np.abs(a[0] - b[0])
        for i in range(1, length_a):
            D[i,0] = D[i-1, 0] + C(a[i], a[i-1], b[0], c_penalty)
        for i in range(1, length_b):
            D[0,i] = D[0, i-1] + C(b[i], a[0], b[i-1], c_penalty)
        for i in range(1, length_a):
            for j in range(1, length_b):
                D[i,j] = min(D[i-1,j-1] + np.abs(a[i] - b[j]),
                             D[i-1,j] + C(a[i], a[i-1], b[j], c_penalty),
                             D[i, j-1] + C(b[j], a[i], b[j-1], c_penalty))
        return D[length_a-1, length_b-1]



    def sigmoid(self, a, b, **kwargs):
        # gamma = 1 / ts_length
        return sigmoid_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=None, coef0=0) * -1



    def wasserstein(self, a, b, **kwargs):
        return wsd(a, b)



# =============================================================================
# THE END
# =============================================================================
