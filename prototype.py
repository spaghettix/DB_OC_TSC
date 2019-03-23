


__author__ = 'Stefano Mauceri'

__email__ = 'stefano.mauceri@ucdconnect.ie'



# =============================================================================
# IMPORT
# =============================================================================



import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture



# =============================================================================
# PROTOTYPE METHODS
# =============================================================================



class prototype(object):

    
    
    def __init__(self):
        pass
        


    def all_training_samples(self, X, **kwargs):
        return X
    
    
    
    def borders(self, X, n, dissimilarity_matrix, **kwargs):
        idx = np.argmax(np.mean(dissimilarity_matrix, axis=0))
        k_furthest = [X[idx]]        
        mask = np.zeros(dissimilarity_matrix.shape, dtype=bool)
        mask[idx] = True
        dissimilarity_matrix = np.ma.array(dissimilarity_matrix, mask=mask)

        for i in range(1, n):
            idx = np.argmax(dissimilarity_matrix[:, idx], axis=0) 
            k_furthest += [X[idx]]
            dissimilarity_matrix.mask[idx] = True

        return np.array(k_furthest)



    def centers_gaussian_mixture(self, X, n, **kwargs):
        if X.shape[0] == 1:
            return X
        else:
            return GaussianMixture(n_components=n).fit(X).means_
    
    
    
    def centers_k_means(self, X, n, **kwargs):
        if X.shape[0] == 1:
            return X
        return KMeans(n_clusters=n).fit(X).cluster_centers_

    
    
    def closest(self, X, n, dissimilarity_matrix, **kwargs):
        avg_dist = np.mean(dissimilarity_matrix, axis=0)
        idx = avg_dist.argsort()[:n]
        return X[idx]

    
    
    def furthest(self, X, n, dissimilarity_matrix, **kwargs):
        avg_dist = np.mean(dissimilarity_matrix, axis=0)
        idx = avg_dist.argsort()[-n:]
        return X[idx]



    def percentiles(self, X, n, **kwargs):
        if n > 1:
            percentages = np.linspace(0, 100, n, dtype=int)
        else:
            percentages = [50]
        return np.array([np.percentile(X, q=i, axis=0, interpolation='linear') for i in percentages])



    def random(self, X, n, **kwargs):
        idx = np.random.choice(range(X.shape[0]), size=n, replace=False)
        return X[idx]
    


    def support_vectors(self, X, n, **kwargs):
        model = OneClassSVM(gamma=X.shape[1], nu=1/X.shape[0])
        model.fit(X)
        sv = model.support_vectors_
        distance_from_hyperplane = model.decision_function(sv).reshape(-1)
        idx = np.argsort(np.abs(distance_from_hyperplane))[:n]
        return sv[idx, :]



# =============================================================================
# END
# =============================================================================


