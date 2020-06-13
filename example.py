


__author__ = 'Stefano Mauceri'

__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import os
import numpy as np
from prototype import prototype
import scipy.spatial.distance as ssd
from dissimilarity import dissimilarity
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors



# =============================================================================
# EXAMPLE
# =============================================================================



# LOAD DATA

dataset_name = 'Plane'

path = os.path.join(os.getcwd(), 'data', dataset_name)

X_train = np.load(os.path.join(path, f'{dataset_name}_X_TRAIN.npy'))
Y_train = np.load(os.path.join(path, f'{dataset_name}_Y_TRAIN.npy'))
X_test = np.load(os.path.join(path, f'{dataset_name}_X_TEST.npy'))
Y_test = np.load(os.path.join(path, f'{dataset_name}_Y_TEST.npy'))



# ADAPT DATA TO ONE-CLASS CLASSIFICATION

print('AVAILABLE CLASSES: ', np.unique(Y_train))

positive = 1 # Choose a positive class
print('POSITVE CLASS: ', positive)
X_train = X_train[(Y_train == positive)]
Y_test = (Y_test == positive).astype(np.int8)



# SELECT DISS. MEASURE and PROTOTYPE METHOD

D, P = dissimilarity(), prototype()

Dissimilarity = D.kullback_leibler
diss_params = {}
# OR
#Dissimilarity = D.EDR # This is fairly slow
#diss_params = {'eps':0.25} # Threshold on distance for EDR computation

Prot_method = P.borders
# OR
#Prot_method = P.centers_k_means



# GET DISSIMILARITY MATRIX

# Some prototype methods eg "centers_k_means" do not require
# the computation of the dissimilarity matrix

Diss_Matrix = ssd.cdist(X_train, X_train, metric=Dissimilarity, **diss_params)



# GET PROTOTYPES

n = 2 # number of prototypes we want to get

Prototypes = Prot_method(X_train, n, Diss_Matrix)
# OR
#Prototypes = Prot_method(X_train, n)



# GET DISSIMILARITY-BASED REPRESENTATION

DBR_X_train = ssd.cdist(X_train, Prototypes, Dissimilarity, **diss_params)
DBR_X_test = ssd.cdist(X_test, Prototypes, Dissimilarity, **diss_params)



# GET AUROC

Classifier = NearestNeighbors(n_neighbors=1)
Classifier.fit(DBR_X_train)

Test_scores = Classifier.kneighbors(DBR_X_test)[0] * -1 
# Test scores are multiplied by -1 because the ROC curve expects that 
# more is better while in terms of dissimilarities less is better.

fpr, tpr, _ = roc_curve(Y_test, Test_scores, pos_label=1)
AUROC = auc(fpr, tpr) * 100

print('AUROC:', round(AUROC, 1))



# =============================================================================
# THE END
# =============================================================================
