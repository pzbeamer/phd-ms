import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.special import logit
import pandas as pd

def clusters_to_distribution(clusters):
    if isinstance(clusters,pd.Series):
        distribution = np.zeros((len(clusters.cat.categories.tolist()),len(clusters)))
        for i in range(len(clusters)):
            if isinstance(clusters.iloc[i],float):
                continue
            index = next(j for j in range(len(clusters.cat.categories.tolist())) if clusters.cat.categories.tolist()[j] == clusters.iloc[i])
            distribution[index,i] = 1
        cmat = distribution.transpose().copy()
            
    elif isinstance(clusters,np.ndarray):
        cmat = clusters.copy()
        distribution = np.zeros((clusters.shape[1],clusters.shape[0]))
        for i in range(clusters.shape[0]):  # Iterate over rows
            for j in range(clusters.shape[1]):  # Iterate over columns
                if clusters[i,j] < 0.01:
                    distribution[j,i] = 0
                else:
                    distribution[j,i] = 1
    
    distribution /= np.sum(distribution,axis=1,keepdims=True)
    return distribution,cmat
