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
                    distribution[j,i] = clusters[i,j]
    
    distribution /= np.sum(distribution,axis=1,keepdims=True)
    return distribution,cmat

def mutual_information(mat1,mat2):
    mask1 = np.sum(mat1,axis=1,keepdims=True)
    mask2 = np.sum(mat2,axis=1,keepdims=True)
    d1 = mat1[np.all(mask1, axis=1) & np.all(mask2, axis=1),:]
    d2 = mat2[np.all(mask1, axis=1) & np.all(mask2, axis=1),:]

    d1 /= np.sum(d1,axis=1,keepdims=True)
    d2 /= np.sum(d2,axis=1,keepdims=True)

    n1 = d1.shape[1]
    n2 = d2.shape[1]
    m = d1.shape[0]
    if m != d2.shape[0]:
        print(f'{m},{d2.shape[0]}')
        raise ValueError("Distributions must have the same number of rows")

    mutual_info = 0
    for i in range(n1):
        d11 = d1[:,i]
        for j in range(n2):
            d22 = d2[:,j]
            mi = 1/m*np.inner(d11,d22)*np.log(m*np.inner(d11,d22)/(np.sum(d11)*np.sum(d22))+1e-16)
            if np.isnan(mi):
                print(d11)
                print(d22)
            mutual_info += mi
    nmi = mutual_info/(0.5*(entropy(d1)+entropy(d2)))
    return mutual_info,nmi

def entropy(d):
    m = d.shape[0]
    n = d.shape[1]
    ent = 0
    for j in range(n):            
        ent += 1/m*np.sum(d[:,j])*np.log(np.sum(d[:,j])+1e-16)
    return ent
