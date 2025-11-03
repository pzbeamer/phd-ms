import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.special import logit
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import trange
from tqdm.contrib import itertools

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

    d1 = mat1.copy()
    d2 = mat2.copy()

    mask1 = np.sum(d1,axis=1,keepdims=True)
    mask2 = np.sum(d2,axis=1,keepdims=True)
    
    d1[np.all(mask1, axis=1),:] /= np.sum(d1[np.all(mask1, axis=1),:],axis=1,keepdims=True)
    d2[np.all(mask2, axis=1),:] /= np.sum(d2[np.all(mask2, axis=1),:],axis=1,keepdims=True)

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
            mutual_info += mi
    nmi = mutual_info/(0.5*(entropy(d1)+entropy(d2)))
    return mutual_info,nmi

def entropy(d):
    m = d.shape[0]
    n = d.shape[1]
    ent = 0
    for j in range(n):            
        ent -= 1/m*np.sum(d[:,j])*np.log(1/m*np.sum(d[:,j])+1e-16)
    return ent
def wasserstein_metric(ground_truth_distributions,multiscale_distributions,dmat):
    ma = np.max(dmat)
    M = dmat/ma
    g_cost = np.zeros(shape=(ground_truth_distributions.shape[0],multiscale_distributions.shape[0]))
    print('computing wasserstein distances...')
    for i1, i2 in itertools.product(range(ground_truth_distributions.shape[0]), range(multiscale_distributions.shape[0])):
        g = ground_truth_distributions[i1,:]
        m = multiscale_distributions[i2,:]
        d = ot.emd2(g,m,M)
        g_cost[i1,i2] = d
    
    costs = list(np.min(g_cost[i1,:])*ma for i1 in range(g_cost.shape[0]))
    matches = list(np.argmin(g_cost[i1,:]) for i1 in range(g_cost.shape[0]))
    return costs,matches

def nmi_metric(gmat,mmat,its):
    mask1 = np.sum(gmat,axis=1,keepdims=True)
    mask2 = np.sum(mmat,axis=1,keepdims=True)
    
    d1 = gmat[np.all(mask1, axis=1) & np.all(mask2, axis=1),:]
    d2 = mmat[np.all(mask1, axis=1) & np.all(mask2, axis=1),:]
    
    mask3 = np.sum(d1,axis=0,keepdims=True)
    mask4 = np.sum(d2,axis=0,keepdims=True)
    d1 = d1[:,np.all(mask3, axis=0)]
    d2 = d2[:,np.all(mask4, axis=0)]


        
    overlap = d1.T @ d2
    for i in range(d1.shape[1]):
        for j in range(d2.shape[1]):
            overlap[i,j] /= (np.linalg.norm(d1[:,i])**2 + np.linalg.norm(d2[:,j])**2 - np.inner(d1[:,i],d2[:,j])+1e-16)
            #overlap[i,j] = 1/d1.shape[0]*np.inner(d1[:,i],d2[:,j])*np.log(d1.shape[0]*np.inner(d1[:,i],d2[:,j])/(np.sum(d1[:,i])*np.sum(d2[:,j]))+1e-16)

    a,b = linear_sum_assignment(overlap,maximize=True)
    nmi_feats = list(zip(a,b))
    nmi_feats = sorted(nmi_feats,key=lambda x: x[0])
    nmi_feats = list(n[1] for n in nmi_feats)
    coverage0 = (1-np.sum(np.all(d2[:,nmi_feats]==0,axis=1))/d2.shape[0])
    _,nmi = mutual_information(d1,d2[:,nmi_feats])
    best = [(nmi,nmi_feats,coverage0)]
    min_norm = np.min(np.linalg.norm(d2, axis=0))
    ms_domains = set(range(d2.shape[1]))
    print('optimizing normalized mutual information...')
    for i in trange(its):

        n = np.random.choice([0,1,2,3])
        if i%1000 < 500:
            if i%1000 == 0:
                best.append(best[np.random.choice(list(range(len(best))))])
            nmi_feats= list(np.random.choice(list(ms_domains),size=min([len(ms_domains),d1.shape[1]+n]),replace=False))        
        else:
            m = np.random.choice([0,1,2,3])
            if 3 >= len(best[-1][1]):
                m = np.random.choice(list(range(len(best[-1][1]))))
            nmi_feats = list(np.random.choice(best[-1][1],size=len(best[-1][1])-m,replace=False))    
            nmi_feats.extend(np.random.choice(list(ms_domains-set(nmi_feats)),size=min([len(ms_domains-set(nmi_feats)),m+n]),replace=False))

        a,b = linear_sum_assignment(overlap[:,nmi_feats],maximize=True)
        dummy = list(n[1] for n in sorted(list(zip(a,b)),key=lambda x: x[0]))
        nmi_feats = dummy + list(set(nmi_feats)-set(dummy))

        matches = d2[:,nmi_feats]
        for j in range(1,matches.shape[1]):
            matches[:,j] = matches[:,j] - np.sum(d2[:,nmi_feats[:j]],axis=1,keepdims=True).transpose()
        matches[matches<0] = 0
        remove_duds = np.linalg.norm(matches,axis=0) >= min_norm
        nmi_feats = list(nmi_feats[j] for j in range(len(nmi_feats)) if remove_duds[j]==True)
        matches = matches[:,remove_duds]
        
        coverage = (1-np.sum(np.all(d2[:,nmi_feats]==0,axis=1))/d2.shape[0])
        _,nmi = mutual_information(d1,d2[:,nmi_feats])
        
        if nmi > best[-1][0]:
            best[-1] = (nmi,nmi_feats,coverage)
    b = max(best,key=lambda x: x[0])
    nmi_feats = b[1]

    mi,nmi = mutual_information(d1,d2[:,nmi_feats])
    return mi,nmi,nmi_feats



