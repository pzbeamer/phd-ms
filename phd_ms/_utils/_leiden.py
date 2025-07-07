import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

def leiden(adata,res=np.linspace(start=0.2,stop=1.2,num=6),show=True,scores=False,embedding='X_gst',res_keys=[],ground_truth='cluster'):
    if not res_keys:
        res_keys = ['leiden_'+str(r) for r in res]
    sc.pp.neighbors(adata,use_rep=embedding,n_neighbors=5)
    
    sc.tl.umap(adata)
    for i in range(len(res)):
        sc.tl.leiden(
        adata,
        key_added=res_keys[i],
        resolution=res[i],
        n_iterations=2,
        directed=False,
        )
        if show:
            sc.pl.umap(adata, color=res_keys[i])
            sc.pl.embedding(adata, basis="spatial", color=res_keys[i])
        if scores:
            print('Resolution, adjusted mutual info, adjusted rand:')
            print(cluster_metrics(adata.obs[ground_truth],adata.obs[res_keys[i]]))
    
    return adata

def cluster_metrics(cluster,r,truth=None):
    clusters = set(int(j) for j in cluster.cat.categories.tolist())
    if not(truth is None):
        l1 = np.zeros(len(truth))
        l2 = np.zeros(len(truth))
        for i in range(0,len(truth.cat.categories.tolist())):
            guy = list(n for n in range(0,len(truth)) if truth.iloc[n]==truth.cat.categories.tolist()[i])
            l1[guy] = i
        for i in range(0,len(cluster.cat.categories.tolist())):
            guy = list(n for n in range(0,len(cluster)) if cluster.iloc[n]==cluster.cat.categories.tolist()[i])
            l2[guy] = i
        return r,len(clusters),adjusted_mutual_info_score(l1,l2),adjusted_rand_score(l1,l2)
    else:
        return r,len(clusters)