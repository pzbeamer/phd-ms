import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

def leiden(adata,res=np.linspace(start=0.2,stop=1.2,num=6),show=True,scores=False,embedding='X_gst',res_keys=[],ground_truth='cluster',neighbors=15):
    if not res_keys:
        res_keys = [f'leiden_{r:.3f}' for r in res]
    sc.pp.neighbors(adata,use_rep=embedding,n_neighbors=neighbors)
    
    for i in range(len(res)):
        sc.tl.leiden(
        adata,
        key_added=res_keys[i],
        resolution=res[i],
        n_iterations=5,
        directed=False,
        )
        if scores:
            print('Resolution, adjusted mutual info, adjusted rand:')
            print(cluster_metrics(adata.obs[res_keys[i]],truth=adata.obs[ground_truth]))
    
    return adata

def cluster_metrics(cluster,truth=None):
    clusters = set(j for j in cluster.cat.categories.tolist())
    if not(truth is None):
        l1 = np.zeros(len(truth))
        l2 = np.zeros(len(truth))
        for i in range(0,len(truth.cat.categories.tolist())):
            guy = list(n for n in range(0,len(truth)) if truth.iloc[n]==truth.cat.categories.tolist()[i])
            l1[guy] = i
        for i in range(0,len(cluster.cat.categories.tolist())):
            guy = list(n for n in range(0,len(cluster)) if cluster.iloc[n]==cluster.cat.categories.tolist()[i])
            l2[guy] = i
        return len(clusters),adjusted_mutual_info_score(l1,l2),adjusted_rand_score(l1,l2)
    else:
        return len(clusters)