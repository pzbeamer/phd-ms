

import sys
sys.path.append('..')
import phd_ms
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tracemalloc
import time
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

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
        return len(clusters),normalized_mutual_info_score(l1,l2),adjusted_rand_score(l1,l2)
    else:
        return len(clusters)
    
IN_DIR = '/home/pbeamer/Documents/h5ad/'
TECH = [('graphst/','adata_','_gst_0'),('scanit/','','_scanit'),('stagate/','adata_','_stagate'),('banksy/','adata_','_banksy')]
EMB = ['X_gst','X_scanit','STAGATE','banksy']
RESOLUTIONS = np.linspace(start=0.15,stop=.95,num=8)
RES_KEYS = ['leiden_'+str(r) for r in RESOLUTIONS]
GROUND_TRUTH = 'cluster'
DATASETS = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
DATASETS = ['151673']
for DATASET in DATASETS:
    print(f'{DATASET} starting')
    for i in range(len(TECH)):
        print(f'{TECH[i][2]} starting')
        INPUT_FILE= f'{IN_DIR}{TECH[i][0]}{TECH[i][1]}{DATASET}{TECH[i][2]}'
        emb = EMB[i]

        #Set the scale parameters we want to use, and keys to save
        if i != 3:
            phd_ms.tl.preprocess_leiden(INPUT_FILE,output_file=INPUT_FILE,emb=emb,resolution=RESOLUTIONS,res_keys=RES_KEYS,ground_truth=GROUND_TRUTH,neighbors=5)
        adata = sc.read_h5ad(INPUT_FILE+'.h5ad')
        ami = []
        for key in RES_KEYS:
            ami.append(cluster_metrics(adata.obs[key],truth=adata.obs[GROUND_TRUTH])[1])
        print(f'ami scores:{ami}')
        print(f'max ami: {np.max(ami)}')

        
        """ tracemalloc.start()
        start=time.perf_counter()
        cluster_complex,clusterings= phd_ms.tl.cluster_filtration(adata,res_keys=RES_KEYS,index='containment',order=range(len(RES_KEYS)),)
        adata.obsm['multiscale']= phd_ms.tl.map_multiscale(adata.obsm['spatial'],cluster_complex,clusterings,num_domains=0,filt=0,order='persistence',redundant_filter=False,plots='off')
        end=time.perf_counter()
        adata.uns['phdms_compute'] = end-start
        adata.uns['phdms_memory'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        adata.uns['metrics'] = {}
        adata.uns['metrics']['multiscale'] = phd_ms.tl.ground_truth_benchmark(adata.obs[GROUND_TRUTH],adata.obsm['multiscale'],adata.obsm['spatial'],plots=False,conversion_factor=16.435,metrics=['wasserstein','nmi'])

        r = np.argmin(list(abs(len(adata.obs[GROUND_TRUTH].cat.categories.to_list()) - len(adata.obs[key].cat.categories.to_list())) for key in RES_KEYS))
        r = RES_KEYS[r]
        #Convert to distribution
        adata.uns['metrics']['ground truth'] = phd_ms.tl.ground_truth_benchmark(adata.obs[GROUND_TRUTH],adata.obs[r],adata.obsm['spatial'],plots=False,conversion_factor=16.435,metrics=['wasserstein','nmi'])
        adata.write_h5ad(f'{IN_DIR}{TECH[i][0]}/multiscale/{DATASET}{TECH[i][2]}_phdms.h5ad')
        print(f'{TECH[i][2]} done') """
    print(f'{DATASET} done')


