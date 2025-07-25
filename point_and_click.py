import phd_ms
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np


## Run this file for phd-ms point and click graphical interface

## Necessary parameters
#Change this line to the file with input data
#Omit .h5ad extension!
DATA = '/home/pbeamer/Documents/graphst/visium_hne_graphst'


## Optional parameters (don't change these unless you know what you're doing.)
#Select embedding
EMBEDDING = 'X_gst'
#Set the scale parameters we want to use, and keys to save
RESOLUTIONS = np.linspace(start=0.15,stop=.95,num=8)
RES_KEYS = ['leiden_'+str(r) for r in RESOLUTIONS]
#comparison order
ORDER = range(len(RES_KEYS))
#filtration index (jaccard/containment)
INDEX = 'containment'


def main(DATA,EMBEDDING,RESOLUTIONS,RES_KEYS):

    phd_ms.tl.preprocess_leiden(DATA,output_file=DATA,emb=EMBEDDING,resolution=RESOLUTIONS,res_keys=RES_KEYS,ground_truth=None)
    adata = sc.read_h5ad(DATA+'.h5ad')
    print('Preprocessing complete. (1/3)')
    cluster_complex,clusterings= phd_ms.tl.cluster_filtration(adata,res_keys=RES_KEYS,index='containment',order=range(len(RES_KEYS)))
    print('Preparing to plot. (2/3)')
    phd_ms.tl.point_click_multiscale(adata.obsm['spatial'],cluster_complex,clusterings)
    print('Done. (3/3)')
main(DATA,EMBEDDING,RESOLUTIONS,RES_KEYS)