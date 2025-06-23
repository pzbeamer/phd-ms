import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import numpy as np




def leiden(adata,res=np.linspace(start=0,stop=1,num=20),show=False):
    sc.pp.neighbors(adata,use_rep='X_gst',n_neighbors=5)
    sc.tl.umap(adata)
    for r in res:
        
        sc.tl.leiden(
        adata,
        key_added="clusters"+str(r),
        resolution=r,
        n_iterations=2,
        directed=False,
        )
        if show:
            sc.pl.umap(adata, color="clusters"+str(r))
            sc.pl.embedding(adata, basis="spatial", color="clusters"+str(r))
        
def main(filename):
    from GraphST import GraphST
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    n_clusters = 7
    dataset = 151673
    adata = sc.read_h5ad(filename+'.h5ad')
    #adata = adata[adata.obs.Bregma == -9]
    adata.var_names_make_unique()

    model = GraphST.GraphST(adata)# train model
    adata = model.train()

    adata.obsm['X_gst'] = adata.obsm['emb']

    adata.write_h5ad(filename+'_graphst.h5ad')
    leiden(adata)

def make_adata(filename,folder):
    adata = sc.read_h5ad(folder+'adata_'+filename+'.h5ad')
    for i in range(0,10):
        embedding = np.load(folder+'gst_npy/'+filename+'.npy')[i,:,:]
        adata.obsm['X_gst'] = embedding
        leiden(adata,show=False)
        adata.write_h5ad(folder+'graphst/'+'adata_'+filename+'_gst_'+str(i)+'.h5ad')

#make_adata('151676','visium-human-dorsalateral_prefrontal_cortex/')
main('mouse_developmental/E11.5_E1S1.MOSTA')