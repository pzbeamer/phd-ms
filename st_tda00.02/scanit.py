import scanpy as sc
import numpy as np

def leiden(adata,res=np.linspace(start=0,stop=1,num=20),show=False):
    sc.pp.neighbors(adata,use_rep='X_scanit',n_neighbors=5)
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

def make_adata(filename,folder):
    adata = sc.read_h5ad(folder+'adata_'+filename+'.h5ad')
    
    embedding = np.load(folder+'scanit_npy/embedding_'+filename+'.npy')
    adata.obsm['X_scanit'] = embedding
    leiden(adata,show=False)
    adata.write_h5ad(folder+'scanit/'+'adata_'+filename+'_scanit.h5ad')
make_adata('151672','visium-human-dorsalateral_prefrontal_cortex/')