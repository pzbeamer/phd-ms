
import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import somde
import anndata as ad
from scanit.tools._scanit_representation import spatial_representation, spatial_graph

def svg(filename,folder='',spatial = '',key=''):
    if folder !='':
        adata = sc.read_h5ad(folder+'/'+filename+'.h5ad')
        param = 20
    elif filename == 'merfish':
        adata = sc.read_h5ad('merfish.h5ad')
        adata = adata[adata.obs.Bregma == -9]
        param = 20
    elif filename == 'slideseq':
        adata = sq.datasets.slideseqv2()
        param = 4000
    else:
        adata = sc.read_h5ad(filename+'.h5ad')
        
    if spatial != '':
        sp = pd.read_csv(spatial+'.csv')
        sp = sp[['x','y']].to_numpy()
        adata.obsm['spatial'] = sp
    
    if key!= '':
        adata = adata[adata.obs['annotation']==key]
    X = adata.X.toarray().transpose()
    X = pd.DataFrame(X)
    corinfo = pd.DataFrame(adata.obsm['spatial'],columns=['x','y'])
    corinfo['total_count'] = X.sum(0)
    spatial = corinfo[['x','y']].values.astype(np.float32)
    print(X)
    som = somde.SomNode(spatial,param)
    ndf,ninfo = som.mtx(X)
    som.view()
    nres = som.norm()
    result, SVnum =som.run()
    print(SVnum)
    print(type(result))
    print(result)
    r = result.iloc[0:SVnum,15]
    file_name1 = filename+key+"_svg.txt"
    r.to_csv(file_name1,sep=' ',header=False,index=True)
    
def svg_list(filename,folder='',delim=','):
    sv = np.loadtxt(filename+'_svg.txt',delimiter=delim)
    sv = sv[:,0]
    sv = np.sort(sv)
    return sv

def scanit(filename,folder='',svg=True,key=''):
    if folder !='':
        adata = sc.read_h5ad(folder+'/'+filename+'.h5ad')
    elif filename == 'slideseq':
        adata = sq.datasets.slideseqv2()
        sc.pl.highly_variable_genes(adata)
        adata_sp = adata[:, adata.var.highly_variable]
    else:
        adata = sc.read_h5ad(filename+'.h5ad')
    if key != '':
        adata = adata[adata.obs['annotation']==key]
    n_sv_genes = 3000
    adata_sp = adata.copy()
    if svg == True:
        sv = svg_list(filename,delim = ' ')
        if len(sv) <n_sv_genes:
            n_sv_genes = len(sv)
            print(n_sv_genes)
        sv = sv[0:n_sv_genes]
        print(sv)
        sc.pp.normalize_total(adata_sp)
        adata_sp = adata_sp[:, sv]
    
    sc.pp.log1p(adata_sp)
    sc.pp.scale(adata_sp)
    spatial_graph(adata_sp, method='alpha shape', alpha_n_layer=1, knn_n_neighbors=5)
    spatial_representation(adata_sp, n_h=10, n_epoch=2000, lr=0.001,\
                           device='cpu', n_consensus=5, projection='mds', python_seed=0, torch_seed=0, numpy_seed=0)
    sc.pp.neighbors(adata_sp, use_rep='X_scanit', n_neighbors=10)
    sc.tl.umap(adata_sp)
    sc.tl.leiden(adata_sp, resolution=0.3,key_added="clusters")
    sc.pl.umap(adata_sp, color="clusters")
    sc.pl.embedding(adata_sp, basis="spatial", color="clusters")
    if folder !='':
        adata_sp.write_h5ad(folder+'/'+filename+key+'_scanit.h5ad')
    else:
        adata_sp.write_h5ad(filename+key+'_scanit.h5ad')

def check(filename,folder=''):
    if folder !='':
        adata = sc.read_h5ad(folder+'/'+filename+'.h5ad')
    else:
        adata = sc.read_h5ad(filename+'.h5ad')
    print(adata.obsm['spatial'])

def create_adata(counts,locations=''):
    spatial = pd.read_csv(locations+'.csv')
    adata = sc.read(counts+'.csv')
    print(adata)
    print(spatial)  
    adata.obsm['spatial'] = spatial[['xcoord','ycoord']].to_numpy()
    print(adata)
    adata.write_h5ad(counts+'.h5ad')
    

#adata = sc.read_h5ad('visium-human-dorsalateral_prefrontal_cortex/adata_151673scanit.h5ad')
#print(np.shape(adata.obsm['X_scanit']))

#create_adata('GSM5713332_Puck_191109_06_MappedDGEForR','GSM5713332_Puck_191109_06_BeadLocationsForR')
#svg('merfish')
#svg('slideseq')
#svg('Zhuang-ABCA-2-log2',spatial='ccf_coordinates')
#svg('adata_151673',folder='visium-human-dorsalateral_prefrontal_cortex')
#check('adata_151673',folder='visium-human-dorsalateral_prefrontal_cortex')
#scanit('slideseq',folder='',svg=False)
#adata = sc.read_h5ad('mouse_developmental/E9.5_E1S1.MOSTA.h5ad')
#print(adata.obs['annotation'])
#print(adata[adata.obs['annotation']=='Brain'].obs['annotation'])
#scanit('E13.5_E1S1.MOSTA','mouse_developmental',svg=False,key='Brain')
#scanit('visium_hne',svg=False)
scanit('adata_151673',folder='visium-human-dorsalateral_prefrontal_cortex')
