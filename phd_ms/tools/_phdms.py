

import numpy as np
import scanpy as sc
import gudhi as gd
from .._utils import leiden, filt_to_matrix, union_find_dmat, get_sub_features, clusters_to_distribution, nmi_metric, wasserstein_metric
import matplotlib.pyplot as plt
from scipy.special import logit
import ot
import tracemalloc
import time
import pandas as pd


def preprocess_leiden(input_file,output_file='',emb='X_gst',
                      resolution=np.linspace(start=0.05,stop=.95,num=10),
                      res_keys=[],ground_truth='cluster',neighbors=15):
    
    adata_k = sc.read_h5ad(input_file+'.h5ad')
    adata = adata_k.copy()
    adata.obs = pd.DataFrame(index=adata.obs.index)
    adata.obs[ground_truth] = adata_k.obs[ground_truth]
    for key in list(adata.uns.keys()):
        if 'leiden_' in key or 'clusters' in key:
            del adata.uns[key]
    tracemalloc.start()
    start=time.perf_counter()
    leiden(adata,res=resolution,show=False,embedding=emb,res_keys=res_keys,ground_truth=ground_truth,scores=False,neighbors=neighbors)
    end=time.perf_counter()
    adata.uns['leiden_compute'] = end-start
    adata.uns['leiden_memory'] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    adata.write_h5ad(output_file+'.h5ad')

def cluster_filtration(adata,res_keys,index='containment',order=[]):

    num_cells = adata.obsm['spatial'].shape[0]

    #Reorder resolutions if order is given
    if order:
        res_keys = [res_keys[i] for i in reversed(order)]

    #Create the filtration
    leiden_complex = gd.SimplexTree()
    
    #List of number of clusters at each resolution
    num_clusters = []
    #List of sets of indices for each cluster at each resolution
    clusters = []

    #We want to iterate through pairs of neighboring resolutions
    for i in range(0,len(res_keys)-1):
        
        #Initialize the pair of resolutions we look at
        fine = adata.obs[res_keys[i]]
        coarse = adata.obs[res_keys[i+1]]
        fine_clusters = [int(j) for j in fine.cat.categories.tolist()]
        coarse_clusters = [int(j) for j in coarse.cat.categories.tolist()]
        
        #Update number of clusters and clusters
        #We want to add this information only on the first iteration
        if i == 0:
            num_clusters.append(len(fine_clusters))
            for j in fine_clusters:
                fine_j = set(n for n in range(0,len(fine)) if int(fine.iloc[n])==j)
                clusters.append(fine_j)
        
        for k in coarse_clusters:
            coarse_k = set(n for n in range(0,len(coarse)) if int(coarse.iloc[n])==k)
            clusters.append(coarse_k)
        num_clusters.append(len(coarse_clusters))
        
        #Now, we compare every cluster in the fine resolution with every cluster in the coarse resolution
        for j in fine_clusters:
            
            #The j-th cluster in the fine resolution
            fine_j = set(n for n in range(0,len(fine)) if int(fine.iloc[n])==j)

            for k in coarse_clusters:
                #The k-th cluster in the coarse resolution
                coarse_k = set(n for n in range(0,len(coarse)) if int(coarse.iloc[n])==k)

                #Compute union and intersection of the fine and coarse clusters
                intersection = fine_j.intersection(coarse_k)
                union = fine_j.union(coarse_k)
                
                #If they intersect, we compute a filtration value
                if intersection:
                    #Compute filtration value as either containment or Jaccard index
                    if index == 'containment':
                        filt = 1 - len(intersection)/len(fine_j)
                    elif index == 'jaccard':
                        filt = 1 - len(intersection)/len(union)
                    
                    #Add edge between clusters with this filtration value
                    leiden_complex.insert([sum(num_clusters[:-2])+j,sum(num_clusters[0:-1])+k],filt)
                                        
    #Make sure that all the clusters have filtration value 0
    for i in range(0,sum(num_clusters)):
        leiden_complex.assign_filtration([i],0)
    return leiden_complex,clusters

def map_multiscale(spatial,cluster_complex,clusterings,num_domains=0,filt=0,plots="on",order='persistence',redundant_filter=False,save=False):

    dmat = filt_to_matrix(cluster_complex)
    diagram_0d,cocycles,_= union_find_dmat(dmat,edge_cut=1)
    diagram_0d[0][2] = 1
    if num_domains == 0:
        num_domains = len(cocycles)
    #Sort persistent homology results by death time
    diagram_0d,cocycles = zip(*sorted(zip(diagram_0d,cocycles),key=lambda x: x[0][2],reverse=True))
    #Filter out non-persistent results if desired.
    if filt > 0:
        index = next(i for i in range(len(diagram_0d)) if diagram_0d[i][2]<filt)
        diagram_0d = diagram_0d[:index]
        cocycles = list(cocycles[:index])
    domains = []

    #Iterate through persistent components
    for n in range(len(cocycles)):

        feature_list = [(set(cocycles[n]),diagram_0d[n][2])]
        #find all the clusters that belong to the multiscale domain
        feature_list = get_sub_features(cocycles,diagram_0d,feature_list[0][0],feature_list)
        
                
        x = spatial[:,0]
        y = spatial[:,1]
        coreness = 1.001*np.ones(len(x))
        #Compute coreness score for each point in the tissue
        #iterate backwards through features to find filtration value where point first appears in multiscale domain
        feature_list = sorted(feature_list,key= lambda x: x[1],reverse=True)
        tracker = set()
        for i in range(len(feature_list)-1, -1, -1):
            
            spots = set()
            for clust in feature_list[i][0]:
                spots = set.union(spots,clusterings[clust])
            
            coreness[list(spots-tracker)] = feature_list[i][1]
            tracker = set.union(tracker,spots)

        nontrivial = True
        #Filter out results which share too similar proportions to previously examined domains
        if redundant_filter:
            for guy in domains:
                zz = set(i for i in range(len(coreness)) if guy[i] < 1)
                gg = set(i for i in range(len(coreness)) if coreness[i] < 1)
                if len(gg.intersection(zz)) > redundant_filter*len(gg.union(zz)):
                    nontrivial = False
                    break
        if nontrivial:
            domains.append(coreness) 
    
    #Default order is by death time
    #Ordered by size of domain if specified
    

    #Normalize the coreness values to be between 0 and 1
    for i in range(len(domains)):
        max = np.max(list(z for z in domains[i]))
        min = np.min(list(z for z in domains[i]))
        z  = list(1-(domains[i][j]-min)/(max-min) for j in range(len(domains[i])))
        domains[i] = np.array(z)
        
    if order == 'size':   
        domains = sorted(domains,key = lambda x : np.linalg.norm(np.array(x))) 
    elif order == 'persistence':
        domains = domains
    elif order == 'size-persistence':
        s = list(np.linalg.norm(np.array(domains[i]))*(diagram_0d[i][2]) for i in range(len(domains)))
        domains = [x for _,x in sorted(zip(s, domains),key=lambda pair: pair[0])]
    if plots == 'on':
        for i in range(len(domains[:num_domains])):
            plot_multiscale(domains[i],spatial)
        mm = np.array(domains[:num_domains]).transpose()
        dd = np.array(list(d[2] for d in diagram_0d[:num_domains]))
        mm =(mm @ dd)-1
        plt.figure(figsize=(8, 20/3))
        plt.scatter(spatial[:,0],spatial[:,1],c=mm,cmap='coolwarm',s=30,linewidths=.5)
        cbar = plt.colorbar()
        plt.show()


    return np.array(domains[:num_domains]).transpose()


#Currently broken
def point_click_multiscale(spatial,cluster_complex,clusterings,filt=0,order='persistence',redundant_filter=True):
    from mpl_point_clicker import clicker

    domains = map_multiscale(spatial,cluster_complex,clusterings,num_domains=0,filt=filt,plots='off',order=order,redundant_filter=redundant_filter)
    tracker = []
    exit = False
    while not exit:
        
        plt.figure(figsize=[15,15])
        plt.rcParams.update({'font.size': 25})
        ax = plt.gca()
        ax.scatter(spatial[:,0],spatial[:,1],c='k')
        ax.set_title('Click once to specify spot. Close to visualize domains')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        klicker = clicker(ax, ["spot"], markers=["*"],markersize=20,colors=['red'])


        plt.tight_layout()
        plt.show()
        xy = klicker.get_positions()['spot']
        
        input_ind = np.argmin((spatial[:,0]-xy[-1][0])**2+(spatial[:,1]-xy[-1][1])**2)
        #print(input_ind)
        input_cluster = list(i for i in range(len(clusterings)) if input_ind in clusterings[i])
        #print(input_cluster)
        x = spatial[:,0]
        y = spatial[:,1]
        
        for d in domains.transpose():
            if d[input_ind] > .95:
                tracker.append(d)
                plot_multiscale(d,spatial,marker=xy[-1])
            plt.show()

        exit = input('Press E to exit.')
        if exit == 'E':
            return tracker
        

def ground_truth_benchmark(ground_truth,multiscale,spatial,plots=False,conversion_factor=1,metrics=['wasserstein','nmi'],single_scale = False):

    ground_truth_distributions,gmat = clusters_to_distribution(ground_truth)
    multiscale_distributions,mmat = clusters_to_distribution(multiscale)
    output = {}
    for metric in metrics:
        if metric == 'wasserstein':
            dmat = ot.dist(spatial*conversion_factor,spatial*conversion_factor,metric='euclidean')
            costs,matches = wasserstein_metric(ground_truth_distributions,multiscale_distributions,dmat)
            if plots:
                for j in range(len(costs)):
                    plot_multiscale(gmat[:,j],spatial,title='Ground truth domain '+str(j))
                    plot_multiscale(mmat[:,matches[j]],spatial,title='Best match '+str(j))
                plt.show()
            output['wasserstein costs'] = costs
            output['wasserstein matches'] = matches

        elif metric == 'nmi':
            mi,nmi,nmi_feats = nmi_metric(gmat,mmat,single_scale,10000)
            print(f'NMI: {nmi}, MI: {mi}')
            if plots:
                for j in range(len(nmi_feats)):
                    plot_multiscale(mmat[:,nmi_feats[j]],spatial,title='Best match'+str(j))
                plt.show()
            output['nmi'] = nmi
            output['nmi matches'] = nmi_feats
            output['mi'] = mi
    return output

def construct_clustering(adata,domains):
    category = np.zeros(shape=(adata.shape[0],1))
    spatial = adata.obsm['spatial']
    #Identify cell spots by the domain they most belong to.
    for n in range(len(category)):
        #Find the max scoring domain
        arg_max = int(np.argmax(list(adata.obsm['multiscale'][n,domain] for domain in domains)))
        #Exclude cells which don't belong to any domain
        max_score = np.max(list(adata.obsm['multiscale'][n,domain] for domain in domains))
        if max_score > 0.05:
            category[n] = arg_max+1
        else:
            category[n] = len(domains)+2
    print(category)
    # We want to get rid of unassigned spots
    unassigned = list(n for n in range(len(category)) if category[n] == len(domains)+2)
    new_spatial = spatial.copy()
    new_spatial[unassigned,0] = 10**10
    new_spatial[unassigned,1] = 10**10
    for n in unassigned:
        nearest_spot = np.argmin((new_spatial[:,0]-spatial[n,0])**2+(new_spatial[:,1]-spatial[n,1])**2)
        category[n] = category[nearest_spot]

    
    plt.figure()
    df = pd.DataFrame({"x":np.array(adata.obsm['spatial'][:,0]).flatten(), 
                   "y":np.array(adata.obsm['spatial'][:,1]).flatten(), 
                   "colors":np.array(category).flatten()})
    cmap = plt.cm.Set1
    norm = plt.Normalize(df['colors'].values.min(), df['colors'].values.max())
    for i, dff in df.groupby("colors"):
        plt.scatter(dff['x'], dff['y'], c=cmap(norm(dff['colors'])), 
                edgecolors='none', label="Feature {:g}".format(i))

    plt.legend()
    plt.show()
    category = list(str(int(n)) for n in category)
    #print(category)
    return pd.Categorical(category,categories=list(str(i) for i in range(1,len(domains)+1)))

def plot_multiscale(multiscale,spatial,title='',marker=np.array([False]),save=''):
    x = spatial[:,0]
    y = spatial[:,1]
    z = multiscale.copy()
    #Correction to approximate logit, we can't take logit of 0 or 1.
    z = z*.99996+.00002
    
    plt.figure(figsize=(3, 3))
    plt.title(title)
    plt.scatter(x,y,c=logit(z),cmap='coolwarm',s=7,linewidths=.1)
    if np.any(marker):
        plt.scatter(marker[0],marker[1],c='r',s=40,edgecolors='k',linewidths=.1,marker='*')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('logit(coreness)')
    frame1 = plt.gca()
    frame1.axis('off')
    if save != '':
            plt.savefig(f'{save}.png',bbox_inches='tight')

def plot_singlescale(adata,res_keys,spatial,title='',marker=np.array([False]),save=False):
    import plotly
    import itertools
    for res in res_keys:
        singlescale = adata.obs[res]
        plt.figure(figsize=(3, 3))
        if title=='res':
            plt.title(f'k={res.replace('leiden_','')}')
        elif title:
            plt.title(title)
        color_list = itertools.cycle(plotly.colors.qualitative.Plotly)
        for cluster in singlescale.cat.categories.tolist():
            cluster_spots = list(k for k in range(len(singlescale)) if cluster == singlescale.iloc[k])
            x = spatial[cluster_spots,0]
            y = spatial[cluster_spots,1]
            plt.scatter(x,y,s=7,c=next(color_list),linewidths=.5)
        if np.any(marker):
            plt.scatter(marker[0],marker[1],c='r',s=7,edgecolors='k',linewidths=.1,marker='*')
        frame1 = plt.gca()
        frame1.axis('off')
        if save:
            plt.savefig(f'{save}{res.replace('leiden_','')}.png',bbox_inches='tight')
