import scanpy as sc
import squidpy as sq
from scipy import spatial as sp
from scipy.sparse import csgraph
import numpy as np
import networkx
import gudhi as gd
from networkx.algorithms.components.connected import connected_components
import pandas as pd
from sklearn.neighbors import BallTree, KernelDensity,kneighbors_graph,radius_neighbors_graph
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import OPTICS,SpectralClustering,KMeans
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
import matplotlib.pyplot as plt
#import morse_smale as ms
#import somde

def load_data(filename,folder='',ydata='spatial_pca'):
    
    if folder != '':
        adata = sc.read_h5ad(folder+'/'+filename+'.h5ad')
    elif filename == 'merfish':
        adata = sc.read_h5ad('merfish.h5ad')
        adata = adata[adata.obs.Bregma == -9]
        print(adata)    
    elif filename == 'slideseq':
        adata = sq.datasets.slideseqv2()
    else:
        adata = sc.read_h5ad(filename+'.h5ad')
    
    #adata.write_csvs('pp.csv',skip_data=False)
    

    #if filename == 'visium_hne' or filename == 'slideseq':
    if filename == 'slideseq':
        sc.pl.highly_variable_genes(adata)
        adata = adata[:, adata.var.highly_variable]
    elif filename == 'merfish':
        x= 1
    else:
        if ydata == 'spatial_pca':
            spatial_pcs = pd.read_csv(filename+'.csv',header =0)
            spatial_pcs = spatial_pcs.to_numpy()
            spatial_pcs = spatial_pcs[:,1:]
            
            #print(spatial_pcs)
        elif ydata == 'svg':
            
            sv = svg_list(filename,delim = ' ')
            print(sv)
            #adata.raw = adata
            adata = adata[:, sv]
        elif ydata == 'scanit':
            if folder != '':
                adata_sp = sc.read_h5ad(folder+'/'+filename+'_scanit.h5ad')
            else: 
                adata_sp = sc.read_h5ad(filename+'_scanit.h5ad')
            
            return adata
        


    #sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca_variance_ratio(adata, log=False)

    if ydata == 'spatial_pca':
        adata.obsm['X_spatialpca'] = np.transpose(spatial_pcs)
    else: 
        pca_coords = adata.obsm['X_pca']
        pca_coords = pca_coords[:,0:20]
        #adata = sc.read_h5ad
        adata.obsm['X_pca'] = pca_coords

    return adata

def find_neighbors(adata,radius=15):

    #spatial_tree = BallTree(adata.obsm['spatial'])
    #neighbors = spatial_tree.query_radius(adata.obsm['spatial'],r=radius)
    neighbors=list([] for i in range(0,adata.n_obs))
    nbhrs = adata.obsp['connectivities'].nonzero()
    for n in range(0,len(nbhrs[0])):
        neighbors[nbhrs[0][n]].append(nbhrs[1][n])
   #print(neighbors)
    return neighbors
def leiden(adata,res=np.linspace(start=0.2,stop=1.2,num=6),show=True,scores=False,embedding='X_scanit'):
    sc.pp.neighbors(adata,use_rep=embedding,n_neighbors=5)
    
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
        if scores:
            print(cluster_metrics(adata.obs['cluster'],adata.obs['clusters'+str(r)]))
        #print(adata.obs['clusters'+str(r)].cat.categories.tolist())
    
    
    return adata
    #sc.pl.umap(adata)

def preprocess_leiden(input_file,output_file,emb,resolution=np.linspace(start=0.05,stop=.95,num=10)):
    
    adata = sc.read_h5ad(input_file+'.h5ad')
    leiden(adata,res=resolution,show=False,embedding=emb)
    adata.write_h5ad(output_file+'.h5ad')

def order_clusterings(adata,key='neighbors',neighbors=[],tags=['0.4scanit']):
    size = []

    if key == 'num_clusters':
        for tag in tags:
            
                clusters = set(int(j) for j in adata.obs['clusters'+tag].cat.categories.tolist())
                size.append(len(clusters))
    elif key == 'neighbors':
        for tag in tags:
                clusters = adata.obs['clusters'+tag]
                neighbor_graph = networkx.Graph()
                for i in range(0,len(neighbors)):
                    for neighbor in neighbors[i]:
                        #print(i,neighbor)
                        if clusters.iloc[i] != clusters.iloc[neighbor]:
                            neighbor_graph.add_edge(i,neighbor)
                size.append(neighbor_graph.number_of_edges())
    #print(size)
    order = list(x for _, x in sorted(zip(size,range(0,len(size))), key=lambda pair: pair[0]))
    return order

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
    


def leiden_filtration(adata,neighbors,index='containment',tags=['scanit0.4'],order=[],slope=1):

    num_cells = adata.obsm['spatial'].shape[0]
    if order:
        tags = [tags[i] for i in reversed(order)]
    #print(resolution)
    containment_index = np.ones((num_cells,num_cells))
    resolution_index = np.ones((num_cells,num_cells))
    leiden_complex = gd.SimplexTree()

    #Initialize pointwise complex as graph between points within radius r
    ind = neighbors
    
    num_clusters = []
    clusters = []

    for i in range(0,len(tags)-1):

        fine = adata.obs['clusters'+tags[i]]
        coarse = adata.obs['clusters'+tags[i+1]]
        fine_clusters = [int(j) for j in fine.cat.categories.tolist()]
        coarse_clusters = [int(j) for j in coarse.cat.categories.tolist()]
        
        if i == 0:
            num_clusters.append(len(fine_clusters))
            for j in fine_clusters:
                fine_j = set(n for n in range(0,len(fine)) if int(fine.iloc[n])==j)
                clusters.append(fine_j)
        for k in coarse_clusters:
            coarse_k = set(n for n in range(0,len(coarse)) if int(coarse.iloc[n])==k)
            clusters.append(coarse_k)
        num_clusters.append(len(coarse_clusters))
        
        for j in fine_clusters:
            #if i == 0:
                #leiden_complex.insert([j],0)
                #visualize.add_node(j,weight=0)
            
            fine_j = set(n for n in range(0,len(fine)) if int(fine.iloc[n])==j)

            for k in coarse_clusters:
                coarse_k = set(n for n in range(0,len(coarse)) if int(coarse.iloc[n])==k)
            
                intersection = fine_j.intersection(coarse_k)
                union = fine_j.union(coarse_k)
                
                if intersection:
                    if index == 'containment':
                        cont_filt = 1 - len(intersection)/len(fine_j)
                    elif index == 'jaccard':
                        cont_filt = 1 - len(intersection)/len(union)
                    #print(sum(num_clusters[:-2])+j,sum(num_clusters[0:-1])+k)
                    if cont_filt <= 1:
                        res_filt = 1-float(''.join(c for c in tags[i+1] if (not c.isalpha() and not c == '-')))
                        resolution_index[sum(num_clusters[:-2])+j,sum(num_clusters[0:-1])+k] = res_filt
                    else: 
                        res_filt = np.inf 
                        
                    if slope == 1:
                        filt = slope*cont_filt
                    else:
                        filt = slope*cont_filt+(1-slope)*res_filt
                    
                    leiden_complex.insert([sum(num_clusters[:-2])+j,sum(num_clusters[0:-1])+k],filt)
                    #leiden_complex.assign_filtration([sum(num_clusters[:-2])+j],np.min([leiden_complex.filtration([sum(num_clusters[:-2])+j]),filt]))
                    #leiden_complex.assign_filtration([sum(num_clusters[0:-1])+k],np.min([leiden_complex.filtration([sum(num_clusters[0:-1])+k]),filt]))
                    

                                        
    
    for i in range(0,sum(num_clusters)):
        filt = leiden_complex.get_filtration()
        edges_i = sum(list(1 for j in filt if (i in j[0] and len(j[0])>1)))
        #leiden_complex.assign_filtration([i],leiden_complex.filtration([i])/edges_i)
        leiden_complex.assign_filtration([i],0)
    return leiden_complex,clusters

def filt_to_matrix(simp_tree,threshold=.05):
    matrix = np.ones((simp_tree.num_vertices(),simp_tree.num_vertices()))
    #print(simp_tree.num_vertices())
    max_node = np.max(list(x[1] for x in simp_tree.get_filtration() if len(x[0])==1))
    for simplex in simp_tree.get_filtration():
        
        if len(simplex[0]) == 2:
            
            #if simp_tree.filtration([simplex[0][0]]) < threshold and simp_tree.filtration([simplex[0][1]]) < threshold:
            matrix[simplex[0][0],simplex[0][1]] = simplex[1]
            
            if matrix[simplex[0][0],simplex[0][1]] == 0:
                matrix[simplex[0][0],simplex[0][1]] = matrix[simplex[0][0],simplex[0][1]]+.0001
            
            matrix[simplex[0][1],simplex[0][0]] = matrix[simplex[0][0],simplex[0][1]]
                
    return matrix



def add_nodes_less(graph,point_filtration,filt_threshold,step):
    begin = point_filtration[0][1]
    end = point_filtration[-1][1]
    s_1 = next(i for i in range(0,len(point_filtration)) if point_filtration[i][1]>=filt_threshold)
    if filt_threshold+step>=end:
        s_2=-1
    else:
        s_2 = next(i for i in range(0,len(point_filtration)) if point_filtration[i][1]>filt_threshold+step or point_filtration[i][1]==end)
    slice = point_filtration[s_1:s_2]
    for s in slice:
        if len(s[0]) > 1 and filt_threshold <= begin + step*2:
            graph.add_edge(s[0][0],s[0][1])
        elif len(s[0]) == 1:
            graph.add_node(s[0][0])
            
    return graph

def merge_components(graph,num_features,neighborhood_complex,neighbors):
    m = 0
    max_size = max([len(component) for component in list(networkx.connected_components(graph))])
    while networkx.number_connected_components(graph)>num_features-1 and m<max_size:
        #print(m,len(list(component for component in list(networkx.connected_components(graph)) if len(component)==m)))
        if not [component for component in list(networkx.connected_components(graph)) if len(component)==m]:
            m+=1
        eliminate = [component for component in list(networkx.connected_components(graph)) if len(component)==m]

        for e in eliminate:
            big = [n for component in list(networkx.connected_components(graph)) for n in component if len(component)>=m \
                   and n not in eliminate]
            filts = []
            
            for node in e:
                #Draw edge between eliminated component and a neighboringlarger component
                #Need to verify that the neighboring component is actually bigger
                filts.extend(list((neighborhood_complex.filtration([node,neighbor]),neighbor,node)\
                                  for neighbor in neighbors[node] if neighbor in big and not graph.has_edge(node,neighbor)))
            if filts:
                f,n1,n2 = min(filts,key = lambda x:x[0])
                graph.add_edge(n1,n2) 
    return graph

def cc(spatial,point_complex,neighborhood_complex,neighbors,num_features):
    point_filtration = list(point_complex.get_filtration())
    begin = point_filtration[0][1]
    end = point_filtration[-1][1]
    step = (end-begin)/20
    graph = networkx.Graph()
    nodes_not_added = list(i for i in range(0,len(neighbors)))
    filt_threshold = 0
    s_1 = 0
    while nodes_not_added:
        print('yo')
        graph = add_nodes_less(graph,point_filtration,filt_threshold,step)
        filt_threshold += step
        
        nodes_not_added = list(i for i in range(0,len(neighbors)) if not graph.has_node(i))

    graph = merge_components(graph,num_features,neighborhood_complex,neighbors)
    
    x = spatial[:,0]
    y = spatial[:,1]
    plt.figure(200)
    ax = plt.axes()
    for component in networkx.connected_components(graph):
        plt.scatter([x[i] for i in list(component)],[y[i] for i in list(component)])
        
    plt.show()

    cluster_labels = np.zeros(len(neighbors))
    components = list(networkx.connected_components(graph))
    for n in range(0,len(components)):
        c = [k for k in range(0,len(neighbors)) if k in components[n]]
        cluster_labels[c] = n
    df = pd.DataFrame({"a":cluster_labels},dtype='category')
    consensus_cluster = df['a']
    
    return consensus_cluster

def consensus_cluster(spatial,point_complex,neighborhood_complex,neighbors,threshold,num_features):
    point_filtration = list(point_complex.get_filtration())
    
    x = spatial[:,0]
    y = spatial[:,1]
    
    iter = 0
    graph = networkx.Graph()
    begin = point_filtration[0][1]
    end = point_filtration[-1][1]
    print(begin,end)
    step = (end-begin)/20
    s_1 = next(i for i in range(0,len(point_filtration)) if point_filtration[i][1]>begin+7*step)
    slice = point_filtration[0:s_1]
    
    for s in slice:
        if len(s[0]) > 1:
             if neighborhood_complex.filtration(s[0])<=threshold:
                graph.add_edge(s[0][0],s[0][1])

    for component in networkx.connected_components(graph):
        plt.scatter([x[i] for i in list(component)],[y[i] for i in list(component)])
        
    plt.show()


    nodes_not_added = list(i for i in range(0,len(neighbors)) if not graph.has_node(i))
        
    for node in nodes_not_added:
            
        #chain = chain_neighbors(node,[node],graph,neighbors,neighborhood_complex,spatial)
        #for i in range(0,len(chain)-1):
            #graph.add_edge(chain[i],chain[i+1])
        graph.add_edge(node,nearest(node,graph,spatial))
    
    nodes_not_added = list(i for i in range(0,len(neighbors)) if not graph.has_node(i))
    graph.add_nodes_from(nodes_not_added)
    graph = merge_components(graph,num_features,neighborhood_complex,neighbors)

    for component in networkx.connected_components(graph):
        plt.scatter([x[i] for i in list(component)],[y[i] for i in list(component)])
        
    plt.show()
    
        
    plt.show()
    #print(list(networkx.connected_components(graph)))
    cluster_labels = np.zeros(len(neighbors))
    components = list(networkx.connected_components(graph))
    for n in range(0,len(components)):
        c = [k for k in range(0,len(neighbors)) if k in components[n]]
        cluster_labels[c] = n
    df = pd.DataFrame({"a":cluster_labels},dtype='category')
    consensus_cluster = df['a']
    
    return consensus_cluster


def nearest(node,graph,spatial):
    
    dists = list((sp.distance.pdist(np.array([spatial[node,:],spatial[i,:]])),i) for i in graph.nodes)
    _,n = min(dists,key = lambda x:x[0])
    return n
 
def chain_neighbors(node,chain,graph,neighbors,neighborhood_complex,spatial):
    filts = list((neighborhood_complex.filtration([node,neighbor]),neighbor) for neighbor in neighbors[node] if not neighbor in chain)
    dist = list((sp.distance.pdist(np.array([spatial[node,:],spatial[neighbor,:]])),neighbor)\
                 for neighbor in neighbors[node] if neighbor not in chain)
    
    
    #if not filts:
    if not dist:
        return [node]
    f,n = min(dist,key = lambda x:x[0])
    
    chain.append(n)
    if graph.has_node(n):
        return chain
    else:
        return chain_neighbors(n,chain,graph,neighbors,neighborhood_complex,spatial)
    

def filt_to_tree(filtration,title='tree_visualization'):
    tree = networkx.Graph()
    for simplex in filtration:
        if len(simplex[0])==1:
            tree.add_node(simplex[0][0])
        if len(simplex[0])==2:
            tree.add_edge(simplex[0][0],simplex[0][1])
    networkx.write_gml(tree,'./'+title+'.gml')
    
        
def fix_edges(simp_complex):
    edges = list(f[0] for f in simp_complex.get_filtration() if len(f[0])==2)
    for edge in edges: 
        if simp_complex.filtration(edge) < np.max([simp_complex.filtration([edge[0]]),simp_complex.filtration([edge[1]])]):
            simp_complex.assign_filtration(edge, np.max([simp_complex.filtration([edge[0]]),simp_complex.filtration([edge[1]])]))
    return simp_complex

def persistence_cluster(num_clusters,diagram_0d,cocycles,clusterings,adata):

    spatial = adata.obsm['spatial']
    num_spots = len(spatial[:,0])
    #Order in terms of persistence
    reorder = sorted(zip(diagram_0d,cocycles), key=lambda pair: pair[0][2],reverse=True)
    diagram_0d = list(d for d,_ in reorder)
    cocycles = list(c for _,c in reorder)
    
    #Create overarching clustering of cluster indices at all resolutions
    sorted_cluster_indices = [set(cocycles[0])]
    #Get all elements in clusters corresponding to cluster indices
    sorted_elements = []
    
    for i in range(1,num_clusters):
        sorted_cluster_indices = list(n.difference(set(cocycles[i])) for n in sorted_cluster_indices)
        sorted_cluster_indices.append(set(cocycles[i]))
    i = num_clusters
    while len(sorted_elements) < num_clusters:
        
        if i >= len(cocycles):
            print("Cluster number too large.")
            break

        sorted_cluster_indices = list(n.difference(set(cocycles[i])) for n in sorted_cluster_indices)
        sorted_cluster_indices.append(set(cocycles[i]))
        sorted_elements = []
        for j in range(len(sorted_cluster_indices)):
            #if len(sorted_cluster_indices[j])<2:
                #continue
            #else:
                #print('no')
            new_feat = set.union(*[clusterings[k] for k in sorted_cluster_indices[j]])
            for k in sorted_elements:
                new_feat = new_feat - k
            if len(new_feat)>num_spots/50:
                sorted_elements.append(new_feat)
        i+=1
        #print(len(sorted_elements))
    
    unsorted = set.union(*clusterings)-set.union(*sorted_elements)
    sorted_elements = sort_nearest(unsorted,sorted_elements,spatial)
    
    sorted_elements = kde_smoothing(sorted_elements,spatial,num_clusters)
    plt.figure(10)
    x = spatial[:,0]
    y = spatial[:,1]
    
    for l in sorted_elements:
        plt.scatter([x[i] for i in l],[y[i] for i in l])
        #plt.show()

    #plt.show()
    cluster_labels = np.zeros(np.shape(x))
    
    for n in range(0,len(sorted_elements)):
        c = [k for k in sorted_elements[n]]
        cluster_labels[c] = n
    df = pd.DataFrame({"a":cluster_labels},dtype='category')
    persistence_cluster = df['a']
    
    return persistence_cluster

def sort_nearest(to_sort,sorted_elements,spatial):
    sorted_elements_new = sorted_elements.copy()
    for element in to_sort:
        dists = []
        for i in range(0,len(sorted_elements)):
            dists.extend(list((sp.distance.pdist(np.array([spatial[element,:],spatial[c,:]])),i) for c in sorted_elements[i]))
        _,n = min(dists,key = lambda x:x[0])
        sorted_elements_new[n].add(element)

    return sorted_elements_new

def kde_smoothing(pc,spatial,num_clusters):
    pc_new = list()
    for cluster in pc:
        c = list(cluster)
        coords = spatial[c,:]
        
        adj = kneighbors_graph(coords, 5,mode='distance')
        adj = adj.toarray()
        radius = np.mean(np.amax(adj,0))
        adj = radius_neighbors_graph(coords,radius)
        _, labels = csgraph.connected_components(adj)

        new = list(set() for i in range(max(labels)+1))
        for i in range(len(labels)):
            new[labels[i]].add(c[i])
        small_fry = set()
        new1 = list()
        for n in new:
            if len(n) >= .1*len(c):
                new1.append(n)
            else:
                small_fry.update(n)

        new = sort_nearest(small_fry,new1,spatial)
        pc_new.extend(new) 
        for n in new:
            plt.scatter(spatial[list(n),0],spatial[list(n),1])
        #plt.show()
        """ 
        params = {"bandwidth": np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(coords)
        kde = grid.best_estimator_
        plt.figure(1)
        scores = kde.score_samples(coords)
        low_dens = set()
        for i in range(0,coords.shape[0]):
            if scores[i] <= np.percentile(scores,20):
                #print(coords[i,:])
                low_dens.add(np.where(np.all((spatial == coords[i,:]),axis=1))[0][0])
       
        high_dens = cluster - low_dens
        high_dens = list(high_dens)
        low_dens = list(low_dens) """
        
        """ db = KMeans(n_clusters=n_components).fit(coords)

        if len(set(db.labels_))==1:
            pc_new.append(cluster)
        else:
            new = list()
            for label in set(db.labels_):
                points = set(np.where(db.labels_ == label)[0])
                points = set(c[i] for i in points)
                new.append(points)
            
            small_fry = set()
            for n in new:
                if len(n) < .3*len(cluster):
                    new.remove(n)
                    small_fry.update(n)
            new = sort_nearest(small_fry,new,spatial)
            
            
            pc_new.extend(new) 
        
            x = np.linspace(np.min(spatial[:,0]),np.max(spatial[:,0]),num=1000)
            y = np.linspace(np.min(spatial[:,1]),np.max(spatial[:,1]),num=1000)
            X, Y = np.meshgrid(x,y)
            xy = np.vstack([X.ravel(),Y.ravel()]).T
            Z = np.exp(kde.score_samples(xy))
        
            Z = Z.reshape(X.shape)
            #levels = np.linspace(Z.min(), Z.max(), 25)
            plt.contourf(X, Y, Z,levels=25,cmap=plt.cm.Reds) """
        
    to_sort = set()
    while len(pc_new) > num_clusters+3:
        size_sorted = sorted(zip(pc_new,list(len(p) for p in pc_new)),key= lambda pair: pair[1])
        pc_new.remove(size_sorted[0][0])
        to_sort.update(size_sorted[0][0])
    pc_new = sort_nearest(to_sort,pc_new,spatial)
    return pc_new




def is_hierarchical(adata,resolution_id):
    
    for i in range(0,len(resolution_id)-1):
        print(adata.obs['clusters'+str(resolution_id[i])])
        fine = adata.obs['clusters'+str(resolution_id[i])]
        coarse = adata.obs['clusters'+str(resolution_id[i+1])]
        num_cluster = np.max(fine)
        for j in range(0,num_cluster):
            fine_indices = [k for k in range(0,len(fine)) if k == j]
            coarse_indices = coarse[fine_indices]
            if np.max(coarse_indices)-np.min(coarse_indices) != 0:
                return False
    return True


def svg_list(filename,folder='',delim=','):
    sv = np.loadtxt(filename+'_svg.txt',delimiter=delim)
    sv = sv[:,0]
    sv = np.sort(sv)
    return sv

def scalar_field(adata,kernel='exponential_kernel',radius=15,embedding='X_scanit'):
    
    spatial_tree = BallTree(adata.obsm['spatial'])
    ind, dspatial = spatial_tree.query_radius(adata.obsm['spatial'],r=5*radius,return_distance=True)
    
    num_cells = adata.obsm['spatial'].shape[0]
    sf = np.zeros((num_cells,1))
    
    for i in range(0,num_cells):
        for j in range(0,len(ind[i])):
            d = dspatial[i][j]

            if kernel == 'radius_kernel':
                k = radius_kernel(d,radius)
            elif kernel == 'exponential_kernel':
                k = exponential_kernel(d,eta=radius)
            elif kernel == 'lorentz_kernel':
                k = lorentz_kernel(d,eta=radius)
            sf[i] += k*sp.distance.euclidean(adata.obsm[embedding][i,:],adata.obsm[embedding][ind[i][j],:])
        sf[i] /= len(ind[i])
            
    #sf = np.divide(sf[:,0],sf[:,1])
    sf = np.nan_to_num(sf)
    adata.obs['sf']= sf
    
    return adata

def exponential_kernel(dspatial,eta=15,k=1):
    return np.exp(-(dspatial/eta)**k)

def lorentz_kernel(dspatial,eta=15,nu=1):
    return 1/(1+(dspatial/eta)**nu)

def radius_kernel(dspatial,r):
    if 0<dspatial<=r:
        return 1
    else:
        return 0

def diff_express_tumor(dataset,domains,compare):
    brca_express =pd.read_csv('expression_diff_BRCA.txt', sep="\t", header=0)
    print(list(brca_express['Gene']))
    adata = sc.read_h5ad(dataset)
    
    #plot the diff expressed genes
    adata.obs['tumor_cat'] = construct_clustering(adata,domains)
    #print(adata.obs['tumor_cat'])
    sc.set_figure_params(scanpy=True, fontsize=30)
    sc.tl.rank_genes_groups(adata, 'tumor_cat', groups=[compare[1]], reference=compare[0],method='t-test',use_raw=False,key_added='compare1')
    sc.pl.rank_genes_groups(adata, groups=[compare[1]], n_genes=5,key='compare1')
    ax=sc.pl.rank_genes_groups_violin(adata,groups=compare[1],n_genes=5,save=True,key='compare1')
    
        
    sig = sc.get.rank_genes_groups_df(adata,group=compare[1],key='compare1')
    genes =[]
    for index,row in sig.iterrows():
        if row['logfoldchanges'] > 1.5 and row['pvals_adj'] < 0.001:
            if row['names'] in list(brca_express['Gene']):
                genes.append(row['names'])
                print(row['names'])
    print(len(genes))
    input()
    
    
    sc.tl.rank_genes_groups(adata, 'tumor_cat', groups=[compare[0]], reference=compare[1],method='t-test',use_raw=False,key_added='compare2')
    sc.pl.rank_genes_groups(adata, groups=[compare[0]], n_genes=5,key='compare2')
    sc.pl.rank_genes_groups_violin(adata,groups=compare[0],n_genes=5,save=True,key='compare2')
    
    
    sig = sc.get.rank_genes_groups_df(adata,group=compare[0],key='compare2')
    genes =[]
    for index,row in sig.iterrows():
        if row['logfoldchanges'] > 1.5 and row['pvals_adj'] < 0.001:
            if row['names'] in list(brca_express['Gene']):
                genes.append(row['names'])
                print(row['names'])
    print(len(genes))
    input()


    #print(len(adata.obs['tumor_cat'].cat.categories.tolist()))
    sc.tl.rank_genes_groups(adata, 'tumor_cat', method='t-test', key_added = "t-test",use_raw=False)
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key = "t-test")
    plt.show()
    

    
    sc.pl.umap(adata, color='tumor_cat',title='Domain UMAP')
    plt.show()
    
    #adata.write_h5ad(dataset)
    for cat in adata.obs['tumor_cat'].cat.categories.tolist():
        
        sig = sc.get.rank_genes_groups_df(adata,group=str(cat),key='t-test')
        genes =[]
        for index,row in sig.iterrows():
            if row['logfoldchanges'] > 1.5 and row['pvals_adj'] < 0.001:
                genes.append(row['names'])
                print(row['names'])
        print(len(genes))
        input()
    
    
    """  sig = list(adata.uns['t-test']['names'][['9.0']][x][0] for x in range(len(adata.uns['t-test']['names']['9.0'])))
    sig = list(sig[i] for i in np.argsort(adata.uns['t-test']['pvals_adj']['9.0']))
    sig = list(sig[i] for i in range(len(sig)) if adata.uns['t-test']['pvals_adj']['9.0'][i]< 0.0001)
    sig = list(sig[i] for i in range(len(sig)) if adata.uns['t-test']['logfoldchanges']['9.0'][i]>2)
    
    
    
    print(len(sig))
    sig = '\n'.join(sig)
    print(sig) """
    
    
        

def construct_clustering(adata,domains):
    category = np.zeros(adata.shape[0])
    spatial = adata.obsm['spatial']
    #Identify cell spots by the domain they most belong to.
    for n in range(len(category)):
        #Find the max scoring domain
        arg_max = int(np.argmax(list(adata.uns['multiscale'][domain][n] for domain in domains)))
        #Exclude cells which don't belong to any domain
        max_score = np.max(list(adata.uns['multiscale'][domain][n] for domain in domains))
        if max_score > 0.05:
            category[n] = str(arg_max+1)
        else:
            category[n] = str(len(domains)+2)
    
    # We want to get rid of unassigned spots
    unassigned = list(n for n in range(len(category)) if category[n] == str(len(domains)+2))
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
        plt.scatter(dff['x'], -dff['y'], c=cmap(norm(dff['colors'])), 
                edgecolors='none', label="Feature {:g}".format(i))

    plt.legend()
    plt.show()
    category = list(str(int(n)) for n in category)
    #print(category)
    return pd.Categorical(category,categories=list(str(i) for i in range(1,len(domains)+1)))

