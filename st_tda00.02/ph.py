#import ripser
import gudhi as gd
import numpy as np
from scipy import spatial as sp
from scipy.special import logit
import networkx
import matplotlib.pyplot as plt
import sys
from operator import itemgetter
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import argrelextrema
import warnings
import ot

def ph_reeb(adata,dim=0,software='gudhi',epsilon=0):
    sys.settrace
    if software == 'ripser':
        a = 1
        #v = ripser(input,maxdim = dim)['dgms']
    elif software == 'gudhi':
        #input.compute_persistence()
        #r = input.persistence_pairs()
        adata.uns['reeb_persistence'] = adata.uns['reeb'].persistence()
        #adata.uns['grid_persistence'] = adata.uns['grid_complex'].persistence()
        adata.uns['simplified_reeb'] = simplify(adata.uns['reeb'],epsilon)
        adata.uns['simplified_reeb_persistence'] = adata.uns['simplified_reeb'].persistence()
        
        
    elif software == 'union_find':
        v = adata.uns['reeb'].persistence()
        #a,b,c = union_find(adata.uns['reeb_filtration'])
        #reeb_persistence = new(adata.uns['reeb_filtration'])
        #grid_persistence = new(adata.uns['grid_filtration'])
        #dict([('diagram_0d',a),('Cocycles',b),('Cocycle_fvalues',c)])
        #a,b,c = union_find(adata.uns['grid_filtration'])
        #grid_persistence = dict([('diagram_0d',a),('Cocycles',b),('Cocycle_fvalues',c)])
    else:
        raise ValueError('Invalid input')
    return adata #grid_persistence#,v#grid_persistence,v

def significant_features(persistence,cm,show=False):
    #print(cm)
    #Reshape persistence results
    persistence = list(persistence[i][1][1] for i in range(0,len(persistence)) if persistence[i][1][0]<.05)
    num_inf = sum(list(1 for p in persistence if p > 1000))
    persistence = list(p for p in persistence if p <= 1000)
    persistence = np.array(persistence).reshape(-1,1)
    #print(np.max(persistence))
    cutoff = (np.shape(persistence)[0]/persistence[0])*.75
    cutoff2 = (np.shape(persistence)[0]/persistence[0])*.5
    #persistence = (persistence*100)
    persistence = (np.shape(persistence)[0]/persistence[0])*persistence

    #Estimate kernel density estimator
    param_grid = {
    'bandwidth': np.linspace(0.05, 5, 30),
    'kernel': ['gaussian', 'tophat', 'exponential']
    }
    warnings.filterwarnings('ignore') 
    grid = GridSearchCV(KernelDensity(), param_grid)
    grid.fit(persistence)
    kde = grid.best_estimator_
    
    #Compute kde scores across range
    s=np.linspace(0,np.max(persistence)+.5,num=50000)
    
    mini = np.min(kde.score_samples(s.reshape(-1,1)))
    e = kde.score_samples(s.reshape(-1,1))-mini

    #extract persistence results between peaks
    y = np.zeros(np.shape(persistence))

    sig_peaks,sig_min = estimate_peaks(s,e)
    
    #Extract features between minima
    num_feats = []
    for i in range(len(sig_min)):
        if i == len(sig_min)-1:
            features = list(p for p in persistence if p>=s[sig_min[i]])
        else:
            features = list(p for p in persistence if p<s[sig_min[i+1]] and p>=s[sig_min[i]])
        num_feats.append(len(features))
    #print(num_feats)
    #Order peaks in terms of closeness to 75% of persistence range
    order = list(range(len(sig_peaks)))
    
    num_feats = list(sum(num_feats[i:])+num_inf for i in range(len(sig_peaks)))
    #print(num_feats)
    #Usually 1, only len(num_feats) if num_feats is empty
    #kk =-1*min(2,len(num_feats))
    
    #order = sorted(zip(order,num_feats), key=lambda pair: abs(pair[1]-num_feats[kk]))
    order = sorted(zip(order,num_feats), key=lambda pair: pair[1])
    #num_feats = list(d for _,d in order)
    estimated_features = []
    for o in order:
        n = o[1]+1
        diffs = list([c[0],np.abs(n-c[1])] for c in cm)
        r = sorted(diffs, key=lambda pair: pair[1])[0][0]
        if r < .95 and r != 0.05:
            estimated_features.append([r,n])
        
    #print(estimated_features)

    #plot kde
    plt.figure(2)
    plt.plot(s,e)
    
    plt.scatter(persistence,y,c='r')
    plt.scatter(s[sig_peaks],e[sig_peaks],c='b')
    plt.scatter(s[sig_min],e[sig_min],c='g')
    plt.xlabel('Persistence')
    plt.ylabel('Estimated Density')
    plt.plot(s,e,linewidth=2,c='b')
    plt.axvline(x=(5/8)*np.max(s))
    
    #plt.plot(s,np.zeros(np.shape(s))+np.mean(e[minima]),c='green')
    #plt.plot(s,np.zeros(np.shape(s))+np.mean(e[ma]),c='orange')
    #plt.scatter(s[minima],e[minima],c= 'green')
    #plt.scatter(s[ma],e[ma],c='orange')
    #plt.axvline(x=cutoff,color='b')
    #plt.axvline(x=cutoff2,color='b')
    if show:
        plt.show()

    #extract significant peaks
    #print(len(list(p for p in persistence if p>cutoff2)))
    return estimated_features

def estimate_peaks(samples,scores):
    filtration = gd.SimplexTree()
    minima, maxima = argrelextrema(scores, np.less)[0], argrelextrema(scores, np.greater)[0] 
    for i in range(len(samples)):
        filtration.insert([i],np.max(scores)-scores[i])
        if i < len(samples)-1:
            filtration.insert([i,i+1],max([np.max(scores)-scores[i],np.max(scores)-scores[i+1]]))
    pers = filtration.persistence()
    sig_peaks = []
    x = []
    y2 = []
    cutoff = list(abs(i[1][1]-i[1][0]) for i in pers)[min(3,len(pers)-1)]

    for i in pers:
        #x = [rTree.index(i), rTree.index(i)]
        #print(adata.uns['reeb'].filtration([i[0]]),adata.uns['reeb'].filtration([i[-1]]))
        x.append(i[1][0])
        if np.abs(i[1][1]-i[1][0]) >= cutoff or np.abs(i[1][1]-i[1][0]) >=0.5: 
            sig_peaks.append(-1*i[1][0]+np.max(scores))
        if i[1][1] < 100000:
            y2.append(i[1][1])
        else:
            y2.append(0)
    plt.xlabel('Birth')
    plt.ylabel('Death')
    ticks = np.linspace(np.min(x),np.max(y2),5)
    labels = np.around(-1*(np.linspace(np.min(x),np.max(y2),5)+np.max(scores)),decimals=2)
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels)
    plt.scatter(x,y2,c='r')

    sig_peaks = [maxima[i] for i in range(len(maxima)) for ex in sig_peaks if abs(scores[maxima[i]]-ex)<0.0001]
    sig_min = [0]
    for i in range(len(sig_peaks)-1):
        
            if scores[sig_peaks[i]:sig_peaks[i+1]].size > 0:
                #We need to rule out situations where we misidentify a minima because its score is close to the actual minima's
                p = next(ex for ex in range(sig_peaks[i],sig_peaks[i+1]) if scores[ex] == np.min(scores[sig_peaks[i]:sig_peaks[i+1]]))
                sig_min.append(p)
    sig_min.append(len(samples)-1)
    return sig_peaks,sig_min
    
def get_sub_features(cocycles,diagram_0d,feature,feature_list):
    
    t,j = next(((set(cocycles[j]),j) for j in range(len(cocycles)) if set(cocycles[j])<feature),([],[])) 
    if t != []:
        feature_list.append((feature-t,diagram_0d[j][2]))
            #feature_list.append((t,diagram_0d[j][2]))
            #cocycles.remove(cocycles[j])
        feature_list = get_sub_features(cocycles,diagram_0d,feature-t,feature_list)
    return feature_list

def map_stable_clusters(num,diagram_0d,cocycles,clusterings,adata,filt=1,plots="on",order='size'):

    #Redirect if we're doing manual plotting
    if plots == "manual":
        return map_clusters_manual(num,diagram_0d,cocycles,clusterings,adata)
    
    #Get spatial coordinates
    spatial = adata.obsm['spatial']
    #Sort persistent homology results by death time
    diagram_0d,cocycles = zip(*sorted(zip(diagram_0d,cocycles),key=lambda x: x[0][2],reverse=True))
    #Filter out non-persistent results if desired.
    index = next(i for i in range(len(diagram_0d)) if diagram_0d[i][2]<filt)
    diagram_0d = diagram_0d[index:]
    cocycles = list(cocycles[index:])

    z_track = []


    for n in range(num):
        if n == 0:
            feature_list = [(set(cocycles[0])-set(cocycles[1]),diagram_0d[n][2])]
        elif n == len(cocycles):
            break
        else:
            feature_list = [(set(cocycles[n]),diagram_0d[n][2])]

        feature_list = get_sub_features(cocycles,diagram_0d,feature_list[0][0],feature_list)
        
                
        x = spatial[:,0]
        y = spatial[:,1]
        z = np.ones(len(x))
        #iterate backwards through features to find first value of each point
        feature_list = sorted(feature_list,key= lambda x: x[1],reverse=True)
        tracker = set()
        for i in range(len(feature_list)-1, -1, -1):
            
            spots = set()
            for clust in feature_list[i][0]:
                spots = set.union(spots,clusterings[clust])
            
            z[list(spots-tracker)] = feature_list[i][1]
            tracker = set.union(tracker,spots)
        #print(feature_list)
        #print(set(z))
        #nontrivial = True
        """ for guy in z_track:
            
            zz = list(i for i in range(len(z)) if (abs(z[i]-guy[i])<.05 and z[i] < .9))
            gg = list(i for i in range(len(z)) if z[i] < .9)
            if len(gg) < 1:
                nontrivial = False
                break
            if len(zz)/len(gg) > 0.5:
                nontrivial = False
                break
        if nontrivial:
            z_track.append(z) """ 
        z_track.append(z)
    for i in range(len(z_track)):
        max = np.max(list(z for z in z_track[i]))
        min = np.min(list(z for z in z_track[i]))-.00001
                
        z  = list(1-(z_track[i][j]-min)/(max-min+.00001) for j in range(len(z_track[i])))
        z_track[i] = z

    if order == 'size':   
        z_track = sorted(z_track,key = lambda x : sum(x)) 
    if plots == 'on':
        for z in z_track:
            plt.figure(figsize=(8, 20/3))
            
            plt.scatter(x,-y,c=logit(z),cmap='magma',s=30,edgecolors='k',linewidths=.5)
            #plt.set_xticklabels([])
            #plt.set_yticklabels([])
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('logit(coreness)')
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
        plt.show()
    return z_track

def map_clusters_manual(num,diagram_0d,cocycles,clusterings,adata,filt=1,manual="off"):
    
    from mpl_point_clicker import clicker
    diagram_0d,cocycles = zip(*sorted(zip(diagram_0d,cocycles),key=lambda x: x[0][2],reverse=True))
    index = next(i for i in range(len(diagram_0d)) if diagram_0d[i][2]<filt)
    diagram_0d = diagram_0d[index:]
    cocycles = list(cocycles[index:]) 
    
    exit = False
    while not exit:
        z_track = []
        spatial = adata.obsm['spatial']
        plt.figure(figsize=(8, 20/3))
        plt.rcParams.update({'font.size': 30})
        ax = plt.gca()
        ax.scatter(spatial[:,0],spatial[:,1],c='k')
        ax.set_title('Click to visualize domains')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        
        klicker = clicker(ax, ["event"], markers=["*"],markersize=20)
        plt.show()
        xy = klicker.get_positions()['event']
        
        input_ind = np.argmin((spatial[:,0]-xy[-1][0])**2+(spatial[:,1]-xy[-1][1])**2)
        #print(input_ind)
        input_cluster = list(i for i in range(len(clusterings)) if input_ind in clusterings[i])
        #print(input_cluster)
        x = spatial[:,0]
        y = spatial[:,1]
        
        for index in range(len(input_cluster)):
            clust = input_cluster[index]
            print(clust)
            z = np.ones(len(x))
            feature_list = list((set(cocycles[i]),diagram_0d[i][2]) for i in range(len(cocycles)) if clust in cocycles[i])
            
            feature_list = sorted(feature_list,key= lambda x: x[1],reverse=True)
            feature_list.append(({clust},0))
            print(feature_list)
            tracker = set()

            tracker = set()
            for i in range(len(feature_list)-1, -1, -1):
            
                spots = set()
                for cc in feature_list[i][0]:
                    spots = set.union(spots,clusterings[cc])
                z[list(spots-tracker)] = feature_list[i][1]
                tracker = set.union(tracker,spots)
            
            #Check if the current feature overlaps too heavily with an already visualized one.
            nontrivial = set(j for j in range(len(z)) if z[j]<.8)
            for guy in z_track:
                
                g = set(j for j in range(len(guy)) if guy[j]<.8)
                
                if len(nontrivial.intersection(g))/len(nontrivial.union(g)) > 0.9:
                    nontrivial = False
                    break
            if nontrivial:        
                z_track.append(z) 
        for i in range(len(z_track)):
            max = np.max(list(z for z in z_track[i]))
            min = np.min(list(z for z in z_track[i]))-.00001
                
            z  = list(1-(z_track[i][j]-min)/(max-min+.00001) for j in range(len(z_track[i])))
            z_track[i] = z
            plt.figure(figsize=(8, 20/3))
            plt.scatter(x,y,c=logit(z),cmap='magma',s=30,edgecolors='k',linewidths=.5)
            plt.scatter(xy[-1][0],xy[-1][1],marker='*',s=100,c='r')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('logit(coreness)')
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            
        plt.show()
        exit = input("Press E to exit")
        if exit == 'E':
            return z_track     
        


#Return a one-hot weighted list for every cluster in pandas categorical clustering
def clusters_to_weighted(clusters):
    
    weights = list(np.zeros(len(clusters)) for i in range(len(clusters.cat.categories.tolist())))

    for i in range(len(clusters)):
        if isinstance(clusters.iloc[i],float):
            
            continue
        index = next(j for j in range(len(clusters.cat.categories.tolist())) if clusters.cat.categories.tolist()[j] == clusters.iloc[i])
        weights[index][i] = 1
    return weights


def cluster_wasserstein(cluster1,cluster2,M):
    c1 = cluster1.copy()
    c2 = cluster2.copy()
    for n in range(len(cluster2)):
        if c1[n] < 0.01:
            c1[n] = 0
        if c2[n]<0.01:
            c2[n] = 0
    c1 = c1/np.sum(c1)
    c2 = c2/np.sum(c2)
    
    d = ot.emd2(c1,c2,M)
    
    return d

def ground_truth_benchmark(ground_truth,multiscale,spatial):
    
    ground_truth = clusters_to_weighted(ground_truth)
    
    optimal_costs = []
    for g in ground_truth:
        g_cost = []
        dmat = ot.dist(spatial,spatial,metric='euclidean')
        ma = np.max(dmat)
        M = dmat/np.max(dmat)
        for m in multiscale:
            
            d = cluster_wasserstein(g,m,M)
            g_cost.append(d)
        optimal_costs.append((min(g_cost),np.argmin(g_cost)))
        print((min(g_cost),int(np.argmin(g_cost))))

    return optimal_costs,ma



def clusters_homology(embedding,clusters=''):
    if clusters != '':
        fig_num=0
        for cluster in clusters:
            fig_num +=1
            cluster = list(cluster)
            e = embedding[cluster,:]
            rips_complex = gd.RipsComplex(points=e)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
            dmat = np.zeros((simplex_tree.num_vertices(),simplex_tree.num_vertices()))
            #print(np.shape(dmat))
            for f in simplex_tree.get_filtration():
                if len(f[0]) == 2:
                    dmat[f[0][0],f[0][1]] = f[1]
                    dmat[f[0][1],f[0][0]] = f[1]
            #print(union_find_dmat(dmat,edge_cut=10000000000))
            pers = simplex_tree.persistence()
            gd.plot_persistence_barcode(pers)
            plt.show()
            p = list(f[1][1]-f[1][0] for f in pers if f[1][1] != np.inf)
            avg_pers = np.mean(p)
            var_pers = np.var(p)
            print('Average persistence: ',avg_pers)
            print('Persistence variance: ',var_pers)
    else:
        
        rips_complex = gd.RipsComplex(points=embedding)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        dmat = np.zeros((simplex_tree.num_vertices(),simplex_tree.num_vertices()))
        #print(np.shape(dmat))
        for f in simplex_tree.get_filtration():
            if len(f[0]) == 2:
                dmat[f[0][0],f[0][1]] = f[1]
                dmat[f[0][1],f[0][0]] = f[1]
        #print(union_find_dmat(dmat,edge_cut=10000000000))
        pers = simplex_tree.persistence()
        gd.plot_persistence_barcode(pers)
        plt.show()
        p = list(f[1][1]-f[1][0] for f in pers if f[1][1] != np.inf)
        avg_pers = np.mean(p)
        var_pers = np.var(p)
        print('Average persistence: ',avg_pers)
        print('Persistence variance: ',var_pers)



    sys.settrace
    
        
    adata.uns['edge_persistence'] = adata.uns['edge_complex'].persistence()
    adata.uns['simplified_edge'] = simplify(adata.uns['edge_complex'],epsilon)
    adata.uns['simplified_edge_persistence'] = adata.uns['simplified_edge'].persistence()
        
    
    return adata #grid_persistence#,v#grid_persistence,v

def ph_leiden(leiden_complex):

    p = leiden_complex.persistence()
    
    x = [p0[1][0] for p0 in p]
    y = [p0[1][1] for p0 in p]
    for index, item in enumerate(y):
        if item == np.inf:
            y[index] = 1
    plt.figure(9)
    plt.scatter(x,y)
    plt.xlim((-.1,1))
    plt.ylim((0,1))
    plt.plot(np.linspace(0,1,num=100),np.linspace(0,1,num=100),'r')
    plt.axvline(x=0.05,c='r')
    plt.show()

def union_find_dmat(dmat, edge_cut):

    N = dmat.shape[0]

    class Node:
        def __init__ (self, loc, birth = 0):
            self.loc = loc
            self.parent = self.loc
            self.birth = birth
        def __str__(self):
            return self.label
        def __int__(self):
            return self.loc

    def Union(x, y, L):
        xRoot = Find(x,L)
        yRoot = Find(y,L)
        if xRoot.loc != yRoot.loc:
            if xRoot.birth > yRoot.birth:
                xRoot.parent = yRoot.loc
                return xRoot.loc # This should return the one that got killed
            else:
                yRoot.parent = xRoot.loc
                return yRoot.loc # This should return the one that got killed

    def Find(x,L):
        if x.parent == x.loc:
            return L[x.parent]
        else:
            return Find(L[x.parent], L)

    # A list of allowed neighbors for each node
    nb_dic = {}
    for i in range(N):
        tmp_list = []
        for j in range(N):
            if dmat[i][j] <= edge_cut and i!=j:
                tmp_list.append(j)
        nb_dic[i] = tmp_list

    simplex_collection = []
    simplex_index = {}
    for i in range(N):
        simplex_collection.append( ( [i], 0.0 ) )
    for i in range(N-1):
        for j in range(i+1, N):
            if j in nb_dic[i]:
                simplex_collection.append( ([i,j], dmat[i][j]))

    filtration = []
    tmp_list = []
    for s in simplex_collection:
        tmp_list.append(s[1])
    simplex_order = np.argsort(tmp_list)
    simplex_order[:N] = np.arange(N)[:]
    cnt = 0; complex_filtration_detail = [];
    for i in simplex_order:
        if len(simplex_collection[i][0]) == 1:
            complex_filtration_detail.append( [ (0,[]), simplex_collection[i][1], set(simplex_collection[i][0]) ] )
            simplex_index[(i)] = cnt
            cnt += 1
        elif len(simplex_collection[i][0]) == 2:
            n1 = simplex_collection[i][0][0];
            n2 = simplex_collection[i][0][1];
            complex_filtration_detail.append( [ (1, [n1,n2]), simplex_collection[i][1], set(simplex_collection[i][0]) ] )
            simplex_index[(n1,n2)] = cnt
            cnt += 1
    persDgm_pairs = []
    L = {0: Node(0, birth = 0)}
    Cocycles = [[i] for i in range(N)]
    Cocycle_fvalues = [[0.0] for i in range(N)]
    for i in range(1, len(complex_filtration_detail)):
        f = complex_filtration_detail[i]
        if f[0][0] == 0:
            L[i] = Node(i, birth=i)
        else:
            [n1,n2] = f[0][1]
            killed = Union(L[n1], L[n2], L)
            if n1 == killed:
                lived = n2;
            else:
                lived = n1
            if killed != None:
                pair = (L[killed].birth, i)
                persDgm_pairs.append(pair)
                # print killed, n1, n2, dmat[n1,n2], L[killed].parent
                Cocycles[L[killed].parent].extend(Cocycles[killed])
                Cocycle_fvalues[L[killed].parent].extend([dmat[n1,n2] for i in range(len(Cocycles[killed]))])

    setReps = [Find(L[v],L).loc for v in L.keys()]
    persDgm = []
    pair_births = []
    for d in persDgm_pairs:
        persDgm.append([complex_filtration_detail[d[0]][1], complex_filtration_detail[d[1]][1]])
        pair_births.append(d[0])
    unpaired = list(set(setReps))
    for i in unpaired:
        persDgm.append([complex_filtration_detail[i][1], np.inf])
        pair_births.append(i)
    pair_births = np.asarray(pair_births)
    pair_births_index = np.argsort(pair_births)
    diagram_0d = []
    for i in pair_births_index:
        d = persDgm[i]
        diagram_0d.append([0,d[0],d[1]])
    # print diagram_0d

    # print Cocycles
    # print Cocycle_fvalues
    return [diagram_0d, Cocycles, Cocycle_fvalues]



def simplify(simp_complex,epsilon):
    
    simplified_complex = simp_complex.copy()
    simp_complex.compute_persistence()
    pairs = simp_complex.persistence_pairs()
    
    elim_forest = identify_pairs(simp_complex,pairs,epsilon)
    nodes = list(elim_forest.nodes)
   
    for component in networkx.connected_components(elim_forest):
            
        directed_tree,root = build_tree(simplified_complex,list(component),elim_forest)
        for node in list(component):
            if node == root:
                continue
            subtree = networkx.DiGraph()
            subtree = build_subtree(directed_tree,node,subtree)
            filtrations = list(simp_complex.filtration([n]) for n in subtree.nodes)
            simplified_complex.assign_filtration([node],min(filtrations))
    
    for x in simplified_complex.get_filtration():
        if len(x[0]) == 2 and tuple(x[0]) in elim_forest.edges:
            simplified_complex.assign_filtration(x[0],\
                max([simplified_complex.filtration([x[0][0]]),simplified_complex.filtration([x[0][1]])]))
    simplified_complex.compute_persistence()
    pairs = simplified_complex.persistence_pairs()
    elim_forest = identify_pairs(simplified_complex,pairs,epsilon)
    nodes = list(elim_forest.nodes)
       
    return simplified_complex



def identify_pairs(simp_complex,pers_pairs,epsilon):
    
    elim_forest = networkx.Graph()
    
    for pair in pers_pairs:
        p = (simp_complex.filtration(pair[0]),simp_complex.filtration(pair[1]))
        if abs(p[0]-p[1]) <= epsilon:
            elim_forest.add_edge(pair[1][0],pair[1][1])
    return elim_forest

def build_tree(simp_complex,nodes,forest):
    filtrations = list(simp_complex.filtration([i]) for i in nodes)
    root = nodes[np.argmin(filtrations)]
    tree = networkx.DiGraph()
    tree = add_edges(root,nodes,forest,tree)
    return tree, root

def add_edges(root,nodes,forest,tree):
    nodes.remove(root)
    for edge in forest.edges:
        if root in edge:
            partner = next(x for x in edge if x != root)
            if partner in nodes:
                tree.add_edge(root,partner)
                tree = add_edges(partner,nodes,forest,tree)
    return tree

def build_subtree(directed_tree,node,subtree):
    subtree.add_node(node)
    for edge in directed_tree.edges:
        if node == edge[0]:
            subtree.add_edge(edge[0],edge[1])
            build_subtree(directed_tree,edge[1],subtree)
    return subtree


#debugging
def toy():
    #t = list()
    t = gd.SimplexTree()
    nodes = [1,2,3,4,7,8,10,11,14]
    for i in nodes:
        t.insert([i],i)
    
    t.insert([2,4],5)
    t.insert([3,4],6)
    t.insert([4,7],9)
    t.insert([1,10],12)
    t.insert([3,11],13)
    t.insert([1,3],15)
    t.insert([3,14],16)
    t.insert([1,8],17)
    
    return t

def toy2():
    #t = list()
    t = gd.SimplexTree()
    for i in range(0,25):
        if i in [0,1,4,9,14,19,24]:
            #t.append(([i],1))
            t.insert([i],0)
        elif i in [2,3,15]:
            #t.append(([i],0.5))
            t.insert([i],.4)
        elif i == 5:
            t.insert([i],.2)
        elif i in [6,12]:
            t.insert([i],1/8)
        elif i == 7:
            t.insert([i],.5)
        elif i in [8,13,18,23]:
            t.insert([i],3/8)
        elif i in [10,21]:
            t.insert([i],.6)
        elif i in [11,17]:
            t.insert([i],.75)
        elif i in [16]:
            t.insert([i],5/8)
        elif i == 20:
            t.insert([i],1)
        elif i == 22:
            t.insert([i],.8)
        
    for i in range(0,25):
        if not i%5 == 4:
            #t.append(([i,i+1],max(t[i][1],t[i+1][1]))) 
            #t.append(([i+1,i],max(t[i][1],t[i+1][1])))
            t.insert([i,i+1],max(t.filtration([i]),t.filtration([i+1]))+.01) 
        if i+5<25:
            #t.append(([i,i+5],max(t[i][1],t[i+5][1])))
            #t.append(([i+5,i],max(t[i+5][1],t[i][1])))
            t.insert([i,i+5],max(t.filtration([i]),t.filtration([i+5]))+.01) 
            
        if not i%5 == 4 and i+5<25:
            t.insert([i,i+6],max(t.filtration([i]),t.filtration([i+6]))+.01)
    
    return t





            

def main():
    print(gd.__debug_info__)
    print("+ Installed modules are: " + gd.__available_modules)
    print("+ Missing modules are: " + gd.__missing_modules)
    print(gd.representations.kernel_methods)
    t=toy()
    #print(list(t.get_filtration()))
    #betti_lower_link(t,6)
    new = simplify(t,3)
    fig,axs = plt.subplots(2)
    #print(adata.uns['grid_persistence'])  
    for i in t.persistence():
        #x = [rTree.index(i), rTree.index(i)]
        #print(adata.uns['reeb'].filtration([i[0]]),adata.uns['reeb'].filtration([i[-1]]))
        
        x = i[1][0]
        if i[1][1] < 100000:
            
            y = i[1][1]
        else:
            y = 17
            
        
        axs[0].scatter(x,y,c='r')
    #axs[0].plot([x/100 for x in range(0,100)],[x/100 for x in range(0,100)],'b')
    #axs[0].plot([x/100 for x in range(0,100)],[(x+30)/100 for x in range(0,100)],'b')
    axs[0].plot(range(1,18),range(1,18),'b')
    axs[0].plot(range(1,18),range(4,21),'b')
    axs[0].set_title('Unsimplified')
    #print(adata.uns['grid_persistence']) 
    num_pairs = 0 
    for i in new.persistence():
        #x = [rTree.index(i), rTree.index(i)]
        #print(adata.uns['reeb'].filtration([i[0]]),adata.uns['reeb'].filtration([i[-1]]))
        x = i[1][0]
        if i[1][1] < 100000:
            
            y = i[1][1]
        else:
            y = 17

        axs[1].scatter(x,y,c='r')
    #axs[1].plot([x/100 for x in range(0,100)],[x/100 for x in range(0,100)],'b')
    #axs[1].plot([x/100 for x in range(0,100)],[(x+30)/100 for x in range(0,100)],'b')
    axs[1].plot(range(1,18),range(1,18),'b')
    axs[1].plot(range(1,18),range(4,21),'b')
    axs[1].set_title('Simplified')
    plt.show()
   # print()
   #case_2_1()
    
#main()