

#from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from data import preprocess_leiden,leiden,leiden_filtration,find_neighbors,order_clusterings,cluster_metrics, filt_to_matrix,diff_express_tumor,construct_clustering
from ph import clusters_homology,significant_features, union_find_dmat, map_stable_clusters,ph_leiden,ground_truth_benchmark,clusters_to_weighted
import networkx
import time
import scanpy as sc
import ot
import ast
from scipy.special import logit
import pandas as pd
from scipy.stats import spearmanr
dir = '/home/pbeamer/Documents/st1.0/'
import scipy.stats as ss

in_dir = '/home/pbeamer/Documents/graphst/'
out_dir = '/home/pbeamer/Documents/st1.0/results00.02/embedding'
    
def test_cluster(filename,ydata,folder,out):

    tag = []
    res = np.linspace(start=0.05,stop=1,num=20)
    preprocess_leiden(filename,folder,ydata,resolution=res)
    
    adata = sc.read_h5ad(filename+'_'+ydata+'.h5ad')
    ground_truth = len(adata.obs['cluster'].cat.categories.to_list())
    print(ground_truth)
    cm = []
    size = []
    print('resolutions:')
    for r in res:#in np.linspace(start=0.2,stop=0.5,num=10):
        adata.obs['clusters'+str(r)] = adata.obs['clusters'+str(r)]
        tag.append(str(r))
        print(r)
        cm.append(cluster_metrics(adata.obs['clusters'+str(r)],r,truth=adata.obs['cluster']))
        
    print(cm)
    output = str(cm)
    neighbors = find_neighbors(adata)
    #order = order_clusterings(adata,key='num_clusters',neighbors=neighbors,tags=tag)
    #print(order)
    i = 1
    opt_res=[]
    opt_num=[]
    while i <= 100:

        r = []
        while len(r)<10:
            n = np.random.randint(0,20)
            if n not in r:
                r.append(n)
            
            if len(r) == 10:
                r = sorted(r)
                if r[1] > 6 or r[-2] < 6:
                    r = []
        
        #r = range(20)
        #print(r)
        leiden_complex,_=leiden_filtration(adata,neighbors,index='containment',tags=list(tag[j] for j in r),order=range(10))
        #ph_leiden(leiden_complex)
        leiden_persistence = leiden_complex.persistence()
        est = significant_features(leiden_persistence,cm)
        if len(est)>1:
            print(str(est)+" , "+str(list(tag[j] for j in r)))
            output = output+"\n"+str(est)+" , "+str(list(tag[j] for j in r))
            est = sorted(est,key = lambda x: abs(x[1]-ground_truth))
            
            opt_res.append(est[0][0])
            opt_num.append(est[0][1])
            i += 1
        else:
            print('fail')
            output = output+"\nfail"
    #ph_leiden(leiden_complex)
    output = output+"\n"+str(opt_num)
    print(opt_num)
    output = output+"\n"+'Mean:'+str(np.mean(opt_res))+" "+str(np.mean(opt_num))
    output = output+"\n"+'Median:'+str(np.median(opt_res))+" "+str(np.median(opt_num))
    output = output+"\n"+'Mode:'+str(np.argmax(np.bincount(opt_num)))
    print('Mean:',np.mean(opt_res),np.mean(opt_num))
    print('Median:',np.median(opt_res),np.median(opt_num))
    print('Mode:',np.argmax(np.bincount(opt_num)))
    with open(out+filename+"-03-27.txt", "w") as f:
        f.write(output)
   
def run_res(filename,folder,res):
    adata = sc.read_h5ad(filename+'_graphst.h5ad')
    leiden(adata,res=res,show=True,embedding='X_gst')

def plot_ground_truth(adata,ground_truth = 'clusters'):
    adata = sc.read_h5ad(adata+'.h5ad')
    plt.figure(figsize=(8, 20/3))
    plt.set_cmap('magma')
    c = adata.obs[ground_truth]
    for i in range(0,len(c.cat.categories.tolist())):
        coords = adata.obsm['spatial'][list(n for n in range(0,len(c)) if c.iloc[n]==c.cat.categories.tolist()[i]),:]
        plt.scatter(coords[:,0],coords[:,1],s=30)
    frame1 = plt.gca()
    plt.colorbar()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])    
    #plt.title('# of Ground Truth Clusters:'+str(len(c.cat.categories.tolist())))

    plt.show()

def plot_matching(cluster,cluster2,spatial,save=False):
    x = spatial[:,0]
    y = spatial[:,1]
    
    plt.figure(figsize=(8, 20/3))
    plt.clf()
    z = np.zeros(len(spatial[:,1]))
    z =  np.array(list(zed + .00001 for zed in z))
    z[cluster] = 1-.00001
    
    plt.scatter(x,y,c=logit(z),cmap='magma',s=30,edgecolors='k',linewidths=.5)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('logit(coreness)')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    z = cluster2
    if  save != False:
        plt.savefig(save+'gt.png',format='png',dpi=300)
    plt.figure(figsize=(8, 20/3))
    plt.clf()
    plt.scatter(x,y,c=logit(z),cmap='magma',s=30,edgecolors='k',linewidths=.5)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('logit(coreness)')
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    if  save != False:
        plt.savefig(save+'.png',format='png',dpi=300)
    plt.show()
   

   
def get_num(filename,ydata,folder):
    adata = sc.read_h5ad(filename+'_'+ydata+'.h5ad')
    print(filename)
    print(len(adata.obs['cluster'].cat.categories.to_list()))

def maps(filename,folder='',r=300,ydata=['scanit'],s=1):
    #151673 r=300
    #visium_hne r=15
    ydata = ['scanit','graphst']
    
    res = np.linspace(start=0.1,stop=1,num=10)
    tag = []
    cm  = []
    #preprocess_leiden(filename,folder+'scanit/','scanit',resolution=res)
    preprocess_leiden(filename,folder,'graphst',resolution=res)
    
    #adata_sp = sc.read_h5ad(filename+'_scanit.h5ad')
    adata_sp = sc.read_h5ad(filename+'_graphst.h5ad')
    for r in res:#in np.linspace(start=0.2,stop=0.5,num=10):
        adata_sp.obs['clusters'+str(r)] = adata_sp.obs['clusters'+str(r)]
        tag.append(str(r))
        print(r)
        cm.append(cluster_metrics(adata_sp.obs['cluster'],adata_sp.obs['clusters'+str(r)],r))
    plt.figure(1)
    c = adata_sp.obs['cluster']
    for i in range(0,len(c.cat.categories.tolist())):
        coords = adata_sp.obsm['spatial'][list(n for n in range(0,len(c)) if c.iloc[n]==c.cat.categories.tolist()[i]),:]
        plt.scatter(coords[:,0],coords[:,1])
    plt.title(len(c.cat.categories.tolist()))
    #plt.show()
   
    
    neighbors = find_neighbors(adata_sp)
    order = order_clusterings(adata_sp,key='num_clusters',neighbors=neighbors,tags=tag)#[tag[9],tag[10]])
    point_complex,leiden_complex,cluster_elements=leiden_filtration(adata_sp,neighbors,index='jaccard',tags=tag,order=order,slope=s)
    #print(clusters)
    #print(list(leiden_complex.get_filtration()))
    dmat = filt_to_matrix(leiden_complex)
    #print(dmat)
    #np.savetxt('dmat.csv',dmat)
    diagram_0d,cocycles,_= union_find_dmat(dmat,1.1)
    for i in range(1,10):
        map_stable_clusters(diagram_0d,cocycles,cluster_elements,adata_sp,.05*i)

def correlation(filename):
    wass = []
    
    ami_max = []
    for f in filename:
        with open('results00.02/embedding/'+f+'-05-23.txt') as file:
            lines = [line.rstrip() for line in file]
        max = lines[1]
        max = max.replace(' Max distance:','')
        max = ast.literal_eval(max)
        w = lines[-2]
        w = w.replace(' Cumulative costs:','')
        w = ast.literal_eval(w)
       
        wass.extend(list((d-np.mean(w))/np.mean(w) for d in w))
        a = lines[0]
        a = a.replace(' Max AMI:','')
        a = ast.literal_eval(a)
        ami_max.extend(list((d-np.mean(a))/np.mean(a) for d in a))
        
    print(np.argmax(wass))
    import scipy.stats as ss
    print(ss.spearmanr(ami_max,wass))
    print(ss.pearsonr(ami_max,wass))
    plt.figure(1,figsize=(4.88,3.84))
    plt.scatter(ami_max,wass)
    plt.title('Embedding quality vs NAME Wasserstein Cost',fontsize=25)
    plt.xlabel('Normalized AMI',fontsize=20)
    plt.ylabel('Normalized Wasserstein',fontsize=20)
    plt.show()
    
def get_ami(filename):
    ami_max = []
    r = np.linspace(start=0.05,stop=.95,num=10)
    for i in range(10):
        ami = []
        
        preprocess_leiden(filename,folder='visium-human-dorsalateral_prefrontal_cortex/graphst/',ydata='graphst',resolution=r,t=i)
        adata = sc.read_h5ad(filename+'_graphst.h5ad')
        for res in r:
            ami.append(cluster_metrics(adata.obs['clusters'+str(res)],res,truth=adata.obs['cluster'])[2])
        ami_max.append(np.max(ami))
def plot_violin(files,merfish = False,legend=[]):
    if not legend:
        legend = files
    cost_array = []
    import ast
    for f in files:
        with open('results00.02/embedding/'+f+'-05-23.txt') as file:
            lines = [line.rstrip() for line in file]
        if merfish:
            costs = lines[1]
            costs = costs.replace('Wasserstein Costs:','')
            costs = ast.literal_eval(costs)
            costs = list(a*1000 for (a,b) in costs)
            cost_array.append(costs)
        else:
            best = lines[-1]
            best = best.replace(' Best embedding:','')
            best = ast.literal_eval(best)
            max = lines[1]
            max = max.replace(' Max distance:','')
            max = ast.literal_eval(max)
            costs = lines[3*(best+1)]
            costs = costs.replace(' Wasserstein Costs:','')
            costs = ast.literal_eval(costs)
            costs = list(a for (a,b) in costs)
            cost_array.append(costs)

    plt.figure(1)
    plt.boxplot(cost_array,labels=legend)
    plt.xlabel('Dataset',fontsize=20)
    plt.ylabel('Wasserstein Cost (microns)',fontsize=20)
    
    plt.show()
    
def merfish_mouse(bregma='-9'):
    file = 'merfish_'+bregma+'_unambiguous_graphst'
    input_file = 'MERFISH/'+file
    output_file = input_file
    r = np.linspace(start=0.15,stop=.95,num=8)
    tag = list(str(ree) for ree in r)
    ydata = 'graphst'
    
    if bregma == '-9':
        ground_truth = 'cluster'
    else:
        ground_truth = 'clusters'
    output = ''
    preprocess_leiden(input_file,output_file,emb='X_gst',resolution=r)
    adata = sc.read_h5ad(output_file+'.h5ad')
    print(np.max(ot.dist(adata.obsm['spatial'],adata.obsm['spatial'])))
    
    neighbors = find_neighbors(adata)
    leiden_complex,clusterings = leiden_filtration(adata,neighbors,index='containment',tags=tag,order=range(len(r)))
    dmat = filt_to_matrix(leiden_complex)
    diagram_0d,Cocycles,_=union_find_dmat(dmat,edge_cut=1)
    cluster_weights = map_stable_clusters(len(Cocycles),diagram_0d,Cocycles,clusterings,adata,plots='off')
    costs,m = ground_truth_benchmark(adata.obs[ground_truth],cluster_weights,adata.obsm['spatial'])
    adata.uns['multiscale'] = cluster_weights
    adata.write_h5ad(output_file+'.h5ad')
    costs = list((cost[0]*m,cost[1]) for cost in costs)
    for i in range(len(costs)):
            coords = list(n for n in range(0,len(adata.obs[ground_truth])) if adata.obs[ground_truth].iloc[n]==adata.obs[ground_truth].cat.categories.tolist()[i])
            #plot_matching(coords,cluster_weights[costs[i][1]],adata.obsm['spatial'])
    
    output += "\nWasserstein Costs:"+ str(costs)
    output += "\nCumulative Cost:" + str(sum(list(c[0]*m for c in costs)))
    print(output)
    with open('results00.02/merfish/'+bregma+'/'+file+'-5-25.txt','w') as file:
            file.write(output)


def tumor():
    input_file = 'V1_Human_Invasive_Ductal_Carcinoma_10xvisium_processed_graphst'
    output_file = input_file
    r = np.linspace(start=0.15,stop=.95,num=8)
    r = [0.15,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
    tag = list(str(ree) for ree in r)
    ydata='graphst'
    
    preprocess_leiden(input_file,output_file,emb='X_gst',resolution=r)
    adata = sc.read_h5ad(output_file+'.h5ad')
    #plot_ground_truth(adata)
    neighbors = find_neighbors(adata)
    leiden_complex,clusterings=leiden_filtration(adata,neighbors,index='jaccard',tags=tag,order=range(len(r)))
    dmat = filt_to_matrix(leiden_complex)
    diagram_0d,Cocycles,_= union_find_dmat(dmat,edge_cut=1)
        #print(diagram_0d)
    cluster_weights = map_stable_clusters(len(Cocycles),diagram_0d,Cocycles,clusterings,adata,plots='on')
    adata.uns['multiscale'] = cluster_weights
    
    sc.write('adata_tumor_multiscale.h5ad',adata)
    #ground_truth_benchmark(adata.obs['cluster'],cluster_weights,adata.obsm['spatial'])
def brain():
    input_file = 'visium_hne_graphst'
    output_file= 'visium_hne_graphst'
    r = np.linspace(start=0.15,stop=.95,num=8)
    tag = list(str(ree) for ree in r)
    ydata='graphst'
    preprocess_leiden(input_file,output_file,emb='X_gst',resolution=r)
    adata = sc.read_h5ad(input_file+'.h5ad')
    #plot_ground_truth(adata)
    neighbors = find_neighbors(adata)
    leiden_complex,clusterings=leiden_filtration(adata,neighbors,index='containment',tags=tag,order=range(len(r)))
    dmat = filt_to_matrix(leiden_complex)
    diagram_0d,Cocycles,_= union_find_dmat(dmat,edge_cut=1)
        #print(diagram_0d)
    cluster_weights = map_stable_clusters(len(Cocycles),diagram_0d,Cocycles,clusterings,adata,plots='on')
    adata.uns['multiscale'] = cluster_weights
    adata.write('results00.02/mouse_brain/mouse_brain_multiscale.h5ad')
    #ground_truth_benchmark(adata.obs['cluster'],cluster_weights,adata.obsm['spatial'])

def benchmarks():
    x = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    t = [5,5,6,4,5,9,3,4,0,4,6,1]
    #x = ['151507','151508','151509','151510','151669','151670','151671','151672','151674','151675','151676']
    correlation(x)
    plot_violin(x)

def benchmarks_merfish():
    x = ['merfish/-4/merfish_-4_unambiguous_graphst-5-25',\
         'merfish/-9/merfish_-9_unambiguous_graphst-5-19',\
            'merfish/-14/merfish_-14_unambiguous_graphst-5-25',\
                'merfish/-19/merfish_-19_unambiguous_graphst-5-25',\
                    'merfish/-24/merfish_-24_unambiguous_graphst-5-25']
    plot_violin(x,merfish=True,legend=['Bregma -.04','Bregma -.09','Bregma -.14','Bregma -.19','Bregma -24'])
def embryo():
    
   
    input_file = 'mouse_developmental/E9.5_E1S1.MOSTA_graphst'
    output_file = input_file
    r = np.linspace(start=0.15,stop=.95,num=8)
    tag = list(str(ree) for ree in r)
    ydata='graphst'
    output = ''
    ground_truth = 'annotation'
    preprocess_leiden(input_file,output_file,emb='X_gst',resolution=r)
    adata = sc.read_h5ad(output_file+'.h5ad')
    print(np.max(ot.dist(adata.obsm['spatial'],adata.obsm['spatial'])))
    plot_ground_truth(adata,ground_truth)
    neighbors = find_neighbors(adata)
    leiden_complex,clusterings=leiden_filtration(adata,neighbors,index='containment',tags=tag,order=range(len(r)))
    dmat = filt_to_matrix(leiden_complex)
    diagram_0d,Cocycles,_= union_find_dmat(dmat,edge_cut=1)
        #print(diagram_0d)
    cluster_weights = map_stable_clusters(len(Cocycles),diagram_0d,Cocycles,clusterings,adata,plots='on')
    costs, m = ground_truth_benchmark(adata.obs[ground_truth],cluster_weights,adata.obsm['spatial'])
    for i in range(len(costs)):
            coords = list(n for n in range(0,len(adata.obs[ground_truth])) if adata.obs[ground_truth].iloc[n]==adata.obs[ground_truth].cat.categories.tolist()[i])
            plot_matching(coords,cluster_weights[costs[i][1]],adata.obsm['spatial'])
    output += "\n Wasserstein Costs:"+ str(costs)
    output += "\n Cumulative Cost:" + str(sum(list(c[0]*m for c in costs)))
    print(output)
    with open('results00.02/'+output_file+'-5-08.txt','w') as file:
            file.write(output)
    

def libd(input = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],ind=range(10)):
    for x in input:
        
        output = ''
        cum_wasserstein = []
        #with open('results00.02/embedding/adata_'+x+'-04-05.txt') as file:
                #lines = [line.rstrip() for line in file]
        #output += lines[-2]
        #output += '\n'+lines[-1]
        for i in ind:
            filename = 'adata_'+x

            emb = 'X_gst'
            tag = []
            
            res = np.linspace(start=0.15,stop=.95,num=8)
            
            output += "\n"+str(i)
            input_file= 'visium-human-dorsalateral_prefrontal_cortex/graphst/'+filename+'_gst_'+str(i)
            output_file = filename+'_gst'
            preprocess_leiden(input_file,output_file,emb='X_gst',resolution=res)
        
            adata = sc.read_h5ad(output_file+'.h5ad')

            cm = []
            size = []
            #print('resolutions:')
            for r in res:#in np.linspace(start=0.2,stop=0.5,num=10):
                adata.obs['clusters'+str(r)] = adata.obs['clusters'+str(r)]
                tag.append(str(r))
                #print(r)
                cm.append(cluster_metrics(adata.obs['clusters'+str(r)],r))#adata.obs['cluster']))
                
                #print(cluster_metrics(adata.obs['clusters'+str(r)],r))
                
            #plot_ground_truth(adata)
            neighbors = find_neighbors(adata)

            #order = order_clusterings(adata,key='num_clusters',neighbors=neighbors,tags=tag)
            #print(order)
            
            leiden_complex,clusterings=leiden_filtration(adata,neighbors,index='containment',tags=tag,order=range(len(tag)))
            leiden_persistence = leiden_complex.persistence()
            #ph_leiden(leiden_complex)
            #print(significant_features(leiden_persistence,cm))
            dmat = filt_to_matrix(leiden_complex)
            diagram_0d,Cocycles,_= union_find_dmat(dmat,edge_cut=1)
            #print(diagram_0d)
            cluster_weights = map_stable_clusters(len(Cocycles),diagram_0d,Cocycles,clusterings,adata,plots='manual')
            adata.uns['multiscale'] = cluster_weights
            adata.write_h5ad('libd_'+x+'_'+str(i)+'_multiscale.h5ad')
            adata.write_h5ad(out_dir+'libd_'+x+'_'+str(i)+'_multiscale.h5ad')
            costs,m = ground_truth_benchmark(adata.obs['cluster'],cluster_weights,adata.obsm['spatial'])
            costs = list((c[0]*m,c[1]) for c in costs)
            costs = list((float(c[0]*m),int(c[1])) for c in costs)
            cum_wasserstein.append(sum(list(c[0] for c in costs)))
            output += "\n Wasserstein Costs:"+ str(costs)
            output += "\n Cumulative Cost:" + str(cum_wasserstein[-1])
            print(costs)
            #for j in range(len(costs)):
                #coords = list(n for n in range(0,len(adata.obs['cluster'])) if adata.obs['cluster'].iloc[n]==adata.obs['cluster'].cat.categories.tolist()[j])
                #plot_matching(coords,cluster_weights[costs[j][1]],adata.obsm['spatial'])
        print(cum_wasserstein)
        print(np.argmin(cum_wasserstein))
        output += "\n Cumulative costs:"+ str(cum_wasserstein)
        output += "\n Best embedding:" + str(np.argmin(cum_wasserstein))
        with open('results00.02/embedding/'+x+'-05-23.txt','w') as file:
        with open(out_dir+x+'-06-27.txt','w') as file:
            file.write(output)

#Compute differential expression for a set of multi-scale domains
def libd_single_scale_benchmark(input = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']):
    for x in input:
        filename = 'libd_'+x
        output = x
        cum_wasserstein = []
        for i in range(10):
            clusters_as_weighted = []
            
            input_file= dir+filename+'_'+str(i)+'_multiscale'

            adata = sc.read_h5ad(input_file + '.h5ad')
            res = np.linspace(start=0.15,stop=.95,num=8)
            for r in res:
                clusters_as_weighted.extend(clusters_to_weighted(adata.obs['clusters'+str(r)]))
        
            costs,m = ground_truth_benchmark(adata.obs['cluster'],clusters_as_weighted,adata.obsm['spatial'])
            costs = list((c[0]*m,int(c[1])) for c in costs)
            cum_wasserstein.append(sum(list(c[0] for c in costs)))
            output += '\n'+str(i)
            output += '\nCosts:'+str(costs)
            output += '\nCumulative:'+str(cum_wasserstein[-1])
            print(cum_wasserstein[-1])
        print(cum_wasserstein)
        print(np.argmin(cum_wasserstein))
        output += "\nList Cumulative:"+ str(cum_wasserstein)
        output += "\nBest:" + str(np.argmin(cum_wasserstein))
        with open(dir+'results00.02/embedding/'+x+'_ground_truth-06-23.txt','w') as file:
            file.write(output)
        
def plot_single_vs_multiscale(filename,method=None):
    wass_mult = []
    wass_single = []
    wass_mult_array = []
    wass_single_array = []
    for f in filename:
        ww = []
        with open(dir + 'results00.02/embedding/'+f+'-05-23.txt') as file:
            lines = [line.rstrip() for line in file]
        best = lines[-1]
        best = best.replace(' Best embedding:','')
        best = ast.literal_eval(best)
        for i in range(10):
            if method == 'best' and i != best:
                continue
            w = lines[3*i+3]
            
            w = w.replace(' Wasserstein Costs:','')
            #This file is formatted strangely

            if f == '151673':
                w = w.replace('np.float64(','')
                w = w.replace('np.int64(','')
                w = w.replace('))','*')
                w = w.replace(')','')
                w = w.replace('*',')')
            
            w = ast.literal_eval(w)
            w = list(d[0] for d in w)
            ww.extend(w)
            wass_mult.extend(list((d-np.mean(w))/np.mean(w) for d in w))
        wass_mult_array.append(ww)

        ww = []
        with open(dir + 'results00.02/embedding/'+f+'_ground_truth-06-23.txt') as file:
            lines = [line.rstrip() for line in file]

        best = lines[-1]
        best = best.replace('Best:','')
        best = ast.literal_eval(best)
        for i in range(10):
            if method == 'best' and i != best:
                continue
            w = lines[3*i+2]
            w = w.replace('Costs:','')
            
            #This f
            if f == '151673':
                w = w.replace('np.float64(','')
                w = w.replace('np.int64(','')
                w = w.replace('), (','*')
                w = w.replace(')]','&')
                w = w.replace(')','')
                w = w.replace('*','),(')
                w = w.replace('&',')]')
            w = ast.literal_eval(w)
            w = list(d[0] for d in w)
            wass_single.extend(list((d-np.mean(w))/np.mean(w) for d in w))
            ww.extend(w)
        wass_single_array.append(ww)
    r  = spearmanr(wass_mult,wass_single)
    r  = ss.linregress(wass_mult,wass_single)
    print(r)
    print('# of clusters with single better than multiscale:'+str(len(list(wass_mult[i] for i in range(len(wass_mult)) if wass_mult[i]>wass_single[i]))))
    print('# of clusters with multi better than single scale:'+str(len(list(wass_single[i] for i in range(len(wass_mult)) if wass_single[i]>wass_mult[i]))))
    plt.figure()
    plt.scatter(wass_mult,wass_single)
    plt.plot([-1,3],[-1,3], c='m')
    plt.plot([-1,3],[-1*r.statistic,3*r.statistic],c='r')
    plt.legend(['','y=x','y=%.2fx' % r.statistic])
    plt.plot([-1,3],[r.intercept+r.slope*-1,r.intercept+r.slope*3],c='r')
    plt.legend(['','y=x','y=%.2fx' % r.slope])
    plt.xlabel('Multiscale Wasserstein')
    plt.ylabel('Single Scale Wass')
    plt.show()

    
    for i in range(len(wass_mult_array)):
        plt.clf()
        cost_array = []
        legend = []
        cost_array.append(wass_mult_array[i])
        cost_array.append(wass_single_array[i])
        legend.append(filename[i]+' Multiscale')
        legend.append(filename[i]+' Single Scale')
        plt.figure()
        plt.boxplot(cost_array,labels=legend)
        plt.xlabel('Dataset',fontsize=20)
        plt.ylabel('Wasserstein Cost (microns)',fontsize=20)
        plt.show()
    
    

def _test_(i):
    file = 'libd_151672_'+str(i)+'_multiscale.h5ad'
    
    adata = sc.read_h5ad(file)
    with open('results00.02/embedding/151676-05-23.txt') as file:
            lines = [line.rstrip() for line in file]
    costs = lines[3*i+3]
    costs = costs.replace(' Wasserstein Costs:','')
    print(costs)
    costs = ast.literal_eval(costs)
    new_cluster = construct_clustering(adata,set(cost[1] for cost in costs))
    new_cluster = pd.Series(new_cluster)
    nmi = cluster_metrics(new_cluster,0,adata.obs['cluster'])[2]
    print("Cluster Metrics:"+str(nmi))
    return nmi

def libd_ami():
    with open('results00.02/embedding/151674-05-20.txt') as file:
            lines = [line.rstrip() for line in file]
    ami=lines[0]
    ami = ami.replace(' Max AMI:','')
    ami = ast.literal_eval(ami)
    nmi = []
    for i in range(10):
        nmi.append(_test_(i))
    print(nmi)
    import scipy.stats as ss
    print(ss.spearmanr(ami,nmi))
    plt.figure(1)
    plt.plot([0,1],[0,1])
    plt.scatter(ami,nmi)
    plt.title('Embedding quality vs NAME Wasserstein Cost',fontsize=25)
    plt.xlabel('Normalized AMI',fontsize=20)
    plt.ylabel('Normalized Wasserstein',fontsize=20)
    plt.show()

def plot_multiscale(filename,save=True,show=True,inds='all'):
    
    adata = sc.read_h5ad(filename+'.h5ad')
    
    multiscale = adata.uns['multiscale']
    if inds == 'all':
        inds = range(len(multiscale))
    x = adata.obsm['spatial'][:,0]
    y = adata.obsm['spatial'][:,1]
    
    for i in inds:
        
            plt.figure(figsize=(8, 20/3))
            
            plt.scatter(x,-y,c=logit(multiscale[i]),cmap='magma',s=30,edgecolors='k',linewidths=.5)
            #plt.set_xticklabels([])
            #plt.set_yticklabels([])
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('logit(coreness)')
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            if save:
                plt.savefig(filename+str(i)+'.png',format='png')
            if show:
                plt.show()

def plot_gt_libd(num,i):
    adata = sc.read_h5ad('results00.02/embedding/libd_'+num+'_'+str(i)+'_multiscale.h5ad')
    with open('results00.02/embedding/'+num+'-05-23.txt') as file:
            lines = [line.rstrip() for line in file]
    costs = lines[3*i+3]
    costs = costs.replace(' Wasserstein Costs:','')
    costs = ast.literal_eval(costs)
    print(costs)
    costs = list(c[1] for c in costs)
    multiscale = adata.uns['multiscale']
    for i in range(len(costs)):
        coords = list(n for n in range(0,len(adata.obs['cluster'])) if adata.obs['cluster'].iloc[n]==adata.obs['cluster'].cat.categories.tolist()[i])
        plot_matching(coords,multiscale[costs[i]],adata.obsm['spatial'])
def plot_gt_merfish(bregma):
    adata = sc.read_h5ad('MERFISH/merfish_'+bregma+'_unambiguous_graphst.h5ad')
    with open('results00.02/merfish/'+bregma+'/merfish_'+bregma+'_unambiguous_graphst-5-25.txt') as file:
            lines = [line.rstrip() for line in file]
    costs = lines[1]
    costs = costs.replace('Wasserstein Costs:','')
    costs = ast.literal_eval(costs)
    print(costs)
    costs = list(c[1] for c in costs)
    multiscale = adata.uns['multiscale']
    for i in range(len(costs)):
        coords = list(n for n in range(0,len(adata.obs['clusters'])) if adata.obs['clusters'].iloc[n]==adata.obs['clusters'].cat.categories.tolist()[i])
        plot_matching(coords,multiscale[costs[i]],adata.obsm['spatial'],save='results00.02/merfish/'+bregma+str(i))
        

#for b in [-24]:#[-4,-14,-19,-24]:
    #merfish_mouse(bregma=str(b))    
#benchmarks_merfish()
#tumor()
#diff_express_tumor('adata_tumor_multiscale.h5ad',[34,19,39,45,44,24,15,31,33],['4','5'])
#plot_gt_merfish('-24')
#libd(['151673'],ind=[0])
#benchmarks()
#plot_gt_libd('151673',1)
#plot_ground_truth('results00.02/mouse_brain/mouse_brain_multiscale',ground_truth='clusters0.75')
#adata = sc.read_h5ad('results00.02/embedding/libd_151673_1_multiscale.h5ad')
#for res in np.linspace(start=.15,stop=.95,num=8):
    #plot_ground_truth('results00.02/embedding/libd_151673_1_multiscale',ground_truth='clusters'+str(res))

#libd_single_scale_benchmark(['151673'])
files = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
plot_single_vs_multiscale(files)