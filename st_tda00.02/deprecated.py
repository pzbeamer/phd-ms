import numpy as np
import gudhi as gd
# OLD UNION FIND
def sf():
    for i in range(0,num_cells):

        for j in range(i+1,num_cells):
            if kernel == 'radius_kernel':
                k = radius_kernel(dspatial,i,j,radius)
            elif kernel == 'exponential_kernel':
                k = exponential_kernel(dspatial,i,j,eta=radius)
            elif kernel == 'lorentz_kernel':
                k = lorentz_kernel(dspatial,i,j,eta=radius)
            
            sf[i,0] += k*dpca[i,j]
            sf[j,0] += k*dpca[i,j]
            sf[i,1] += k
            sf[j,1] += k
    sf = np.divide(sf[:,0],sf[:,1])
    sf = np.nan_to_num(sf)
    adata.obs['sf']= sf

def union_find(simplex_collection):
    N = len([i[0][0] for i in simplex_collection if len(i[0]) == 1])
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
    # Edit from line 68 to 84, delete neighbor dictionary and alter the filtration values

    simplex_index = {}

    filtration = []
    tmp_list = []
    for s in simplex_collection:
        tmp_list.append(s[1]+len(s[0])/1000)
    
    simplex_order = np.argsort(tmp_list)
    #simplex_order[:N] = np.arange(N)[:]
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
    
    filter = complex_filtration_detail[0][1]
        
    for i in range(1, len(complex_filtration_detail)):
        f = complex_filtration_detail[i]

        if f[0][0] == 0:
            L[list(f[2])[0]] = Node(list(f[2])[0], birth=list(f[2])[0])
        else:
            [n1,n2] = f[0][1]
            killed = Union(L[n1], L[n2], L)
            if n1 == killed:
                lived = n2
            else:
                lived = n1
            if killed != None:
                pair = (L[killed].birth, i)
                persDgm_pairs.append(pair)
                # print killed, n1, n2, dmat[n1,n2], L[killed].parent
                Cocycles[L[killed].parent].extend(Cocycles[killed])
                Cocycle_fvalues[L[killed].parent].extend([next(x for x in simplex_collection if x[0] == [n1,n2]) for i in range(len(Cocycles[killed]))])

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

def new(simplex_collection):
    #print(simplex_collection)
    def Union(i, j, f):
        xRoot = Find(i,f)
        yRoot = Find(j,f)
    
        if xRoot > yRoot:
            return yRoot
        else:
            return xRoot
    

    def Find(i,f):
        
        if f[i][3] == f[i][0]:
            return f[i][3]
        else:
            parent = next(y for y in range(0,len(f)) if f[y][0]==f[i][3])
            return Find(parent, f)
        
    #Simplex collection is a list of tuples.
    #Tuples contain node/edge index and filtration values. 
    #Nodes must be listed before edges.
    N = len([i[0][0] for i in simplex_collection if len(i[0]) == 1])
    nodeCollection = simplex_collection[slice(N)]
    edgeCollection = simplex_collection[slice(N,len(simplex_collection))]
    nodeEdges = list()
    for i in nodeCollection:
        nodeEdges.append([n[0][1] for n in edgeCollection if i[0][0] == n[0][0]])

    tmp_list = []
    for s in simplex_collection:
        tmp_list.append(s[1]+s[0][0]/10000000000)
    simplex_order = np.argsort(tmp_list)
    tmp_list = []
    
    node_filtration_detail = []
    edge_filtration_detail = []
    for i in simplex_order:
        if len(simplex_collection[i][0])==1:
            node_filtration_detail.append(\
                (simplex_collection[i][0][0],\
                simplex_collection[i][1],\
                [n[0][1] for n in edgeCollection if simplex_collection[i][0][0] == n[0][0]],\
                simplex_collection[i][0][0]))
        else:
            edge_filtration_detail.append((simplex_collection[i][0],simplex_collection[i][1]))

    #print(node_filtration_detail)
    Cocycles = [n[0] for n in nodeCollection]
    filter = node_filtration_detail[0][1]
    x = next(i for i in range(0,len(node_filtration_detail)) if node_filtration_detail[i][1] > filter)
    x1 = 0
    y = next(i for i in range(0,len(edge_filtration_detail)) if edge_filtration_detail[i][1] > filter)
    y1 = 0
    while filter <= node_filtration_detail[-1][1]:
        
        f = node_filtration_detail[slice(x)]
        fcomp = node_filtration_detail[slice(x,len(node_filtration_detail))]
        g = edge_filtration_detail[slice(y)]
        gcomp = edge_filtration_detail[slice(y,len(edge_filtration_detail))]
        #print(g)
        for i in range(y1,len(g)):
            
            comp1 = next(p for p in range(0,len(f)) if f[p][0] == g[i][0][0])
            #print([p for p in range(0,len(f)) if f[p][0] == g[i][0][1]])
            #print(g[i][0][1])
            comp2 = next(p for p in range(0,len(f)) if f[p][0] == g[i][0][1])
            if f[comp1][0]==f[comp1][3]:
                Cocycles[f[comp1][0]].extend(Cocycles[f[comp2][0]])
        #print(filter)
        for i in range(y1,len(g)):
            #print(g[i][0][1])
            comp1 = next(p for p in range(0,len(f)) if f[p][0] == g[i][0][0])
            comp2 = next(p for p in range(0,len(f)) if f[p][0] == g[i][0][1])
            root = Union(comp1,comp2,f)
            f[comp1] = (f[comp1][0], f[comp1][1],f[comp1][2],root)
            f[comp2] = (f[comp2][0], f[comp2][1],f[comp2][2],root)
            #f,Cocycles = find(i,j,f,Cocycles)
        
        for i in range(x1,len(f)):
            Cocycles[f[i][3]].extend(Cocycles[f[i][0]])
            if f[i][0] != f[i][3]:
                pp = [x for x in Cocycles[f[i][0]] if nodeCollection[x][1] > nodeCollection[f[i][0]][1]]
                Cocycles[f[i][0]] = [f[i][0]]
                Cocycles[f[i][0]].extend(pp)
            else:
                pp = [x for x in Cocycles[f[i][0]] if nodeCollection[x][1] >= nodeCollection[f[i][0]][1]]
                Cocycles[f[i][0]] = pp
                    
        f.extend(fcomp)
        g.extend(gcomp)
        node_filtration_detail = f.copy()
        edge_filtration_detail = g.copy()
        if x < len(node_filtration_detail):
            filter = node_filtration_detail[x][1]
        x1 = x
        y1 = y
        if x == len(node_filtration_detail):
            filter+=1
        elif filter == node_filtration_detail[-1][1]:  
            x = len(node_filtration_detail)
            y = len(edge_filtration_detail)
        else:
            x = next(i for i in range(0,len(node_filtration_detail)) if node_filtration_detail[i][1] > filter)
            y = next(i for i in range(0,len(edge_filtration_detail)) if edge_filtration_detail[i][1] > filter)
            
    for i in range(0,len(node_filtration_detail)):
            f = node_filtration_detail
            if f[i][0] != f[i][3]:
                pp = [x for x in Cocycles[f[i][0]] if nodeCollection[x][1] > nodeCollection[f[i][0]][1]]
                Cocycles[f[i][0]] = [f[i][0]]
                Cocycles[f[i][0]].extend(pp)
            else:
                pp = [x for x in Cocycles[f[i][0]] if nodeCollection[x][1] >= nodeCollection[f[i][0]][1]]
                Cocycles[f[i][0]] = pp

    placeholder = []
    for i in Cocycles:
        res = []
        [res.append(x) for x in i if x not in res]
        placeholder.append(res)
    Cocycles = placeholder
    return Cocycles


#OLD BETTI LOWER LINK FOR SIMPLE GRID COMPLEX
def betti_lower_link(grid_complex,index):
    
    size = np.sqrt(grid_complex.num_vertices())
    notright = index%size != size-1
    notbottom = index+size < grid_complex.num_vertices()
    notleft =  index%size != 0
    nottop = index > size
    link = gd.SimplexTree()
    
    if notbottom and notright and notleft and nottop:
        link.insert([index-1],grid_complex.filtration([index-1]))
        link.insert([index+1],grid_complex.filtration([index+1]))
        link.insert([index-size],grid_complex.filtration([index-size]))
        link.insert([index+size],grid_complex.filtration([index+size]))
        link.insert([index-size-1],grid_complex.filtration([index-size-1]))
        link.insert([index-size+1],grid_complex.filtration([index-size+1]))
        link.insert([index+size-1],grid_complex.filtration([index+size-1]))
        link.insert([index+size+1],grid_complex.filtration([index+size+1]))
        link.insert([index-size-1,index-size],\
            max([grid_complex.filtration([index-size-1]),grid_complex.filtration([index-size])]))
        link.insert([index-1,index+size],\
            max([grid_complex.filtration([index+size]),grid_complex.filtration([index-1])]))
        link.insert([index-size,index+1],\
            max([grid_complex.filtration([index-size]),grid_complex.filtration([index+1])]))
        link.insert([index-size,index-1],\
            max([grid_complex.filtration([index-size]),grid_complex.filtration([index-1])]))
    else:
        return [1,0]
            
    max_filtration = grid_complex.filtration([index])
    lower_link = gd.SimplexTree()
    count = 0
    print(max_filtration)
    for simplex in link.get_filtration():
        if simplex[1] <= max_filtration:
            count+=1

    if count == 0:
        return [0,0]
    elif  count == 1:
        return [1,0]
    elif count == 2:
        return [2,0]
    elif count == 3:
        return[1,0]
    elif count == 8:
        return[1,1]
    else:
        return[1,0]
    
def toy():
    #t = list()
    t = gd.SimplexTree()
    for i in range(0,25):
        if i in [7,11,13,17]:
            #t.append(([i],1))
            t.insert([i],1)
        elif i == 12:
            #t.append(([i],0.5))
            t.insert([i],0.75)
        else:
            #t.append(([i],0))
            t.insert([i],0)
    for i in range(0,25):
        if not (i+1)%5 == 0:
            #t.append(([i,i+1],max(t[i][1],t[i+1][1]))) 
            #t.append(([i+1,i],max(t[i][1],t[i+1][1])))
            t.insert([i,i+1],max(t.filtration([i]),t.filtration([i+1]))) 
        if i+5<25:
            #t.append(([i,i+5],max(t[i][1],t[i+5][1])))
            #t.append(([i+5,i],max(t[i+5][1],t[i][1])))
            t.insert([i,i+5],max(t.filtration([i]),t.filtration([i+5]))) 
        if not (i+1)%5 == 0 and i+5<25:
            t.insert([i,i+6],max(t.filtration([i]),t.filtration([i+6])))
        if not (i+1)%5 == 1 and i+5<25:
            t.insert([i,i+4],max(t.filtration([i]),t.filtration([i+4])))
    
    
    return t

def simplify_grid(grid_complex,epsilon):
    
    #grid_complex = adata.uns['grid_complex']
    grid_persistence = grid_complex.persistence()
    num_vertices = grid_complex.num_vertices()
    dimension = np.sqrt(num_vertices)
    pairs = grid_complex.persistence_pairs()
    betti = []
    
    for i in range(0,num_vertices):
        left = (i%dimension == 0)
        right = i%dimension == dimension-1
        top = i<dimension
        bottom = i + dimension >= num_vertices
        if left or right or top or bottom:
            betti.append([1,0])
        else: 
            betti.append(betti_lower_link(grid_complex,i))
    
    elim = identify_pairs(grid_complex,grid_persistence,pairs,betti,epsilon)
    print(elim)

    simplices = list(grid_complex.get_filtration())
    V = []
    
    for simplex in simplices:
        if len(simplex[0]) == 1:
            V.append((simplex[0][0],simplex[1],betti[simplex[0][0]]))
    
    reorder = np.argsort(list(vertex[1] for vertex in V))
    V = [V[i] for i in reorder]
    print(simplices)

    for e in elim:
        cancel(V,e,simplices)
    #print(V)

def identify_pairs(simp_complex,persistence,pers_pairs,betti,epsilon):
    
    elim = []
    
    for pair in pers_pairs:
    
        p = (simp_complex.filtration(pair[0]),simp_complex.filtration(pair[1]))
        if abs(p[0]-p[1]) < epsilon:
            node1 = pair[0][0]
            node2 = list(pair[1][i] for i in [0,1] if pair[1][i]!=node1)[0]
            betti_nums  = (betti[node1],betti[node2])
            if betti_nums == ([0,0],[2,0]):
                e = ((node1,node2),(p[0],p[1]),betti_nums)
            elif betti_nums == ([2,0],[1,1]):
                e = ((node2,node1),(p[1],p[0]),betti_nums)
            else:
                continue
            if not e in elim:
                elim.append(e)

    #order elim in increasing filtration value order
    reorder = np.argsort(list(max(f[1]) for f in elim))
    elim = [elim[i] for i in reorder]

    return elim


def cancel(V,e,filtration):
    s = (e[0][0],e[1][0],e[2][0])
    t = (e[0][1],e[1][1],e[2][1])
    
    simplices = list(map(itemgetter(0), filtration))
        
    size = np.sqrt(len(V))
    num_vertices = size
    t_ind = next(i for i in range(0,len(V)) if V[i][0] == t[0])
    print(t)
    T = [V[t_ind]]
    W = V[0:t_ind]
    U = V[t_ind+1:None]
    w = W[-1]
    link_T = link([t],filtration)
    
    while w[0] != s[0]:
    
        if not w[0] in link_T:
            W.remove(w)
            U.insert(0,w)
        elif w[2] == [1,0] and is_contractible(networkx.intersection(star(T,filtration,category='lower'),\
                                                                     link([w],filtration))):
            W.remove(w)
            T.insert(0,w)
            link_T = update_link(link_T,w[0],filtration)
        elif w[2] == [1,0]:
            star_T = star(T,filtration,category='lower')
            link_W = link([w],filtration)

            starT_linkW = networkx.intersection(star_T,link_W)
            component = min(networkx.connected_components(starT_linkW),key=len)
            
            #select edges for subdivision
            subdivide = []
            simplices = list(map(itemgetter(0),filtration))
            lower_link_w = link([(4,4)],filtration,category='lower')
            edge = share_edge(lower_link_w.nodes,component,simplices)
            node = next(n for n in edge if n in lower_link_w.nodes)
            subdivide.append(node)

            while node not in list(map(itemgetter(0),U)):
                edge = share_edge([node],component,simplices)
                if edge == []:
                    edge = share_edge([node],list(map(itemgetter(0),U)),simplices)
                    node = next(n for n in edge if n != node)
                    subdivide.append(node)
            
                else:
                    simplices.remove(edge)
                    node = next(n for n in edge if n != node)
                    subdivide.append(node)
                    

            v1filts = list(T[-1][1]+i*(U[0][1]-T[-1][1])/(len(subdivide)-2) for i in range(1,len(subdivide)-1))
            v2filts = list(w[1]+i*(T[0][1]-w[1])/(len(subdivide)-1) for i in range(1,len(subdivide)-1))
            new_v1  = list()
            new_v2  = list()
            for i in range(1,len(subdivide)-1):
                
                new_v1.append((num_vertices,v1filts[i-1],[1,0]))
                new_v2.append((num_vertices+1,v2filts[i-1],[1,0]))
                edge = next(v for v in filtration if set(v[0]) == {subdivide[i],w[0]})
                filtration.remove(edge)
                filtration.extend([([new_v1[i-1][0]],new_v1[i-1][1]),([new_v2[i-1][0]],new_v2[i-1][1])])
                num_vertices += 2
                filtration.append(([w[0],new_v1[i-1][0]],new_v1[i-1][1]))
                filtration.append(([new_v1[i-1][0],new_v2[i-1][0]],new_v1[i-1][1]))
                filtration.append(([subdivide[i],new_v2[i-1][0]],\
                                    next(f[1] for f in filtration if f[0] == [subdivide[i]])))
                
            filtration.append(([subdivide[0],new_v1[0][0]],new_v1[0][1]))
            filtration.append(([subdivide[0],new_v2[0][0]],new_v2[0][1]))
            filtration.append(([new_v1[-1][0],subdivide[-1]],\
                                next(f[1] for f in filtration if f[0] == [subdivide[-1]])))
            filtration.append(([new_v2[-1][0],subdivide[-1]],\
                                next(f[1] for f in filtration if f[0] == [subdivide[-1]])))
            if len(new_v1) > 1:
                for i in range(0,len(new_v1)-1):
                    filtration.append(([new_v1[i][0],new_v1[i+1][0]],new_v1[i][1]))
                    filtration.append(([new_v2[i][0],new_v2[i+1][0]],new_v2[i][1]))

            U = new_v1 + U
            T = new_v2 + T    

            W.remove(w)
            T.insert(0,w)
            link_T = link(T,filtration)

                
        
        elif w[2] == [2,0]:
            if t[0] == w[0] + 1 or t[0] == w[0] -1:
                neighbor_vertex_1 = next(v for v in filtration if v[0] == [w[0]-size])
                neighbor_vertex_2 = next(v for v in filtration if v[0] == [w[0]+size])
                neighbor_edge_1 = next(v for v in filtration if v[0] == [w[0]-size,w[0]])
                neighbor_edge_2 = next(v for v in filtration if v[0] == [w[0],w[0]+size])
                
            elif t[0] == w[0] + size or t[0] == w[0]-size:
                neighbor_vertex_1 = next(v for v in filtration if v[0] == [w[0]-1])
                neighbor_vertex_2 = next(v for v in filtration if v[0] == [w[0]+1])
                neighbor_edge_1 = next(v for v in filtration if v[0] == [w[0]-1,w[0]])
                neighbor_edge_2 = next(v for v in filtration if v[0] == [w[0],w[0]+1])
                
            filtration.remove(neighbor_edge_1)
            filtration.remove(neighbor_edge_2)

            v1 = ([num_vertices],np.mean([w[1],W[-1][1]]))
            v2 = ([num_vertices+1],np.mean([w[1],U[0][1]]))
            x = ([num_vertices+2],np.mean([v1[1],W[-1][1]]))
            num_vertices+=3

            filtration.extend([v1,v2,x])
                
            filtration.append([w[0],v1[0][0]],max([w[1],v1[1]]))
            filtration.append([w[0],v2[0][0]],max([w[1],v2[1]]))
            filtration.append([v1[0][0],x[0][0]],max([v1[1],x[1]]))
            filtration.append([v2[0][0],x[0][0]],max([v2[1],x[1]]))
            filtration.append([neighbor_vertex_1[0][0],x[0][0]],max([neighbor_vertex_1[1],x[1]]))
            filtration.append([neighbor_vertex_2[0][0],x[0][0]],max([neighbor_vertex_2[1],x[1]]))
            #add some point into the graph.

            U.insert(0,(v2[0],v2[1],[1,0]))
            U.insert(0,(v1[0],v1[1],[1,0]))
            U.insert(0,(x[0],x[1],[1,1]))
            W.remove(w)
            T.insert(0,w)
            link_T = update_link(link_T,w,filtration)
            

        w = W[-1]
    
    #Now w = s
    W.remove(w)
    T.insert(0,w)

    return V,filtration

def share_edge(A,B,simplices):
    for a in A:
        for b in B:
            if [a,b] in simplices:
                return [a,b]
            if [b,a] in simplices:
                return [b,a]
    return []

def star(T,filtration,category = 'regular'):
    star = networkx.Graph()
    minFiltration = min(list(map(itemgetter(1),T)))
    for t in T:
        for f in filtration:
            if len(f[0])==3 and t[0] in f[0]:
                filt0 = next(g[1] for g in filtration if g[0] == [f[0][0]])
                filt1 = next(g[1] for g in filtration if g[0] == [f[0][1]])
                filt2 = next(g[1] for g in filtration if g[0] == [f[0][2]])
                if category == 'regular':
                    star.add_node(f[0][0],filtration = filt0)
                    star.add_node(f[0][1],filtration = filt1)
                    star.add_node(f[0][2],filtration = filt2)
                    star.add_edge(f[0][0],f[0][1],filtration = max([filt0,filt1]))
                    star.add_edge(f[0][0],f[0][2],filtration = max([filt0,filt2]))
                    star.add_edge(f[0][1],f[0][2],filtration = max([filt1,filt2]))
                else:
                    if filt0 < minFiltration:
                        star.add_node(f[0][0],filtration = filt0)
                    if filt1 < minFiltration:
                        star.add_node(f[0][1],filtration = filt1)
                    if filt2 < minFiltration:
                        star.add_node(f[0][2],filtration = filt2)
                    if filt0 < minFiltration and filt1 < minFiltration:
                        star.add_edge(f[0][0],f[0][1],filtration = max([filt0,filt1]))
                    if filt0 < minFiltration and filt2 < minFiltration:
                        star.add_edge(f[0][0],f[0][2],filtration = max([filt0,filt2]))
                    if filt1 < minFiltration and filt2 < minFiltration:
                        star.add_edge(f[0][1],f[0][2],filtration = max([filt1,filt2]))
    return star

def link(T,filtration,category = 'regular'):
    link = networkx.Graph()
    #Iterate over nodes in T and filtration
    minFiltration = min(list(map(itemgetter(1),T)))
    
    for t in T:
        for f in filtration:
            if len(f[0])==2 and t[0] in f[0]:
                
                nodeIndex = next(f[0][i] for i in [0,1] if f[0][i] !=t[0])
                nodeFiltration = next(g[1] for g in filtration if g[0] == [nodeIndex])
                if nodeFiltration < minFiltration or category == 'regular':
                    link.add_node(nodeIndex,filtration =nodeFiltration)

            if len(f[0])==3 and t[0] in f[0]:
                edge = list(f[0][i] for i in [0,1,2] if f[0][i] !=t[0])
                
                edgeFiltration = max([next(g[1] for g in filtration if g[0] == [edge[0]]),\
                               next(g[1] for g in filtration if g[0] == [edge[1]])])
                
                if edgeFiltration < minFiltration or category == 'regular':
                    link.add_edge(edge[0],edge[1],filtration = edgeFiltration)
                    
                
    return link

def update_link(link,w,filtration):
    for f in filtration:
        if len(f[0])==2 and w[0] in f[0]:
            nodeIndex = next(f[0][i] for i in [0,1] if f[0][i] !=w[0])
            nodeFiltration = next(g[1] for g in filtration if g[0] == [nodeIndex])
            link.add_node(nodeIndex,filtration =nodeFiltration)
        if len(f[0])==3 and w[0] in f[0]:
            edge = list(f[0][i] for i in [0,1,2] if f[0][i] !=w[0])
            edgeFiltration = max([next(g[1] for g in filtration if g[0] == [edge[0]]),\
                               next(g[1] for g in filtration if g[0] == [edge[1]])])
            link.add_edge(edge[0],edge[1],filtration = edgeFiltration)
    return link

def is_contractible(simp_complex):
    
    b1 = len(list(networkx.connected_components(simp_complex)))
    b2 = simp_complex.number_of_edges() + b1 - simp_complex.number_of_nodes()
    if b1 == 1 and b2 == 0:
        return True
    else:
        return False
    



def betti_lower_link(simp_complex,index):
    
    link = networkx.Graph()
    max_filtration = simp_complex.filtration([index])
    filtration = list(simp_complex.get_filtration())

    edges = list(f for f in filtration if len(f[0])==2 and index in f[0])
    link_nodes = []

    for edge in edges:
        node = list([edge[0][i] for i in [0,1] if edge[0][i] != index])
        if node not in link_nodes and simp_complex.filtration(node)<max_filtration:
            link_nodes.append(node)
            link.add_node(node[0])

    triangles = list(f for f in filtration if len(f[0])==3 and index in f[0])
    link_edges = []
    for t in triangles:
        edge = list([t[0][i] for i in [0,1,2] if t[0][i] != index])
        if edge not in link_edges and \
        simp_complex.filtration([edge[0]]) < max_filtration and \
        simp_complex.filtration([edge[1]]) < max_filtration :
            link_edges.append(edge)
            link.add_edge(edge[0],edge[1])

    
    #print(edges)
    #print(list(link.get_filtration()))
    components = len(list(networkx.connected_components(link)))
    betti = [components,len(link_edges)+components-len(link_nodes)]
    
    return betti
    
# DEBUGGIN STUFF
def case_2_1():

    #initialize Edelsbrunner example figure 4
    filtration = []
    for i in range(0,18):
        if i == 16 or i == 17:
            filtration.append(([i],i-11))
        else:
            filtration.append(([i],i))

        
            
        if i == 4:
            continue
        elif i == 17: 
            filtration.append(([17,4],6))
            filtration.append(([17,0],6))
            filtration.append(([17,0,4],6))
        elif i == 16:
            filtration.append(([16,4],5))
            filtration.append(([16,17],6))
            filtration.append(([16,17,4],6))
        elif i == 15:
            filtration.append(([15,4],15))
            filtration.append(([15,16],15))
            filtration.append(([15,16,4],15))
        elif i == 3:
            filtration.append(([3,4],4))
            filtration.append(([3,5],5))
            filtration.append(([3,4,5],5))
        else:
            filtration.append(([i,4],max([i,4])))
            filtration.append(([i,i+1],i+1))
            filtration.append(([i,i+1,4],max([i,i+1,4])))

    star_T = networkx.Graph()
    star_T.add_nodes_from([0,4,3,4,5,16,17])
    star_T.add_edges_from([(16,4),(16,17),(17,4),(0,17),(0,4),(3,5),(5,6)])
    link_W = link([(4,4,[1,0])],filtration)
    w = (4,4,[1,0])
    num_vertices = 18
    U = list((f[0][0],f[1]) for f in filtration if f[1]>6 and len(f[0])==1)
    
    starT_linkW = networkx.intersection(star_T,link_W)
    component = min(networkx.connected_components(starT_linkW),key=len)
            
            #componentFiltrations = list(s[1] for s in componentNodes)
            #componentNodes = componentNodes[np.argsort(componentFiltrations)]
            
            #select edges for subdivision
    subdivide = []
    simplices = list(map(itemgetter(0),filtration))
    lower_link_w = link([w],filtration,category='lower')
    edge = share_edge(lower_link_w.nodes,component,simplices)
    node = next(n for n in edge if n in lower_link_w.nodes)
    subdivide.append(node)
    
    while node not in list(map(itemgetter(0),U)):
        edge = share_edge([node],component,simplices)
        if edge == []:
            edge = share_edge([node],list(map(itemgetter(0),U)),simplices)
            node = next(n for n in edge if n != node)
            subdivide.append(node)
            
        else:
            simplices.remove(edge)
            node = next(n for n in edge if n != node)
            subdivide.append(node)
    print(subdivide)
    v1filts = list(6+i*(U[0][1]-6)/(len(subdivide)-1) for i in range(1,len(subdivide)-1))
    v2filts = list(w[1]+i*(5-w[1])/(len(subdivide)-1) for i in range(1,len(subdivide)-1))
    new_v1  = list()
    new_v2  = list()
    for i in range(1,len(subdivide)-1):
                
        new_v1.append((num_vertices,v1filts[i-1],[1,0]))
        new_v2.append((num_vertices+1,v2filts[i-1],[1,0]))
        edge = next(v for v in filtration if set(v[0]) == {subdivide[i],w[0]})
        filtration.remove(edge)
        filtration.extend([([new_v1[i-1][0]],new_v1[i-1][1]),([new_v2[i-1][0]],new_v2[i-1][1])])
        num_vertices += 2
        filtration.append(([w[0],new_v1[i-1][0]],new_v1[i-1][1]))
        filtration.append(([new_v1[i-1][0],new_v2[i-1][0]],new_v1[i-1][1]))
        filtration.append(([subdivide[i],new_v2[i-1][0]],\
                            next(f[1] for f in filtration if f[0] == [subdivide[i]])))
                
    filtration.append(([subdivide[0],new_v1[0][0]],new_v1[0][1]))
    filtration.append(([subdivide[0],new_v2[0][0]],new_v2[0][1]))
    filtration.append(([new_v1[-1][0],subdivide[-1]],\
                        next(f[1] for f in filtration if f[0] == [subdivide[-1]])))
    filtration.append(([new_v2[-1][0],subdivide[-1]],\
                        next(f[1] for f in filtration if f[0] == [subdivide[-1]])))
    if len(new_v1) > 1:
        for i in range(0,len(new_v1)-1):
            filtration.append(([new_v1[i][0],new_v1[i+1][0]],new_v1[i][1]))
            filtration.append(([new_v2[i][0],new_v2[i+1][0]],new_v2[i][1]))

    U = new_v1 + U
    print(U)
    print(filtration)


def make_reeb(adata):
    layers = adata.uns['layer_filtration']
    reeb_filtration = list()

    #Set up variables
    hierarchy = networkx.Graph()
    num_comps = layers['num_comps']
    comps = layers['comps']
    total_comps = int(np.sum(num_comps))
    #Want the number of components in each layer and all preceding layers
    num_comps = [np.sum(num_comps[0:i]) for i in range(0,num_comps.shape[0]+1)]
    
    #Initialize reeb graph
    reeb = gd.SimplexTree()
    #Add a node for every component
    hierarchy.add_nodes_from(range(0,total_comps))

    #Assign filtrations according to layers for individual nodes
    for i in range(0,num_comps[-1]):
        reeb.insert([i],np.min([num_comps.index(a) for a in num_comps if i<a]))
        reeb_filtration.append(([i],np.min([num_comps.index(a) for a in num_comps if i<a])))

    #Check every pair of adjacent layers
    for i in range(1,layers.shape[0]):

        #List of pairs of components in adjacent layers
        e = [(a,b) for a in comps[i-1] for b in comps[i]]

        #List of node indexes for daughter-parent pairs of components across adjacent layers
        edges = [(comps[i-1].index(a)+num_comps[i-1],comps[i].index(b)+num_comps[i]) for (a,b) in e if a.issubset(b)]
        #Add edge between daughter parent pairs
        hierarchy.add_edges_from(edges)
          
        #Add filtration for edges
        for t in edges:
            reeb.insert(list(t),max([reeb.filtration([t[0]]),reeb.filtration([t[1]])]))
            reeb_filtration.append((list(t),max([reeb.filtration([t[0]]),reeb.filtration([t[1]])])))
        
    networkx.write_gml(hierarchy,'./hierarchy.gml')
    adata.uns['reeb_graph'] = hierarchy
    adata.uns['reeb_filtration'] = reeb_filtration
    adata.uns['reeb'] = reeb

    return adata

def make_grid(adata,size):
    xdata = list(adata.obsm['spatial'][:,0])
    ydata = list(adata.obsm['spatial'][:,1])
    num_grid = size**2
    num_cells = len(xdata)
    
    ##particular to this data set, needs to be altered to work in general
    h = (max(xdata)-min(xdata))/size
    xmin = min(xdata)
    xmax = max(xdata)
    ymin = (min(ydata)+max(ydata))/2 - (max(xdata)-min(xdata))/2
    ymax = (min(ydata)+max(ydata))/2 + (max(xdata)-min(xdata))/2
    xgrid = np.linspace(xmin,xmax,num=size)
    ygrid = np.linspace(ymin,ymax,num=size)
    
    
    M = np.zeros([num_grid,num_cells])
    for i in range(0,num_grid):
            for j in range(0,num_cells):
                xind = int(np.floor(i/size))
                yind = i%size
                if abs(xgrid[xind] - xdata[j])<h and abs(ygrid[yind] - ydata[j])<h:
                    M[i,j] = (abs(xgrid[xind] - xdata[j])/h)*(abs(ygrid[yind] - ydata[j])/h)

    for i in range(0,num_grid):
        factor = M[i,:].max()
        if factor != 0:
            M[i,:] = M[i,:]/factor
            
    #vector of scalar grid components, listed in ascending dictionary order of (x,y) component
    zeros = []
    scalargrid = np.matmul(M,adata.obs['sf'])
    for i in range(0,len(scalargrid)):
        if scalargrid[i] < 5:
            zeros.append(i)
    #Build matrix of coordinates
    x = np.empty(num_grid)
    y = np.empty(num_grid)
    length = int(np.sqrt(num_grid))
    for i in range(0,length):
        #dictionary order
        x[length*i:length*(i+1)] = np.full(length,xgrid[i])
        y[length*i:length*(i+1)] = ygrid

    adata.uns['full_grid'] = np.column_stack((x,y,scalargrid))
    adata.uns['xgrid'] = xgrid
    adata.uns['ygrid'] = ygrid
    adata.uns['scalar_grid'] = scalargrid
    adata.uns['grid_size'] = num_grid
    adata.uns['zeros'] = zeros
    return adata


def make_grid_complex(adata):
    grid_size= adata.uns['grid_size']
    size = np.sqrt(grid_size)
    scalar_grid = adata.uns['scalar_grid']
    gridSimplex = gd.SimplexTree()
    gridSimplexFull = gd.SimplexTree()
    grid_filtration = list()
    #Assign filtrations for individual nodesaccording to scalar field value
    for i in range(0,grid_size):
        if i not in adata.uns['zeros']:
            gridSimplex.insert([i],scalar_grid[i])
            gridSimplexFull.insert([i],scalar_grid[i])
            grid_filtration.append(([i],scalar_grid[i]))
    for i in range(0,grid_size):
        #Add edges between nodes that share an x value
        notright = i%size != size-1
        notbottom = i+size < size**2
        notleft =  i%size != 0
        nottop = i > size
        if notright and i not in adata.uns['zeros'] and i+1 not in adata.uns['zeros']:
            gridSimplex.insert([i,i+1],.0001+max([gridSimplex.filtration([i]),gridSimplex.filtration([i+1])]))
            #grid_filtration.append(([i,i+1],max([gridSimplex.filtration([i]),gridSimplex.filtration([i+1])])))
        #add edges between nodes that share a y value
        if notbottom and i not in adata.uns['zeros'] and i+size not in adata.uns['zeros']:
            gridSimplex.insert([i,i+size],.0001+max([gridSimplex.filtration([i]),gridSimplex.filtration([i+size])]))
            
            #grid_filtration.append(([i,i+size],max([gridSimplex.filtration([i]),gridSimplex.filtration([i+size])])))
        if notright and notbottom and i not in adata.uns['zeros'] and i+size+1 not in adata.uns['zeros']:
            gridSimplex.insert([i,i+size+1],.0001+max([gridSimplex.filtration([i]),gridSimplex.filtration([i+size+1])]))
            
    adata.uns['grid_complex'] = gridSimplex
    #adata.uns['grid_filtration'] = grid_filtration

    return adata

def morse(adata):
    grid_complex = adata.uns['grid_complex']
    simplified_complex = adata.uns['simplified_complex']
    x = adata.uns['full_grid'][:,0]
    y = adata.uns['full_grid'][:,1]
    sf_uns = []
    sf_simp = []
    for i in range(0,len(x)):
        if i not in adata.uns['zeros']:
            sf_uns.append(grid_complex.filtration([i]))
            sf_simp.append(simplified_complex.filtration([i]))
    
    x = [x[i] for i in range(0,len(x)) if i not in adata.uns['zeros']]
    y = [y[i] for i in range(0,len(y)) if i not in adata.uns['zeros']]

    sf_uns = np.array(sf_uns)
    sf_simp = np.array(sf_simp)

    #adata.uns['morse_smale_unsimplified'] = ms.nn_merged_partition(np.array([x,y]), sf_uns, k_neighbors=8)
    #adata.uns['morse_smale_simplified'] = ms.nn_merged_partition(np.array([x,y]), sf_simp, k_neighbors=8)
    return adata

def layer_filtration(adata,r,iterations = 50):

    #unpack scalar field
    sf = adata.obs['sf']
    #calculate layer height
    dheight = (sf.max()-sf.min())/iterations
    #scalar field matrix
    sf_spatial = np.array([adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1],sf]).transpose()
    #set up layer graph
    graph = sp.KDTree(sf_spatial[:,0:2])
    components = networkx.Graph()
    height = sf.min() + dheight

    #Set up layers
    layers = np.zeros(iterations,dtype = [('comps',list),('num_comps',int),('size_comps',list)])

    #Iterate over number of layers in scalar field
    for i in range(0,iterations):
        #Iterate over cells
        for j in range(0,sf.shape[0]):
            #check if cell j falls within the current layer
            if height-dheight< sf.iloc[j] and sf.iloc[j] <= height:
                #Add node to graph
                components.add_node(j)
                #Find neighboring locations within 300 micrometers
                l = graph.query_ball_point(sf_spatial[j,0:2],r)

                #check neighbors 
                for k in l:
                    #if neighbor k falls in current layer, add an edge between j,k
                    if  sf.iloc[k] <= height:
                        components.add_node(k)
                        components.add_edge(j,k)

        #Find connected components + size, and number in each layer
        comps = [set(components.subgraph(c).nodes) for c in connected_components(components)]
        num_comps = len(comps)
        size_comps = [len(x) for x in comps]
        layers[i] = (comps,num_comps,size_comps)

        #Increment height
        height += dheight
    
    adata.uns['layer_filtration'] = layers
    
    return adata

def make_edge_filtration(adata,neighbors,radius=15,embedding='X_scanit'):

    ind =neighbors
    edge_complex = gd.SimplexTree()
    num_cells = adata.obsm['spatial'].shape[0]
    for i in range(0,num_cells):
        for j in range(0,len(ind[i])):
            edge_complex.insert([i,ind[i][j]],sp.distance.euclidean(adata.obsm[embedding][i,:],adata.obsm[embedding][ind[i][j],:]))
    adata.uns['edge_complex'] = edge_complex
    return adata

