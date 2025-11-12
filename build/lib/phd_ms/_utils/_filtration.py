import numpy as np
import gudhi as gd

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

def get_sub_features(cocycles,diagram_0d,feature,feature_list):
    
    t,j = next(((set(cocycles[j]),j) for j in range(len(cocycles)) if set(cocycles[j])<feature),([],[])) 
    if t != []:
        #feature_list.append((feature-t,diagram_0d[j][2]))
        feature_list.append((t,diagram_0d[j][2]))
            #cocycles.remove(cocycles[j])
        feature_list = get_sub_features(cocycles,diagram_0d,feature-t,feature_list)
    return feature_list