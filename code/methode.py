import numpy as np
import copy
import sys
sys.path.append("../..")

def read_file(fichier):
    """
    str -> G(S, A)
    construit un graphe depuis un fichier
    """
    f = open("../instance/"+fichier, "r")

    # lecture sommets
    S = []
    f.readline()
    nb_sommet = int(f.readline().replace("\n", ""))
    f.readline()
    for i in range(nb_sommet):
        S.append(int(f.readline().replace("\n", "")))

    # lecture arêtes
    A = [[] for i in range(nb_sommet)]
    f.readline()
    nb_arete = int(f.readline().replace("\n", ""))
    f.readline()
    for i in range(nb_arete):
        u, v = f.readline().replace("\n", "").split(" ")
        A[S.index(int(u))].append(int(v))
        A[S.index(int(v))].append(int(u))
    
    for l in A:
        l.sort()

    f.close()

    return (S, A)

# 2 - Graphes

def delete_sommet(G, v):
    sommet, arete = G

    for l in arete:
        if v in l:
            l.remove(v)

    arete.pop(sommet.index(v))
    sommet.remove(v)
    return (sommet, arete)

def delete_ens_sommet(G, V):
    G_prim = copy.deepcopy(G)
    for v in V:
        G_prim = delete_sommet(G_prim, v)
    return G_prim

def degre_sommet(G):
    """
    retourne dans l'ordre de sommet les degrés des sommets
    """
    return [len(i) for i in G[1]]

def degre_max_sommet(G):
    sommet, arete = G
    t_degre = degre_sommet(G)
    return sommet[np.argmax(t_degre)]

def generate_graphe(n, p):
    sommet = [i for i in range(n)]
    arete = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                succes = np.random.rand()
                if succes > p:
                    arete[i].append(j)
                    arete[j].append(i)
    return (sommet, arete)

# 3 - Méthodes approchées

def getListArete(G):
    """
    retourne la liste des arêtes d'un graphe
    """
    A = []
    sommet, adj = G
    for i in range(len(sommet)):
        for v in adj[i]:
            if (((sommet[i], v) not in A) and ((v, sommet[i]) not in A)):
                A.append((sommet[i], v))
    return A

def algo_couplage(G):
    C = set()
    l_arete = getListArete(G)
    for i in range(len(l_arete)):
        u, v = l_arete[i]
        if ((u not in C) and (v not in C)):
            C.add(u)
            C.add(v)
    return C

def algo_glouton(G):
    sommet, arete = copy.deepcopy(G)
    C = set()
    while(sum([len(arete[i]) for i in range(len(arete))]) != 0):
        v = degre_max_sommet((sommet, arete))
        C.add(v)
        sommet, arete = delete_sommet((sommet, arete), v)
    return C


def compare_algo(t_c, t_g):
    coef = []
    for i in range(len(t_c)):
        if len(t_c[i]) != 0 and len(t_g[i]) != 0:
            coef.append(len(t_c[i])/len(t_g[i]))
    return np.mean(coef)

def branchement(G):
    C = set()
    l_arete = getListArete(G)
    if l_arete:
        for arete in l_arete:
            u, v = arete
            C_u = branchement(delete_sommet(copy.deepcopy(G), u))
            C_u.add(u)
            C_v = branchement(delete_sommet(copy.deepcopy(G), v))
            C_v.add(v)
            if len(C_u) > len(C_v):
                return C.union(C_v)
            else:
                return C.union(C_u)
    else:
        return C
    
G = read_file("../instance/exemple_instance.txt")
#G = ([0, 1, 2, 3, 4], [[1], [0, 2], [1, 3], [2, 4], [3]])
C = branchement(G)
print(C)