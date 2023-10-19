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

def borne_inf(G_bis):
    """Calcule des valeurs de b1, b2 et b3 afin d'obtenir une borne inf pour le branchement."""
    sommets = G_bis[0]
    aretes = getListArete(G_bis)
    n = len(sommets)
    m = len(aretes)
    couplage = algo_couplage(G_bis)
    delta = degre_max_sommet(G_bis)
    b1 = np.ceil(m/delta)
    b2 = len(couplage)//2
    b3 = ((2*n)-1-np.sqrt(((2*n-1)**2)-(8*m)))/2
    return max(b1,b2,b3)

def branchement(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        current_graph, current_cover = pile.pop()
        aretes = getListArete(current_graph)

        # Élagage si toutes les arêtes sont couvertes
        if len(aretes)==0:
            # Élagage : si la couverture actuelle est meilleure que la meilleure trouvée jusqu'à présent
            if len(current_cover) < b_sup:
                best_cover = current_cover
                b_sup = len(best_cover)
        else:
            # Élagage si la borne supérieure est trop grande
            if borne_inf(current_graph)+len(current_cover) >= b_sup:
                continue
            u,v = aretes[0]
            couplage = algo_couplage(current_graph)
            couplage = couplage.union(current_cover)
            # Mise à jour de la meilleure couverture si nécessaire
            #best_cover = current_cover if len(current_cover) < b_sup else best_cover # on ne sait pas que current_cover est un couverture
            b_sup = min(len(couplage),b_sup)   # len(couplage) n'est pas un couplage de G initiale
            # Exploration des nœuds enfants
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_sommet(current_graph, v),current_cover | {v}))
    return best_cover