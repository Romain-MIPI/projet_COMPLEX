import numpy as np
import copy

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

# une fonction qui converti la liste d'adjacent en liste d'arêtes

def algo_couplage(G):
    sommet, arete = G
    C = set()
        

G_gen = generate_graphe(5, 0.5)
G_prim = delete_ens_sommet(G_gen, {1, 2})
print(G_gen)
print(G_prim)
#t = degre_max_sommet(G)
#print(t)