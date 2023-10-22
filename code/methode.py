import numpy as np
import copy
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
sys.path.append("../..")

def read_file(chemin_fichier):
    """
    Lit un fichier pour construire et retourner un graphe sous forme de liste d'adjacence.
    
    Parameters:
    - chemin_fichier (str) : Le chemin vers le fichier contenant la représentation du graphe.

    Returns:
    tuple: Un tuple (sommet, adjacences) où 'sommet' est une liste des sommets et 'adjacences' est la liste d'adjacence du graphe.
    
    Exemple:
    >>> generate_graphe("../instance/exemple_instance.txt")
    ([0, 1, 2, 3, 4], [[1, 2], [0, 2, 3], [0, 1, 4], [1], [2]])
    """
    
    with open(chemin_fichier, "r") as fichier:
        
        # Lecture des sommets
        sommets = []
        fichier.readline()
        nombre_sommets = int(fichier.readline().strip())
        fichier.readline()
        for _ in range(nombre_sommets):
            sommets.append(int(fichier.readline().strip()))

        # Lecture des arêtes
        adjacences = [[] for _ in range(nombre_sommets)]
        fichier.readline()
        nombre_aretes = int(fichier.readline().strip())
        fichier.readline()
        for _ in range(nombre_aretes):
            u, v = map(int, fichier.readline().strip().split())
            adjacences[sommets.index(u)].append(v)
            adjacences[sommets.index(v)].append(u)
        
        for liste in adjacences:
            liste.sort()

    return (sommets, adjacences)

######## 2 - Graphes ########

def delete_sommet(G, v):
    """
    Supprime un sommet et toutes ses adjacences d'un graphe.(exo: 2.1.1)

    Parameters:
    - G (tuple): Le graphe représenté par un tuple (sommet, adjacences).
    - v (int): Le sommet à supprimer.

    Returns:
    - tuple: Le nouveau graphe sans le sommet v et ses adjacences.
    """
    sommets, adjacences = deepcopy(G[0]), deepcopy(G[1])
    index_sommets_a_supprimer = [sommets.index(v)]
    for i,l in enumerate(adjacences):
        if v in l:
            l.remove(v)
        if not l:
            index_sommets_a_supprimer.append(i)
    for sommet_index in sorted(index_sommets_a_supprimer,reverse=True):
        adjacences.pop(sommet_index)
        sommets.pop(sommet_index)
    return (sommets, adjacences)

def delete_ens_sommet(G, V):
    """
    Supprime un ensemble de sommets et toutes leurs adjacences d'un graphe.(exo: 2.1.2)

    Parameters:
    - G (tuple): Le graphe représenté par un tuple (sommet, adjacences).
    - v (int): L'ensemble de sommets à supprimer.

    Returns:
    - tuple: Le nouveau graphe sans le sommet v et ses adjacences.
    """
    for v in V:
        G = delete_sommet(G, v)
    return G

def degre_sommets(G):
    """
    Calcule le degré de chaque sommet dans un graphe.(exo: 2.1.3)

    Parameters:
    - G (tuple): Le graphe représenté par un tuple (sommet, adjacences).

    Returns:
    - list[int]: Une liste des degrés pour chaque sommet du graphe.
    """
    return [len(i) for i in G[1]]

def sommet_degree_max(G):
    """
    Trouve le sommet avec le degré maximum dans un graphe.(exo: 2.1.3)

    Parameters:
    - G (tuple): Le graphe représenté par un tuple (sommet, adjacences).

    Returns:
    - int: Le sommet qui a le degré le plus élevé dans le graphe.
    """
    degres = degre_sommets(G)
    max_index = np.argmax(degres)
    return G[0][max_index]

def generate_graphe(n, p):
    """
    Génère un graphe aléatoire sur n sommets.(exo: 2.2)

    Chaque arête (i, j) est présente avec une probabilité p. Les tirages sont 
    indépendants pour les différentes arêtes.

    Paramètres:
    - n (int): Nombre de sommets du graphe.
    - p (float): Probabilité de présence d'une arête entre deux sommets.

    Retour:
    - tuple: Un tuple contenant deux listes. La première liste représente les sommets
             et la deuxième liste représente les arêtes sous forme de listes adjacentes.

    Exemple:
    >>> generate_graphe(5, 0.5)
    ([0, 1, 2, 3, 4], [[1, 2], [0, 2, 3], [0, 1, 4], [1], [2]])
    """
    sommets = [i for i in range(n)]
    adjacences = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                succes = np.random.rand()
                if succes > 1-p:
                    adjacences[i].append(j)
                    adjacences[j].append(i)
    return (sommets, adjacences)

######## 3 - Méthodes approchées ########

def get_liste_aretes(G):
    """
    Retourne la liste des arêtes d'un graphe afin de faciliter l'implementation des algorithmes a la suite.
    
    Parameters:
    - G (tuple): Le graphe, représenté par un tuple (sommet, adjacences).
    
    Returns:
    - list[tuple]: Une liste des arêtes du graphe sous la forme de paires (u, v), où u et v sont des sommets.

    Exemple:
    >>> obtenir_liste_aretes(([0, 1, 2], [[1, 2], [0, 2], [0, 1]]))
    [(0, 1), (0, 2), (1, 2)]
    """
    aretes = []
    sommets, adjacences = G
    for u in sommets:
        for v in adjacences[sommets.index(u)]:
            if (u, v) not in aretes and (v, u) not in aretes:
                aretes.append((u, v))
    return aretes

def algo_couplage(G):
    """
    Calcule une couverture a l'aide de l'algorithme de couplage.

    Parameters:
    - G (tuple): Le graphe, représenté par un tuple (sommet, adjacences).

    Returns:
    - set[int]: Un ensemble de sommets qui forment la couverture.
    """
    C = set()
    aretes = get_liste_aretes(G)
    for u,v in aretes:
        if u not in C and v not in C:
            C.add(u)
            C.add(v)
    return C

def algo_glouton(G):
    """
    Calcule une couverture a l'aide de l'algorithme glouton.

    Parameters:
    - G (tuple): Le graphe, représenté par un tuple (sommet, adjacences).

    Returns:
    - set[int]: Un ensemble de sommets qui forment la couverture.
    """
    sommets, adjacences = G
    C = set()

    while any(adjacences):
        v = sommet_degree_max((sommets, adjacences))
        C.add(v)
        sommets, adjacences = delete_sommet((sommets, adjacences), v)

    return C

def compare_algo(t_c, t_g):
    coef = []
    for i in range(len(t_c)):
        if len(t_c[i]) != 0 and len(t_g[i]) != 0:
            coef.append(len(t_c[i])/len(t_g[i]))
    return max(coef), np.mean(coef)

# 4 -  Séparation et évaluation
def degre_max_sommet(G):
    sommet, l_aretes = G
    t_degre = [len(sublist) for sublist in l_aretes]
    return max(t_degre)

def borne_inf(G_bis):
    """Calcule des valeurs de b1, b2 et b3 afin d'obtenir une borne inf pour le branchement."""
    sommets = G_bis[0]
    aretes = get_liste_aretes(G_bis)
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

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        current_graph, current_cover = pile.pop()
        aretes = get_liste_aretes(current_graph)

        # Élagage si toutes les arêtes sont couvertes
        if len(aretes)==0:
            # Élagage : si la couverture actuelle est meilleure que la meilleure trouvée jusqu'à présent
            if len(current_cover) < len(best_cover):
                best_cover = current_cover
        else:
            u,v = aretes[0]
            # Exploration des nœuds enfants
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_sommet(current_graph, v),current_cover | {v}))
    return best_cover

def branchement_borne(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        current_graph, current_cover = pile.pop()
        aretes = get_liste_aretes(current_graph)

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
            best_cover = couplage if len(couplage) < b_sup else best_cover
            b_sup = min(len(couplage),b_sup)
            # Exploration des nœuds enfants
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_sommet(current_graph, v),current_cover | {v}))
    return best_cover

def branchement_ameliore_q1(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        print("pile =", pile)
        print("best_cover =", best_cover)
        current_graph, current_cover = pile.pop()
        aretes = get_liste_aretes(current_graph)

        # Élagage si toutes les arêtes sont couvertes
        if len(aretes)==0:
            # Élagage : si la couverture actuelle est meilleure que la meilleure trouvée jusqu'à présent
            if len(current_cover) < b_sup:
                best_cover = current_cover
                b_sup = len(best_cover)
                print("changement best_cover =", best_cover)
        else:
            # Élagage si la borne supérieure est trop grande
            if borne_inf(current_graph)+len(current_cover) >= b_sup:
                print("on elague")
                continue
            u,v = aretes[0]
            couplage = algo_couplage(current_graph)
            couplage = couplage.union(current_cover)
            # Mise à jour de la meilleure couverture si nécessaire
            best_cover = couplage if len(couplage) < b_sup else best_cover
            b_sup = min(len(couplage),b_sup)
            # Exploration des nœuds enfants
            current_sommet, current_s_adj = current_graph
            voisin_u = set(current_s_adj[current_sommet.index(u)])
            print("u, v =", u, v)
            print("voisin_u =", voisin_u)
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_ens_sommet(current_graph, voisin_u),current_cover | voisin_u))
    return best_cover

def branchement_ameliore_q2(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        print("pile =", pile)
        print("best_cover =", best_cover)
        current_graph, current_cover = pile.pop()
        aretes = get_liste_aretes(current_graph)

        # Élagage si toutes les arêtes sont couvertes
        if len(aretes)==0:
            # Élagage : si la couverture actuelle est meilleure que la meilleure trouvée jusqu'à présent
            if len(current_cover) < b_sup:
                best_cover = current_cover
                b_sup = len(best_cover)
                print("changement best_cover =", best_cover)
        else:
            # Élagage si la borne supérieure est trop grande
            if borne_inf(current_graph)+len(current_cover) >= b_sup:
                print("on elague")
                continue
            # Recherche du sommet de degré max
            u = sommet_degree_max(current_graph)
            current_sommet, current_s_adj = current_graph
            v = current_s_adj[current_sommet.index(u)][0]   # un sommet voisin de u

            couplage = algo_couplage(current_graph)
            couplage = couplage.union(current_cover)
            # Mise à jour de la meilleure couverture si nécessaire
            best_cover = couplage if len(couplage) < b_sup else best_cover
            b_sup = min(len(couplage),b_sup)
            # Exploration des nœuds enfants
            voisin_u = set(current_s_adj[current_sommet.index(u)])
            print("u, v =", u, v)
            print("voisin_u =", voisin_u)
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_ens_sommet(current_graph, voisin_u),current_cover | voisin_u))
    return best_cover

def write_file(t, fichier):
    f = open(fichier, "w")
    for i in range(len(t)):
        f.write("%d : {"%(i+2))
        for u in t[i]:
            f.write(" %d"%u)
        f.write(" }\n")
    f.close()

def compare_en_n():
    t_couplage = []
    s_couplage = []
    t_glouton = []
    s_glouton = []

    n = [i for i in range(10, 101, 10)]

    for i in n:
        tc_i = []
        tg_i = []
        sc_i = []
        sg_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(i, 0.5)

            #print("algo_couplage:")

            debut_couplage = time.time()
            s = algo_couplage(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            tc_i.append(t)
            sc_i.append(len(s))
            #print("t = %f"%t)
            #print("s =", s)

            #print("\nalgo glouton:")

            debut_glouton = time.time()
            s = algo_glouton(G_gen)
            fin_glouton = time.time()

            t = fin_glouton-debut_glouton
            tg_i.append(t)
            sg_i.append(len(s))
            #print("t = %f"%t)
            #print("s =", s)

        t_couplage.append(np.mean(tc_i))
        t_glouton.append(np.mean(tg_i))
        s_couplage.append(np.mean(sc_i))
        s_glouton.append(np.mean(sg_i))

    # affichage t/n
    plt.plot(n, t_couplage)
    plt.plot(n, t_glouton)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend(["algo_couplage", "algo_glouton"])
    plt.savefig('courbe_t_n.png')
    plt.show()

    # affichage diff/n
    x = np.arange(len(n))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, s_couplage, width, label='solutions couplage')
    rects2 = ax.bar(x + width/2, s_glouton, width, label='solutuon glouton')

    ax.set_xlabel('taille n')
    ax.set_ylabel('nombre de sommets')
    ax.set_title('nombre de sommets trouvé en fonction de n')
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.legend()
    plt.savefig("hist_n.png")
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s_couplage, "solutions_couplage.txt")
    #write_file(s_glouton, "solutions_glouton.txt")

    #print(compare_algo(s_couplage, s_glouton))

    return

def compare_en_p():
    t_couplage = []
    s_couplage = []
    t_glouton = []
    s_glouton = []

    p_n = [i/10 for i in range(1, 11)]

    for p in p_n:
        tc_i = []
        tg_i = []
        sc_i = []
        sg_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(100, p)

            #print("algo_couplage:")

            debut_couplage = time.time()
            s = algo_couplage(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            tc_i.append(t)
            sc_i.append(len(s))
            #print("t = %f"%t)
            #print("s =", s)

            #print("\nalgo glouton:")

            debut_glouton = time.time()
            s = algo_glouton(G_gen)
            fin_glouton = time.time()

            t = fin_glouton-debut_glouton
            tg_i.append(t)
            sg_i.append(len(s))
            #print("t = %f"%t)
            #print("s =", s)

        t_couplage.append(np.mean(tc_i))
        t_glouton.append(np.mean(tg_i))
        s_couplage.append(np.mean(sc_i))
        s_glouton.append(np.mean(sg_i))

    # affichage t/n
    plt.plot(p_n, t_couplage)
    plt.plot(p_n, t_glouton)
    plt.title("courbe de temps en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("temps en s")
    plt.legend(["algo_couplage", "algo_glouton"])
    plt.savefig('courbe_t_p.png')
    plt.show() 

    # affichage diff/n
    x = np.arange(len(p_n))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, s_couplage, width, label='solutions couplage')
    rects2 = ax.bar(x + width/2, s_glouton, width, label='solutuon glouton')

    ax.set_xlabel('probabilité p')
    ax.set_ylabel('nombre de sommets')
    ax.set_title('nombre de sommets trouvé en fonction de p')
    ax.set_xticks(x)
    ax.set_xticklabels(p_n)
    ax.legend(loc="lower right")
    plt.savefig("hist_p.png")
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s_couplage, "solutions_couplage.txt")
    #write_file(s_glouton, "solutions_glouton.txt")

    #print(compare_algo(s_couplage, s_glouton))

    return

G = read_file("../instance/exemple_instance.txt")
print(G)
#G = ([0, 1, 2, 3], [[1], [0, 2, 3], [1], [1]])
#print("Q1 :")
#C = branchement_ameliore_q1(G)
#print(C)

#print("\n-------------------------------------------------\nQ2 :")
#C2 = branchement_ameliore_q2(G)
#print(C2)