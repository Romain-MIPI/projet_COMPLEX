import numpy as np
import copy
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
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
    sommet, arete = list(G[0]), [list(l) for l in G[1]] 

    for l in arete:
        if v in l:
            l.remove(v)

    arete.pop(sommet.index(v))
    sommet.remove(v)
    return (sommet, arete)

def delete_ens_sommet(G, V):
    for v in V:
        G = delete_sommet(G, v)
    return G

def generate_graphe(n, p):
    sommet = [i for i in range(n)]
    arete = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                succes = np.random.rand()
                if succes > 1-p:
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

def sommet_degree_max(G):
    sommet, l_aretes = G
    t_degre = [len(sublist) for sublist in l_aretes]
    return sommet[np.argmax(t_degre)]

def algo_glouton(G):
    sommet, arete = copy.deepcopy(G)
    C = set()
    while(sum([len(arete[i]) for i in range(len(arete))]) != 0):
        v = sommet_degree_max((sommet, arete))
        C.add(v)
        sommet, arete = delete_sommet((sommet, arete), v)
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

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        current_graph, current_cover = pile.pop()
        aretes = getListArete(current_graph)

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
        aretes = getListArete(current_graph)

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
        aretes = getListArete(current_graph)

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

#G = read_file("../instance/exemple_instance.txt")
#G = ([0, 1, 2, 3], [[1], [0, 2, 3], [1], [1]])
#print("Q1 :")
#C = branchement_ameliore_q1(G)
#print(C)

#print("\n-------------------------------------------------\nQ2 :")
#C2 = branchement_ameliore_q2(G)
#print(C2)