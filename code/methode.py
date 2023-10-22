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
             On accepte de solution retournée avec des sommets sans arete,
             i.e. [0, 3, 4], [[3], [0], []] avec 4 un sommet sans arete.
    """
    sommets, adjacences = list(G[0]), [list(l) for l in G[1]] 
    for l in adjacences:
        if v in l:
            l.remove(v)

    adjacences.pop(sommets.index(v))
    sommets.remove(v)
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
    Calcule une couverture a l'aide de l'algorithme de couplage.(exo: 3.2)

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
    Calcule une couverture a l'aide de l'algorithme glouton.(exo: 3.2)

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
    compteur_noeud = 0

    # Initialisation de la pile
    pile = [(G, set())]
    compteur_noeud += 1
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
            compteur_noeud += 2
    return best_cover, compteur_noeud

def branchement_borne(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)
    compteur_noeud = 0

    # Initialisation de la pile
    pile = [(G, set())]
    compteur_noeud += 1
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
            compteur_noeud += 2
    return best_cover, compteur_noeud

def branchement_ameliore_q1(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)
    compteur_noeud = 0

    # Initialisation de la pile
    pile = [(G, set())]
    compteur_noeud += 1
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
            compteur_noeud += 2
    return best_cover, compteur_noeud

def branchement_ameliore_q2(G):
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)
    compteur_noeud = 0

    # Initialisation de la pile
    pile = [(G, set())]
    compteur_noeud += 1
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
            compteur_noeud += 2
    return best_cover, compteur_noeud

def write_file(t, fichier):
    f = open(fichier, "w")
    for i in range(len(t)):
        f.write("%d : {"%(i+2))
        for u in t[i]:
            f.write(" %d"%u)
        f.write(" }\n")
    f.close()

######### Methodes pour la comparaison de l'algorithme couplage et glouton

def measure_algo_time_and_solution(algo, G_gen):
    """
    Mesure le temps d'exécution d'un algorithme et la taille de la solution obtenue.
    Fonction auxiliaire pour les méthodes "compare_en_n" et compare_en_p".

    Paramètres:
    - algo: Une fonction représentant l'algorithme à évaluer.
    - G_gen: Le graphe sur lequel l'algorithme doit être exécuté.

    Retourne:
    - Le temps d'exécution de l'algorithme (en secondes).
    - La taille de la solution produite par l'algorithme.
    """
    start_time = time.time()
    solution = algo(G_gen)
    end_time = time.time()
    return end_time - start_time, len(solution)

def plot_time_graph(n, temps_couplage, temps_glouton):
    """
    Affiche deux courbe représentant le temps d'exécution des deux algorithmes en fonction de la taille n.
    Fonction auxiliaire pour les méthodes "compare_en_n" et compare_en_p".
    
    Paramètres:
    - n: Une liste indiquant le nombre de sommets des graphes.
    - temps_couplage: Une liste représentant le temps d'exécution de l'algo_couplage pour chaque taille dans n.
    - temps_glouton: Une liste représentant le temps d'exécution de l'algo_glouton pour chaque taille dans n.
    """
    plt.plot(n, temps_couplage, label="algo_couplage")
    plt.plot(n, temps_glouton, label="algo_glouton")
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend()
    plt.savefig('courbe_t_n.png')
    plt.show()

def plot_solution_histogram(n, solutions_couplage, solutions_glouton):
    """
    Affiche un histogramme comparant la taille de couverture trouvé par deux algorithmes en fonction de la taille n.
    Fonction auxiliaire pour les méthodes "compare_en_n" et compare_en_p".

    Paramètres:
    - n: Une liste indiquant le nombre de sommets des graphes.
    - solutions_couplage: Une liste représentant la couverture trouvée par l'algo_couplage pour chaque taille dans n.
    """
    x = np.arange(len(n))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, solutions_couplage, width, label='solutions couplage')
    ax.bar(x + width/2, solutions_glouton, width, label='solutuon glouton')
    ax.set_xlabel('taille n')
    ax.set_ylabel('nombre de sommets')
    ax.set_title('nombre de sommets trouvé en fonction de n')
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.legend()
    plt.savefig("hist_n.png")
    plt.show()

def compare_en_n(n_debut=10, n_fin=101, step=10):
    """
    Compare les temps d'exécution et les couvertures trouvées par deux algorithmes sur des graphes de tailles différentes.

    Cette fonction génère des graphes aléatoires avec des tailles variant de `n_debut` à `n_fin` par étapes de `step`. 
    Pour chaque graphe, elle mesure le temps d'exécution et le nombre de sommets trouvés(couverture) par `algo_couplage` et `algo_glouton`.
    Ensuite, elle affiche deux graphiques: un montrant le temps d'exécution en fonction de la taille n du graphe 
    et un autre montrant le nombre de sommets de la couverture trouvée en fonction de la taille du graphe.

    Paramètres:
    - n_debut: La taille initiale du graphe (valeur par défaut: 10).
    - n_fin: La taille finale du graphe (valeur par défaut: 101).
    - step: L'intervalle d'augmentation de la taille du graphe (valeur par défaut: 10).
    """
    temps_couplage,temps_glouton = [], []
    solutions_couplage, solutions_glouton = [], []

    n = list(range(n_debut, n_fin, step))

    for i in n:
        tc_i, tg_i, sc_i, sg_i = [], [], [], []

        for _ in range(10):
            G_gen = generate_graphe(i, 0.5)
            
            t, s = measure_algo_time_and_solution(algo_couplage, G_gen)
            tc_i.append(t)
            sc_i.append(s)

            t, s = measure_algo_time_and_solution(algo_glouton, G_gen)
            tg_i.append(t)
            sg_i.append(s)

        temps_couplage.append(np.mean(tc_i))
        temps_glouton.append(np.mean(tg_i))
        solutions_couplage.append(np.mean(sc_i))
        solutions_glouton.append(np.mean(sg_i))

    plot_time_graph(n,temps_couplage,temps_glouton)
    plot_solution_histogram(n,solutions_couplage,solutions_glouton)

    return

def compare_en_p():
    """
    Compare les temps d'exécution et les couvertures trouvées par deux algorithmes sur des graphes avec différentes probabilités.

    Cette fonction génère des graphes aléatoires de taille fixe (100) avec des probabilités variant de 0.1 à 1.0 par pas de 0.1. 
    Pour chaque graphe, elle mesure le temps d'exécution et le nombre de sommets trouvés par `algo_couplage` et `algo_glouton`.
    Ensuite, elle affiche deux graphiques: un montrant le temps d'exécution en fonction de la probabilité
    et un autre montrant le nombre de sommets trouvés pour la couverture en fonction de la probabilité.
    """
    temps_couplage,temps_glouton = [], []
    solutions_couplage, solutions_glouton = [], []

    p_n = [i/10 for i in range(1, 11)]

    for p in p_n:
        tc_i, tg_i, sc_i, sg_i = [], [], [], []

        for _ in range(10):
            G_gen = generate_graphe(100, p)

            t, s = measure_algo_time_and_solution(algo_couplage, G_gen)
            tc_i.append(t)
            sc_i.append(s)

            t, s = measure_algo_time_and_solution(algo_glouton, G_gen)
            tg_i.append(t)
            sg_i.append(s)

        temps_couplage.append(np.mean(tc_i))
        temps_glouton.append(np.mean(tg_i))
        solutions_couplage.append(np.mean(sc_i))
        solutions_glouton.append(np.mean(sg_i))

    plot_time_graph(p_n,temps_couplage,temps_glouton)
    plot_solution_histogram(p_n,solutions_couplage,solutions_glouton)

    #écriture des solutions dans un fichier
    #write_file(s_couplage, "solutions_couplage.txt")
    #write_file(s_glouton, "solutions_glouton.txt")

    #print(compare_algo(s_couplage, s_glouton))

    return

def test_branchement_en_n():
    solutions = []
    temps = []
    noeud = []

    n = [i for i in range(10, 21, 2)]

    for i in n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(i, 1/np.sqrt(20))

            debut_couplage = time.time()
            s, c_noeud = branchement(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(n, temps)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_n.png')
    plt.show()

    # affichage noued crée/n
    plt.plot(n, noeud)
    plt.title("courbe de nombre de noeuds crées en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_noeud_n.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_en_p():
    solutions = []
    temps = []
    noeud = []

    p_n = [i/np.sqrt(20) for i in range(1, int(np.ceil(np.sqrt(20))))]

    for p in p_n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(20, p)

            debut_couplage = time.time()
            s, c_noeud = branchement(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(p_n, temps)
    plt.title("courbe de temps en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_p.png')
    plt.show()

    # affichage t/n
    plt.plot(p_n, noeud)
    plt.title("courbe de nombre de noeud crée en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_noeud_p.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_borne_en_n():
    solutions = []
    temps = []
    noeud = []

    n = [i for i in range(10, 21, 2)]

    for i in n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(i, 1/np.sqrt(20))

            debut_couplage = time.time()
            s, c_noeud = branchement_borne(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(n, temps)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_borne_n.png')
    plt.show()

    # affichage noued crée/n
    plt.plot(n, noeud)
    plt.title("courbe de nombre de noeuds crées en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_borne_noeud_n.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_borne_en_p():
    solutions = []
    temps = []
    noeud = []

    p_n = [i/np.sqrt(20) for i in range(1, int(np.ceil(np.sqrt(20))))]

    for p in p_n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(20, p)

            debut_couplage = time.time()
            s, c_noeud = branchement_borne(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(p_n, temps)
    plt.title("courbe de temps en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_borne_p.png')
    plt.show()

    # affichage t/n
    plt.plot(p_n, noeud)
    plt.title("courbe de nombre de noeud crée en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_borne_noeud_p.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_ameliore_q1_en_n():
    solutions = []
    temps = []
    noeud = []

    n = [i for i in range(10, 21, 2)]

    for i in n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(i, 1/np.sqrt(20))

            debut_couplage = time.time()
            s, c_noeud = branchement_ameliore_q1(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(n, temps)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q1_n.png')
    plt.show()

    # affichage noued crée/n
    plt.plot(n, noeud)
    plt.title("courbe de nombre de noeuds crées en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q1_noeud_n.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_ameliore_q1_en_p():
    solutions = []
    temps = []
    noeud = []

    p_n = [i/np.sqrt(20) for i in range(1, int(np.ceil(np.sqrt(20))))]

    for p in p_n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(20, p)

            debut_couplage = time.time()
            s, c_noeud = branchement_ameliore_q1(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(p_n, temps)
    plt.title("courbe de temps en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q1_p.png')
    plt.show()

    # affichage t/n
    plt.plot(p_n, noeud)
    plt.title("courbe de nombre de noeud crée en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q1_noeud_p.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_ameliore_q2_en_n():
    solutions = []
    temps = []
    noeud = []

    n = [i for i in range(10, 21, 2)]

    for i in n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(i, 1/np.sqrt(20))

            debut_couplage = time.time()
            s, c_noeud = branchement_ameliore_q2(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(n, temps)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q2_n.png')
    plt.show()

    # affichage noued crée/n
    plt.plot(n, noeud)
    plt.title("courbe de nombre de noeuds crées en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q2_noeud_n.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return

def test_branchement_bameliore_q2_en_p():
    solutions = []
    temps = []
    noeud = []

    p_n = [i/np.sqrt(20) for i in range(1, int(np.ceil(np.sqrt(20))))]

    for p in p_n:
        t_i = []
        s_i = []
        n_i = []
        
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        for j in range(10):
            G_gen = generate_graphe(20, p)

            debut_couplage = time.time()
            s, c_noeud = branchement_ameliore_q2(G_gen)
            fin_couplage = time.time()

            t = fin_couplage-debut_couplage
            t_i.append(t)
            s_i.append(len(s))
            n_i.append(c_noeud)
            #print("t = %f"%t)
            #print("s =", s)

        temps.append(np.mean(t_i))
        solutions.append(np.mean(s_i))
        noeud.append(np.mean(n_i))

    # affichage t/n
    plt.plot(p_n, temps)
    plt.title("courbe de temps en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("temps en s")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q2_p.png')
    plt.show()

    # affichage t/n
    plt.plot(p_n, noeud)
    plt.title("courbe de nombre de noeud crée en fonction de p")
    plt.xlabel("probabilité p")
    plt.ylabel("nombre de noeuds créés")
    plt.legend(["branchement"])
    plt.savefig('courbe_branchement_ameliore_q2_noeud_p.png')
    plt.show()

    #écriture des solutions dans un fichier
    #write_file(s, "solutions_branchement.txt")

    return