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
def branchement(G):
    """
    Utilise une méthode basique de branchement pour trouver la couverture de sommets minimale d'un graphe.
    
    Parameters:
    - G: un tuple composé de (liste de sommets, liste d'adjacence).
    
    Returns:
    - set: un ensemble de sommets représentant la couverture minimale.
    """
    print("branchement_basique!!!!!")
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        current_graph, current_cover = pile.pop()
        aretes = get_liste_aretes(current_graph)

        # Élagage si toutes les arêtes sont couvertes
        if len(aretes)==0:
            # Garde la couverture actuelle si elle est meilleure que la meilleure trouvée jusqu'à présent
            if len(current_cover) < len(best_cover):
                best_cover = current_cover
        else:
            u,v = aretes[0]
            # Exploration des noeuds enfants
            pile.append((delete_sommet(current_graph, u),current_cover | {u}))
            pile.append((delete_sommet(current_graph, v),current_cover | {v}))
    print("best_cover finale=", best_cover)
    return best_cover

def borne_inf(G_bis):
    """Calcule des valeurs de b1, b2 et b3 afin d'obtenir une borne inf pour le branchement."""
    sommets = G_bis[0]
    aretes = get_liste_aretes(G_bis)
    n = len(sommets)
    m = len(aretes)
    couplage = algo_couplage(G_bis)
    delta = max(degre_sommets(G))
    b1 = np.ceil(m/delta)
    b2 = len(couplage)//2
    b3 = ((2*n)-1-np.sqrt(((2*n-1)**2)-(8*m)))/2
    return max(b1,b2,b3)

def branchement_borne(G):
    print("branchement_borne!!!!!")
    # Initialisation : borne supérieure = sommets du graphe initial
    best_cover = G[0]
    b_sup = len(best_cover)

    # Initialisation de la pile
    pile = [(G, set())]
    while pile:
        print("pile =", pile)
        print("best_cover debut boucle=", best_cover)
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
    print("best_cover finale=", best_cover)
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

def plot_time_graph(n, temps):
    """
    Affiche des courbes représentant le temps d'exécution des algorithmes en fonction de la taille n.
    Fonction auxiliaire pour les méthodes "compare_en_n" et "compare_en_p".
    
    Paramètres:
    - n: Une liste indiquant le nombre de sommets des graphes.
    - temps: Un dictionnaire où les clés sont les noms des algorithmes et les valeurs sont des listes 
             représentant le temps d'exécution de cet algorithme pour chaque taille dans n.
    """
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '*']
    for index, (algo_name, execution_times) in enumerate(temps.items()):
        plt.plot(n, execution_times, label=algo_name, linestyle=line_styles[index], marker=markers[index], linewidth=1.5)
    plt.title("courbe de temps en fonction de n")
    plt.xlabel("taille n")
    plt.ylabel("temps en s")
    plt.legend()
    plt.savefig('courbe_t_n.png')
    plt.show()

def plot_solution_histogram(n, solutions):
    """
    Affiche un histogramme comparant la taille de couverture trouvé par plusieurs algorithmes en fonction de la taille n.
    Fonction auxiliaire pour les méthodes "compare_en_n" et compare_en_p".

    Paramètres:
    - n: Une liste indiquant le nombre de sommets des graphes.
    - solutions: Un dictionnaire où les clés sont les noms des algorithmes et les valeurs sont des listes 
                 représentant la couverture trouvée par cet algorithme pour chaque taille dans n.
    """
    x = np.arange(len(n))
    width = 0.35 / len(solutions) 
    fig, ax = plt.subplots()

    print(f"{solutions=}")
    for idx, (algo_name, cover_sizes) in enumerate(solutions.items()):
        ax.bar(x + idx*width, cover_sizes, width, label=algo_name)

    ax.set_xlabel('taille n')
    ax.set_ylabel('nombre de sommets')
    ax.set_title('nombre de sommets trouvé en fonction de n')
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.legend()
    plt.savefig("hist_n.png")
    plt.show()

def compare_en_n(algos, n_debut=10, n_fin=101, step=10, q_fixe=0.5):
    """
    Compare les temps d'exécution et les couvertures trouvées par une liste d'algorithmes sur des graphes de tailles différentes.

    Cette fonction génère des graphes aléatoires avec des tailles variant de `n_debut` à `n_fin` par étapes de `step`. 
    Pour chaque graphe, elle mesure le temps d'exécution et le nombre de sommets trouvés (couverture) pour chaque algorithme dans la liste `algos`.
    Ensuite, elle affiche deux graphiques montrant le temps d'exécution et le nombre de sommets de la couverture trouvée en fonction de la taille du graphe pour chaque algorithme.

    Paramètres:
    - algos: Liste des algorithmes à comparer.
    - n_debut: La taille initiale du graphe (valeur par défaut: 10).
    - n_fin: La taille finale du graphe (valeur par défaut: 101).
    - step: L'intervalle d'augmentation de la taille du graphe (valeur par défaut: 10).
    - q_fixe: Probabilité utilisée pour la génération aléatoire du graphe (valeur par défaut: 0.5).
    """
    temps = {algo.__name__: [] for algo in algos}
    solutions = {algo.__name__: [] for algo in algos}

    n = list(range(n_debut, n_fin, step))

    for i in n:
        temps_i = {algo.__name__: [] for algo in algos}
        solutions_i = {algo.__name__: [] for algo in algos}

        for _ in range(10):
            G_gen = generate_graphe(i, q_fixe)
            for algo in algos:
                t, s = measure_algo_time_and_solution(algo, G_gen)
                temps_i[algo.__name__].append(t)
                solutions_i[algo.__name__].append(s)

        for algo in algos:
            temps[algo.__name__].append(np.mean(temps_i[algo.__name__]))
            solutions[algo.__name__].append(np.mean(solutions_i[algo.__name__]))

    plot_time_graph(n,temps)
    plot_solution_histogram(n,solutions)

    return

def compare_en_p(algos, p_step=0.1, n_fixe=100):
    """
    Compare les temps d'exécution et les couvertures de sommets trouvées par différents algorithmes sur des graphes ayant différentes probabilités d'arêtes.

    Cette fonction génère des graphes aléatoires de taille `n_fixe` avec des probabilités variant de 0.1 à 1.0 en utilisant un pas défini par `p_step`. 
    Pour chaque graphe, elle évalue le temps d'exécution et le nombre de sommets couverts par chaque algorithme présent dans la liste `algos`.
    Ensuite, elle produit deux graphiques : 
        - Un montrant le temps d'exécution en fonction de la probabilité.
        - Un autre illustrant le nombre de sommets de la couverture obtenue en fonction de la probabilité.

    Paramètres:
    - algos : Liste des algorithmes à évaluer.
    - p_step : Pas d'incrémentation de la probabilité. Par exemple, un p_step de 0.1 donnera une séquence de probabilités [0.1, 0.2, ..., 1.0].
    - n_fixe : Taille fixe du graphe pour chaque évaluation.
    """
    temps = {algo.__name__: [] for algo in algos}
    solutions = {algo.__name__: [] for algo in algos}

    p_n = np.arange(0.1, 1.0 + p_step, p_step).tolist()

    for p in p_n:
        algo_times = {algo.__name__: [] for algo in algos}
        algo_solutions = {algo.__name__: [] for algo in algos}

        for _ in range(10):
            G_gen = generate_graphe(n_fixe, p)
            
            for algo in algos:
                t, s = measure_algo_time_and_solution(algo, G_gen)
                algo_times[algo.__name__].append(t)
                algo_solutions[algo.__name__].append(s)

        for algo in algos:
            temps[algo.__name__].append(np.mean(algo_times[algo.__name__]))
            solutions[algo.__name__].append(np.mean(algo_solutions[algo.__name__]))

    plot_time_graph(p_n, temps)
    plot_solution_histogram(p_n, solutions)

    #écriture des solutions dans un fichier
    #write_file(s_couplage, "solutions_couplage.txt")
    #write_file(s_glouton, "solutions_glouton.txt")

    #print(compare_algo(s_couplage, s_glouton))

    return

G = read_file("../instance/exemple_instance.txt")
# print(G)
#G = ([0, 1, 2, 3], [[1], [0, 2, 3], [1], [1]])
#print("Q1 :")
#C = branchement_ameliore_q1(G)
#print(C)

#print("\n-------------------------------------------------\nQ2 :")
#C2 = branchement_ameliore_q2(G)
#print(C2)