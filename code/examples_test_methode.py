import sys
sys.path.append("./")
from methode import *

# 1. Read File
G = read_file("../instance/exemple_instance.txt")
print(f"{G=}")

# 2. Graphe
G_sup_sommet = delete_sommet(G,v=1)
print(f"{G_sup_sommet=}")

ensemble_sommet = [1,2]
G_sup_ensemble_sommet = delete_ens_sommet(G,V=ensemble_sommet)
print(f"{G_sup_ensemble_sommet=}")

liste_degre_sommets = degre_sommets(G)
print(f"{liste_degre_sommets=}")

sommet_degre_max = sommet_degree_max(G)
print(f"{sommet_degre_max=}")

G_generated = generate_graphe(5,0.5)
print(f"{G_generated=}")

# 3. Méthodes approchées
print(f"Pour les tests suivant, on utilise toujours le graphe G de l'exemple: \n {G=}")
liste_aretes = get_liste_aretes(G) # Test de méthode auxiliaire
print(f"{liste_aretes=}")

couverture_couplage = algo_couplage(G)
print(f"{couverture_couplage=}")

couverture_glouton = algo_glouton(G)
print(f"{couverture_glouton=}")

# 4. Séparation et évaluation

branchement_basique = branchement(G)
print(f"{branchement_basique=}")

branchement_bornee = branchement_borne(G)
print(f"{branchement_borne=}")

branchement_ameliore_1 = branchement_ameliore_q1(G)
print(f"{branchement_ameliore_1=}")

branchement_ameliore_2 = branchement_ameliore_q2(G)
print(f"{branchement_ameliore_2}")