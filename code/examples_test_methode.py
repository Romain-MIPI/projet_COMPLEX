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
# methodes de comparaison
# compare_en_n()
# compare_en_p()

# 4. Branchement
# branchement_basique = branchement(G)
# print(f"{branchement_basique=}")
# branchement_bornee = branchement_borne(G)
# print(f"{branchement_bornee=}")
# for i in range(10):
#     g = generate_graphe(n=7, p=0.5)
#     branchement_basique = branchement(g)
#     branchement_bornee = branchement_borne(g)
#     if len(branchement_basique) != len(branchement_bornee):
#         print("ERREUR ici!!")
#         raise ValueError(f"value error: {branchement_basique=}, {branchement_bornee=} ")
    
g=([0, 1, 2, 3, 4, 5, 6], [[1, 2, 5, 6], [0, 3, 4], [0, 5], [1, 5], [1, 5], [0, 2, 3, 4, 6], [0, 5]])
print(branchement_borne(g))
# for i in range(10):
#     branchement_basique = branchement(g)
#     branchement_bornee = branchement_borne(g)
#     if len(branchement_basique) != len(branchement_bornee):
#         print("ERREUR ici!!")
#         raise ValueError(f"value error: {branchement_basique=}, {branchement_bornee=} ")