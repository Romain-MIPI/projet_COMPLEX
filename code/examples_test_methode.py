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

### Algorithmes de comparaison:
# 3.2
# compare_en_n(algos=[algo_couplage,algo_glouton],n_debut=10,n_fin=101,step=10,p_fixe=0.5)
# compare_en_p(algos=[algo_couplage,algo_glouton],p_step=10,n_fixe=100)

# 4.1.2
# compare_en_n(algos=[branchement],n_debut=3,n_fin=25,step=2,p_fixe=0.5)
# compare_en_p(algos=[branchement],p_step=np.sqrt(20),n_fixe=20)

# 4.2.2
# compare_en_n(algos=[branchement_borne],n_debut=3,n_fin=25,step=2,p_fixe=0.5)
# compare_en_p(algos=[branchement_borne],p_step=np.sqrt(20),n_fixe=20)

# 4.2.3 (facultatif)
# compare_en_p(algos=[algo_glouton,branchement_borne],p_step=np.sqrt(20),n_fixe=20)

# 4.3.1
# compare_en_n(algos=[branchement_ameliore_q1],n_debut=3,n_fin=25,step=2,p_fixe=0.5)
# compare_en_p(algos=[branchement_ameliore_q1],p_step=np.sqrt(20),n_fixe=20)

# 4.3.2
# compare_en_n(algos=[branchement_ameliore_q2],n_debut=3,n_fin=25,step=2,p_fixe=0.5)
# compare_en_p(algos=[branchement_ameliore_q2],p_step=np.sqrt(20),n_fixe=20)

# 4.4 rapport d'approximation pour l'algo couplage et glouton
# get_rapport_approximation([algo_couplage,algo_glouton],n_debut=3,n_fin=50,step=5,p_fixe=0.5)
# get_rapport_approximation([algo_couplage, algo_glouton],n_debut=3,n_fin=50,step=5,p_fixe=0.5)

# Comparer entre les branchements
# compare_en_n(algos=[branchement,branchement_borne,branchement_ameliore_q1,branchement_ameliore_q2],n_debut=3,n_fin=15,step=2,p_fixe=0.5)
# compare_en_p(algos=[branchement_borne,branchement_ameliore_q1,branchement_ameliore_q2],p_step=np.sqrt(20),n_fixe=20)