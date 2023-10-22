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
