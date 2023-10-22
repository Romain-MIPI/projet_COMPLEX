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
