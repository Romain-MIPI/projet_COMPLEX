import sys
sys.path.append("./")
from methode import *
import time
import matplotlib.pyplot as plt


def write_file(t, fichier):
    f = open(fichier, "w")
    for i in range(len(t)):
        f.write("%d : {"%(i+2))
        for u in t[i]:
            f.write(" %d"%u)
        f.write(" }\n")
    f.close()


t_couplage = []
s_couplage = []
t_glouton = []
s_glouton = []

n = [i for i in range(2, 51)]

for i in n:
    G_gen = generate_graphe(i, 0.5)
    print("\n------------------------------------------------------------\n")
    print("i =%d\n"%i)

    print("algo_couplage:")

    debut_couplage = time.time()
    s = algo_couplage(G_gen)
    fin_couplage = time.time()

    t = fin_couplage-debut_couplage
    t_couplage.append(t)
    s_couplage.append(s)
    print("t = %f"%t)
    print("s =", s)

    print("\nalgo glouton:")

    debut_glouton = time.time()
    s = algo_glouton(G_gen)
    fin_glouton = time.time()

    t = fin_glouton-debut_glouton
    t_glouton.append(t)
    s_glouton.append(s)
    print("t = %f"%t)
    print("s =", s)

plt.plot(n, t_couplage)
plt.plot(n, t_glouton)
plt.xlabel("taille n")
plt.ylabel("temps en s")
plt.savefig('courbe_t.png')
plt.show() 

#write_file(s_couplage, "solutions_couplage.txt")
#write_file(s_glouton, "solutions_glouton.txt")

print(compare_algo(s_couplage, s_glouton))