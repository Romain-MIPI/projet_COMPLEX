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

n = [i for i in range(1, 101, 10)]

for i in n:
    #tc_i = []
    #tg_i = []
    sc_i = []
    sg_i = []
    for j in range(10):
        G_gen = generate_graphe(i, 0.5)
        #print("\n------------------------------------------------------------\n")
        #print("i =%d\n"%i)

        #print("algo_couplage:")

        debut_couplage = time.time()
        s = algo_couplage(G_gen)
        fin_couplage = time.time()

        #t = fin_couplage-debut_couplage
        #tc_i.append(t)
        sc_i.append(len(s))
        #print("t = %f"%t)
        #print("s =", s)

        #print("\nalgo glouton:")

        debut_glouton = time.time()
        s = algo_glouton(G_gen)
        fin_glouton = time.time()

        #t = fin_glouton-debut_glouton
        #tg_i.append(t)
        sg_i.append(len(s))
        #print("t = %f"%t)
        #print("s =", s)
    s_couplage.append(np.mean(sc_i))
    s_glouton.append(np.mean(sg_i))


# affichage t/n
#plt.plot(n, t_couplage)
#plt.plot(n, t_glouton)
#plt.xlabel("taille n")
#plt.ylabel("temps en s")
#plt.savefig('courbe_t.png')
#plt.show() 

# affichage diff/n
plt.hist(s_couplage, bins=len(n))
plt.hist(s_glouton, bins=len(n))
plt.show()

#write_file(s_couplage, "solutions_couplage.txt")
#write_file(s_glouton, "solutions_glouton.txt")

print(compare_algo(s_couplage, s_glouton))