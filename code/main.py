import sys
sys.path.append("./")
from methode import *
import time
import matplotlib.pyplot as plt
import numpy as np


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
x = np.arange(len(n))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, s_couplage, width, label='solutions couplage')
rects2 = ax.bar(x + width/2, s_glouton, width, label='solutuon glouton')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('nombre de sommets des solutions')
ax.set_title('nombre de sommets genere')
ax.set_xticks(x)
ax.set_xticklabels(n)
ax.legend()
# plt.hist(s_couplage, bins=range(0,len(n)))
# plt.hist(s_glouton, bins=range(0,len(n)))
plt.show()

#write_file(s_couplage, "solutions_couplage.txt")
#write_file(s_glouton, "solutions_glouton.txt")

# print(compare_algo(s_couplage, s_glouton))