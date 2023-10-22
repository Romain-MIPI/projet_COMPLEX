import sys
sys.path.append("./")
from methode import *

# compare_en_p()
compare_en_n(algos=[branchement,branchement_borne,branchement_ameliore_q1],n_debut=1,n_fin=25,step=5)