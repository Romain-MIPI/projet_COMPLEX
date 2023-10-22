import sys
sys.path.append("./")
from methode import *

# compare_en_n(algos=[branchement_borne,branchement_ameliore_q1,branchement_ameliore_q2],n_debut=10,n_fin=21,step=2,q_fixe=1/np.sqrt(20))
compare_en_p(algos=[branchement_ameliore_q1],p_step=15,n_fixe=15)