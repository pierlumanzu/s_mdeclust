import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from gurobipy import Model, GRB

from blpkm import BLPKM


class S_MDEClust:

    def __init__(self, 
                 assignment, mutation, 
                 P, Nmax, max_iter, tol_pop, 
                 Nmax_ls, max_iter_ls, 
                 tol_sol,
                 F, alpha,
                 verbose):
        
        self.__assignment = assignment
        self.__mutation = mutation
        self.__P = P
        self.__Nmax = Nmax
        self.__max_iter = max_iter
        self.__tol_pop = tol_pop
        self.__tol_sol = tol_sol
        self.__F = F
        self.__alpha = alpha
        self.__verbose = verbose

        self.__ls = BLPKM(Nmax_ls, max_iter_ls)

    def initialize_population(self, D, K, ML, CL, start_time):
        N, d = D.shape

        phi = np.empty((self.__P, N), dtype=int)
        psi = np.empty((self.__P, K, d), dtype=float)
        scores = np.inf * np.ones(self.__P, dtype=float)
        n_iter_ls = 0

        for n_p in range(self.__P):
            center_ids = np.random.choice(np.arange(D.shape[0]), size=K, replace=False)
            centers = D[center_ids, :]

            phi[n_p, :], psi[n_p, :, :], scores[n_p], n_iter = self.__ls.run(D, centers, K, ML, CL)

            n_iter_ls += n_iter

            if self.__verbose:
                print('||' + str(-1).rjust(20) + ' |' + str(n_p+1).rjust(20) + ' |' + str(round(np.min(scores), 3)).rjust(20) + ' |' + 'N/A'.rjust(20) + ' |' + 'N/A'.rjust(20) + ' |' + str(n_p+1).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

        return phi, psi, scores, np.argmin(scores), n_iter_ls
    
    @staticmethod
    def population_diversity(scores):
        div = 0
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                div = div + abs(scores[i] - scores[j])
        return div

    @staticmethod
    def exact_matching(psi1, psi2):
        W = np.linalg.norm(psi1[:, None, :]-psi2[None, :, :], axis=2)**2

        row_indices, col_indices = linear_sum_assignment(W)

        new_psi1 = np.zeros(psi1.shape)
        new_psi1[col_indices] = psi1[row_indices]

        return new_psi1

    def crossover(self, psi1, psi2, psi3):
        
        if self.__F == 'random':
            return psi1 + (1e-7 + np.random.rand() * (2-2e-7)) * (psi2 - psi3)
        
        elif self.__F == 'mdeclust':
            return psi1 + (0.5 + np.random.rand() * 0.3) * (psi2 - psi3)
        
        elif type(self.__F) == float or type(self.__F) == int:
            return psi1 + self.__F * (psi2 - psi3) 
        
        else:
            raise AssertionError

    def assign_objects_excluding_c(self, D, centers, ML, CL, idx_c, ML_groups, CL_groups):
        
        if self.__assignment == 'exact':
            n = D.shape[0]
            k = centers.shape[0]
            distances = cdist(D, centers)
            assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

            m = Model()

            m.setParam("OutputFlag", False)

            y = m.addVars(assignments, obj=assignments, vtype=GRB.BINARY)

            m.addConstrs(y.sum(i, '*') == 1 for i in range(n))
            m.addConstrs(y.sum('*', j) >= 1 if j != idx_c else y.sum('*', j) == 0 for j in range(k))
            m.addConstrs(y[i, j] == y[i_, j] for j in range(k) for i, i_ in ML)
            m.addConstrs(y[i, j] + y[i_, j] <= 1 for j in range(k) for i, i_ in CL)

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])
                return labels, True
            else:
                return np.empty(0), False
            
        else:
            labels = -1 * np.ones(len(D), dtype=int)
            groups_to_centers = []
            
            for k in range(len(centers)):
                groups_to_centers.append([])
            
            for idx_ml_gr, ml_gr in enumerate(ML_groups):
                possible_ks = np.array([k for k in range(len(centers)) if len(set(groups_to_centers[k]) & CL_groups[idx_ml_gr]) == 0])
                
                if len(possible_ks) != 0:
                    ml_gr_k = np.argmin(np.array([np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[pk], axis=1)**2) for pk in possible_ks]))
                else:
                    ml_gr_k = np.argmin(np.array([np.sum(np.linalg.norm(D[np.array(list(ml_gr))] - centers[pk], axis=1)**2) for pk in range(len(centers))]))
                
                groups_to_centers[ml_gr_k].append(idx_ml_gr)
                labels[np.array(list(ml_gr))] = ml_gr_k
                
            return labels, True

    @staticmethod
    def calculate_probs(D, phi, psi, alpha):
        if alpha != 0:
            d = np.linalg.norm(D - psi[phi], axis=1)
            return (((1 - alpha) / len(D)) * np.ones(len(D))) + (alpha * (d / np.sum(d)))
        else:
            return (1 / len(D)) * np.ones(len(D))
    
    def mutation(self, D, psiO, ML, CL, ML_groups, CL_groups):
        idx_removed_c = np.random.randint(len(psiO))
        
        tmp_phi, success = self.assign_objects_excluding_c(D, psiO, ML, CL, idx_removed_c, ML_groups, CL_groups)
        
        probs = self.calculate_probs(D, tmp_phi, psiO, self.__alpha if success else 0)
        idx_new_center = np.random.choice(np.arange(len(D)), p=probs)

        psiO[idx_removed_c] = np.copy(D[idx_new_center])
        
        return psiO

    def run(self, D, K, ML, CL, seed, ML_groups, CL_groups):

        start_time = time.time()

        if self.__verbose:
            print('||' + 'N°iter'.rjust(20) + ' |' + 'Sol'.rjust(20) + ' |' + 'f*'.rjust(20) + ' |' + 'N°w/oImprBest'.rjust(20) + ' |' + 'Pop_tol'.rjust(20) + ' |' + 'N°ls'.rjust(20) + ' |' + 'N°iter_ls'.rjust(20) + ' |' + 'time'.rjust(20) + ' ||')
    
        np.random.seed(seed)
        
        phi, psi, scores, best_s_idx, n_iter_ls = self.initialize_population(D, K, ML, CL, start_time)

        n_iter = 0
        n_cons_it_wo_impr = 0

        if self.__verbose:
            print('||' + str(n_iter).rjust(20) + ' |' + str(0).rjust(20) + ' |' + str(round(scores[best_s_idx], 3)).rjust(20) + ' |' + str(n_cons_it_wo_impr).rjust(20) + ' |' + str(round(self.population_diversity(scores), 3)).rjust(20) + ' |' + str(len(scores)).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

        while n_cons_it_wo_impr < self.__Nmax and n_iter < self.__max_iter:

            if self.population_diversity(scores) < self.__tol_pop:
                break

            for s in range(self.__P):

                s1, s2, s3 = np.random.choice(np.arange(self.__P), size=3, replace=False, p=np.array([1/(self.__P - 1) if i != s else 0 for i in range(self.__P)]))

                psi1 = self.exact_matching(psi[s1], psi[s3])
                psi2 = self.exact_matching(psi[s2], psi[s3])
                psiO = self.crossover(psi1, psi2, psi[s3])

                if self.__mutation and np.random.rand() < 1/(n_iter + 1):
                    psiO = self.mutation(D, psiO, ML, CL, ML_groups, CL_groups)

                phiO, psiO, scoreO, add_n_iter_ls = self.__ls.run(D, psiO, K, ML, CL)
                n_iter_ls += add_n_iter_ls

                if scores[s] - scoreO >= self.__tol_sol:
                    
                    phi[s] = phiO
                    psi[s] = psiO
                    scores[s] = scoreO
                    
                    if s == best_s_idx or scores[best_s_idx] - scores[s] >= self.__tol_sol:
                        best_s_idx = s
                        n_cons_it_wo_impr = 0
                    else:
                        n_cons_it_wo_impr += 1
                
                else:
                    n_cons_it_wo_impr += 1

                if self.__verbose:
                    print('||' + str(n_iter).rjust(20) + ' |' + str(s+1).rjust(20) + ' |' + str(round(scores[best_s_idx], 3)).rjust(20) + ' |' + str(n_cons_it_wo_impr).rjust(20) + ' |' + str(round(self.population_diversity(scores), 3)).rjust(20) + ' |' + str(n_iter * len(scores) + len(scores) + s + 1).rjust(20) + ' |' + str(n_iter_ls).rjust(20) + ' |' + str(round(time.time() - start_time, 3)).rjust(20) + ' ||')

            n_iter += 1
            
        return phi[best_s_idx], psi[best_s_idx], scores[best_s_idx], n_iter, n_iter * len(scores) + len(scores), n_iter_ls, time.time() - start_time, self.population_diversity(scores) < self.__tol_pop
