# Code adapted from the original version in https://github.com/phil85/BLPKM-CC
# © 2021, Universität Bern, Chair of Quantitative Methods, Philipp Baumann


from scipy.spatial.distance import cdist
import gurobipy as gb
import numpy as np


class BLPKM:

    def __init__(self, Nmax, max_iter):
        self.__Nmax = Nmax
        self.__max_iter = max_iter

    @staticmethod
    def assign_objects(D, centers, ML, CL):

        n = D.shape[0]
        k = centers.shape[0]
        distances = cdist(D, centers)
        assignments = {(i, j): distances[i, j] for i in range(n) for j in range(k)}

        m = gb.Model()

        m.setParam("OutputFlag", False)

        y = m.addVars(assignments, obj=assignments, vtype=gb.GRB.BINARY)

        m.addConstrs(y.sum(i, '*') == 1 for i in range(n))
        m.addConstrs(y.sum('*', j) >= 1 for j in range(k))
        m.addConstrs(y[i, j] == y[i_, j] for j in range(k) for i, i_ in ML)
        m.addConstrs(y[i, j] + y[i_, j] <= 1 for j in range(k) for i, i_ in CL)

        m.optimize()

        labels = np.array([j for i, j in y.keys() if y[i, j].X > 0.5])

        return labels
    
    @staticmethod
    def update_centers(D, centers, K, labels):
        for i in range(K):
            centers[i] = D[labels == i, :].mean(axis=0)
        return centers
    
    @staticmethod
    def get_total_distance(D, centers, labels):
        dist = np.sqrt(((D - centers[labels, :]) ** 2).sum(axis=1)).sum()
        return dist
    
    def run(self, D, centers, K, ML, CL):

        labels = self.assign_objects(D, centers, ML, CL)
        best_labels = labels

        centers = self.update_centers(D, centers, K, labels)
        best_centers = centers

        best_total_distance = self.get_total_distance(D, centers, labels)

        n_iter = 1
        n_cons_it_wo_impr = 0

        while n_iter < self.__max_iter:

            # print(n_iter)

            labels = self.assign_objects(D, centers, ML, CL)

            centers = self.update_centers(D, centers, K, labels)

            total_distance = self.get_total_distance(D, centers, labels)

            if total_distance >= best_total_distance:
                n_cons_it_wo_impr += 1
                if n_cons_it_wo_impr >= self.__Nmax:
                    n_iter += 1
                    break
            else:
                n_cons_it_wo_impr = 0

                best_labels = labels
                best_centers = centers
                best_total_distance = total_distance

            n_iter += 1

        return best_labels, best_centers, best_total_distance, n_iter
