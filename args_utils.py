import argparse
import sys
import os
import numpy as np


def get_args():

    parser = argparse.ArgumentParser(description='memetic and genetic algorithms for Global Multi-Objective Optimization')

    parser.add_argument('--dataset', type=str, help='dataset path (CSV file required)')

    parser.add_argument('--constraints', type=str, help='constraints path (JSON file required)')

    parser.add_argument('--K', type=int, help='number of clusters (if not provided, it is assumed by the labels of the dataset)', default=None)

    parser.add_argument('--seed', type=int, help='seed for the pseudo-random number generator (provide it in order to have reproducible results)', default=None)

    ####################################################
    ### S-MDEClust Parameters ###
    ####################################################

    parser.add_argument('--assignment', type=str, help='assignment option', default='exact', choices=['exact', 'greedy'])

    parser.add_argument('--mutation', help='use mutation operator', action='store_true', default=False)
    
    parser.add_argument('--P', type=int, help='size of the population', default=20)

    parser.add_argument('--Nmax', type=int, help='maximum number of consecutive iterations without improvement of the best solution', default=5000)

    parser.add_argument('--max_iter', type=int, help='maximum number of iterations', default=np.inf)

    parser.add_argument('--tol_pop', type=float, help='population tolerance', default=1e-4)

    parser.add_argument('--Nmax_ls', type=int, help='maximum number of consecutive iterations without improvement of the best solution for the local search', default=2)

    parser.add_argument('--max_iter_ls', type=int, help='maximum number of iterations for the local search', default=np.inf)

    parser.add_argument('--tol_sol', type=float, help='tolerance to choose if a solution is better than an another one', default=1e-6)

    parser.add_argument('--F', help='F paramater used for the crossover operator', default='mdeclust')  # Available options: random, mdeclust, float/integer value in (0, 2)

    parser.add_argument('--alpha', type=float, help='alpha paramater used for the mutation operator', default=0.5)

    return parser.parse_args(sys.argv[1:])


def check_args(args):
    
    assert os.path.exists(args.dataset)
    assert os.path.exists(args.constraints)

    if args.K is not None:
        assert args.K > 0

    if args.seed is not None:
        assert args.seed > 0

    assert args.P > 0
    assert args.Nmax > 0
    assert args.max_iter > 0
    assert args.tol_pop > 0
    assert args.Nmax_ls > 0
    assert args.max_iter_ls > 0
    assert args.tol_sol > 0

    if type(args.F) == str:
        assert args.F in ['random', 'mdeclust']
    elif type(args.F) == float or type(args.F) == int:
        assert args.F > 0
    else:
        raise AssertionError
    
    assert 0 <= args.alpha <= 1


def args_file_creation(res_path, args):
    args_file = open(os.path.join(res_path, 'params.csv'), 'w')
    
    for key in args.__dict__.keys():
        if type(args.__dict__[key]) == float:
            args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
        else:
            args_file.write('{};{}\n'.format(key, args.__dict__[key]))
    
    args_file.close()
