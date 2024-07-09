import json
import os
import pandas as pd
import pickle as pkl
from datetime import datetime

from args_utils import get_args, check_args, args_file_creation
from utils import create_groups
from s_mdeclust import S_MDEClust


args = get_args()
check_args(args)

date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

res_path = os.path.join('Results', date, str(args.seed) if args.seed is not None else 'Random', args.dataset.split('/')[-1].split('.')[0])
os.makedirs(res_path, exist_ok=True)
args_file_creation(res_path, args)

dataset = pd.read_csv(args.dataset, sep=',')
D = dataset.values[:, :-1]

if args.K is not None:
    K = args.K
else:
    K = int(max(dataset.values[:, -1]) + 1)

f_json = json.load(open(args.constraints, 'r'))

ML = [tuple(i) for i in f_json['ml']]
CL = [tuple(i) for i in f_json['cl']]

ML_groups, CL_groups = create_groups(D, ML, CL)

s_mde = S_MDEClust(
    args.assignment, args.mutation,
    args.P, args.Nmax, args.max_iter, args.tol_pop,
    args.Nmax_ls, args.max_iter_ls,
    args.tol_sol,
    args.F, args.alpha,
    args.verbose
)
labels, centers, score, n_iter, n_ls, n_iter_ls, elapsed_time, is_pop_collapsed = s_mde.run(D, K, ML, CL, args.seed, ML_groups, CL_groups)

pkl.dump({
    'labels': labels, 'centers': centers, 'score': score,
    'n_iter': n_iter, 'n_ls': n_ls, 'n_iter_ls': n_iter_ls, 'elapsed_time': elapsed_time,
    'is_pop_collapsed': is_pop_collapsed
}, open(os.path.join(res_path, '{}.pkl'.format(args.constraints.split('/')[-1].split('.')[0])), 'wb'))
