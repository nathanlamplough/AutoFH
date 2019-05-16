import numpy as np
from bayes_opt import BayesianOptimization
import DatabaseFunctions as df
import LandmarkMetaFunctions as lmf
import MetaFunctions as mf
import FeatureSelectionFunctions as fs
import pandas as pd
import sys
import Network as n
import torch
from sklearn import preprocessing

fsa_dict = []
y = []

def AutoFH(X, y_, iter, norm):

    init = 10

    dfx = pd.DataFrame(X)

    df = dfx

    df['y'] = y_

    data = (X, y_, dfx)

    d_in = X.shape[1]

    if norm:
        X = normalise(X)

    d_out = np.unique(y_).shape[0]

    x = X
    global fsa_dict
    fsa_dict = fs.select_features(df, x, y_)

    hyperparameters = Get_start_parameters(data)

    if d_out == 2:
        y_ = to_one_hot(y_, d_out)

    global y
    y = y_

    return Bayesian_optimisation(hyperparameters, iter, init, d_in, d_out)

def normalise(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    return X

def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[int(i), int(label)] = int(1)
    return results

def Run_Network(l_rate, dropout, epochs, momentum, layers, nodes, l_reg, reg_lambda, FSA):
    FSA = int(FSA)
    l_reg = int(l_reg)
    epochs = int(epochs)
    layers = int(layers)
    nodes = int(nodes)

    global y
    global fsa_dict
    X_ = fsa_dict[FSA]

    d_in = X_.shape[1]
    d_out = np.unique(y).shape[0]

    model = n.Net(d_in, nodes, d_out, layers, dropout)

    X_ = torch.tensor(X_, dtype=torch.float)
    if d_out == 2:
        Y = torch.tensor(y, dtype=torch.float)
    else:
        Y = torch.tensor(y, dtype=torch.long)

    f1 = n.run_network(epochs, X_, Y, model, l_rate, 5, l_reg, reg_lambda, momentum, d_out)
    return f1

def Bayesian_optimisation(start_parameters, iter, init, d_in, d_out):
    pbounds = {'l_rate': (0.0001, 0.1), 'dropout': (0, 1), 'epochs': (1, 1500), 'momentum': (0, 1), 'layers': (0, 5),
               'nodes': (d_out, d_in), 'l_reg': (0, 2.99), 'reg_lambda': (0, 1), 'FSA': (0, 3.99)}
    optimizer = BayesianOptimization(
        f=Run_Network,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'l_rate': start_parameters[0], 'dropout': start_parameters[1], 'epochs': int(start_parameters[2]),
                'momentum': start_parameters[3], 'layers': start_parameters[4], 'nodes': start_parameters[5],
                'l_reg': start_parameters[6], 'reg_lambda': start_parameters[7], 'FSA': 0},
        lazy=True,
    )

    optimizer.maximize(
        init_points=init,
        n_iter=iter,
    )

    scores = np.empty(0)
    max_score = 0
    best_params = np.empty(0)
    for i, res in enumerate(optimizer.res):
         scores = np.append(scores,res['target'])
         if res['target'] > max_score:
             max_score = res['target']
             best_params = np.empty(0)
             best_params = np.append(best_params, res['params'])
         elif res['target'] == max_score:
             best_params = np.append(best_params, res['params'])


    return best_params

def Get_start_parameters(data):
    l_meta_features = lmf.compute_l_meta_features(data, False)

    db_mf = df.select_all()
    l_mf_float = to_float(db_mf)

    closest = Get_Closest(l_meta_features,l_mf_float)
    test_string = db_mf[closest].hyperparameters.split(' ')
    if len(test_string) > 8:
        test_string.pop(8)
    hyp = np.asarray(test_string).astype(np.float)
    return hyp

def to_float(meta_features):
    mf_float = []
    for i in meta_features:
        split = np.asarray(i.l_meta_features.split(' '))
        mf_float.append(split.astype(np.float))
    return np.asarray(mf_float)

def m_to_float(meta_features):
    mf_float = []
    for i in meta_features:
        split = np.asarray(i.meta_features.split(' '))
        mf_float.append(split.astype(np.float))
    return np.asarray(mf_float)


def Get_Closest(mf, db_mf):
    lowest = sys.maxsize
    index = 0
    for i in range(db_mf.shape[0]):
        distance = np.linalg.norm(mf - db_mf[i])
        if(distance < lowest):
            lowest = distance
            index = i
    return index



