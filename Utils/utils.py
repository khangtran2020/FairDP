import pickle
import random
import os
import numpy as np
import torch
from sklearn.metrics import log_loss

def get_gaussian_noise(clipping_val, noise_scale):
    return noise_scale * clipping_val


def bound(args):
    if args.mode == 'onebatch':
        return 2 * args.lr * args.clip * np.sqrt(
            args.num_params) * (1 + args.ns * np.sqrt(
            -1 * np.log(1 - args.confidence) * (1 / (args.bs_male ** 2) + 1 / (args.bs_female ** 2))))
    else:
        return args.n_batch * (2 * args.lr * args.clip * np.sqrt(
            args.num_params) * (1 + args.ns * np.sqrt(
            -1 * np.log(1 - args.confidence) * (1 / (args.bs_male ** 2) + 1 / (args.bs_female ** 2)))))


def bound_alg1(args):
    return args.epochs * args.lr * (2 * args.clip + 8 * args.clip ** 2 * args.ns ** 2) / 2


def save_res(fold, args, dct, current_time):
    name = get_name(args=args,current_date=current_time, fold=fold)
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logloss(y, pred):
    return log_loss(y_true=y, y_pred=pred)


def get_coefficient(X, y, epsilon=None, lbda=None, mode='torch'):
    num_data_point = X.shape[0]
    num_feat = X.shape[1]
    sensitivity = num_feat**2/4 + 3*num_feat
    lbda = sensitivity*4/epsilon
    coff_0 = 1.0
    coff_1 = np.sum(X/2 - X*y, axis = 0).astype(np.float32)
    coff_2 = np.dot(X.T, X).astype(np.float32)
    noise_1 = np.random.laplace(0.0, sensitivity/epsilon, coff_1.shape).astype(np.float32)
    noise_2 = np.random.laplace(0.0, sensitivity/epsilon, coff_2.shape).astype(np.float32)
    if mode == 'scipy':
        return coff_0, (1/num_data_point)*coff_1.reshape(-1, 1), (1/num_data_point)*coff_2
    elif mode == 'scipy_dp':
        coff_1 = coff_1.reshape(-1, 1)
        coff_1 = coff_1 + noise_1
        coff_2 = coff_2 + np.triu(noise_2, k=0) + np.triu(noise_2, k=1).T + lbda*np.identity(num_feat)
        return coff_0, (1/num_data_point)*coff_1, (1/num_data_point)*coff_2
    elif mode == 'torch' or mode == 'fair':
        return coff_0, (1/num_data_point)*torch.from_numpy(coff_1.reshape(-1, 1)), (1/num_data_point)*torch.from_numpy(coff_2)
    elif mode == 'func' or mode == 'fairdp':
        coff_1 = coff_1 + noise_1
        coff_2 = coff_2 + np.triu(noise_2, k=0) + np.triu(noise_2, k=1).T + lbda*np.identity(num_feat)
        w, V = np.linalg.eig(coff_2)
        indx = np.where(w > 0)[0]
        w = w[indx].astype(np.float32)
        V = V[indx, :].astype(np.float32)
        coff_2 = np.identity(len(w))*w.astype(np.float32)
        coff_1 = np.dot(coff_1, V.T).astype(np.float32)
        return coff_0, (1/num_data_point)*torch.from_numpy(coff_1.reshape(-1, 1)), (1/num_data_point)*torch.from_numpy(coff_2), V,

def icml_bound(args, d):
    return d/(np.sqrt(2*np.pi)*args.ns_)

def get_name(args, current_date, fold):
    if args.mode != 'func':
        return '{}_{}_fold_{}_sigma_{}_C_{}_epochs_{}_{}{}{}_{}{}{}'.format(args.dataset,
                                                                        args.mode, fold,
                                                                        args.ns,
                                                                        args.clip,
                                                                        args.epochs,
                                                                        current_date.day,
                                                                        current_date.month,
                                                                        current_date.year,
                                                                        current_date.hour,
                                                                        current_date.minute,
                                                                        current_date.second)
    else:
        return '{}_{}_fold_{}_eps_{}_C_{}_epochs_{}_{}{}{}_{}{}{}'.format(args.dataset,
                                                                            args.mode, fold,
                                                                            args.tar_eps,
                                                                            args.clip,
                                                                            args.epochs,
                                                                            current_date.day,
                                                                            current_date.month,
                                                                            current_date.year,
                                                                            current_date.hour,
                                                                            current_date.minute,
                                                                            current_date.second)

def bound_kl(args, num_ep):
    M = (args.bs_male + args.bs_female)/(args.clip*(args.ns**2))
    epochs = np.arange(num_ep)
    return np.sqrt(1 - np.exp(-1*(M*epochs)))

def get_Mt(args, norm_grad):
    M = norm_grad / (args.clip**2 * (args.ns ** 2))
    return M

def bound_kl_emp(M):
    return np.sqrt(1 - np.exp(-1*(M)))
    # pass