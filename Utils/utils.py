import pickle
import random
import os
import numpy as np
import torch


def get_gaussian_noise(clipping_val, noise_scale):
    return noise_scale * clipping_val

def bound(args):
    if args.mode == 'onebatch':
        return 2 * args.lr * args.clip * np.sqrt(
        args.num_params) + 2 * args.lr / args.batch_size * args.ns * args.clip * np.sqrt(
        -1 * args.num_params * np.log(1 - args.confidence))
    else:
        return args.n_batch * (2 * args.lr * args.clip * np.sqrt(
            args.num_params) + 2 * args.lr / args.batch_size * args.ns * args.clip * np.sqrt(
            -1 * args.num_params * np.log(1 - args.confidence)))

def bound_alg1(args):
    return args.epochs * args.lr * (2 * args.clip + 8 * args.clip ** 2 * args.ns ** 2) / 2

def save_res(fold, args, dct, current_time):
    save_name = args.plot_path + '{}_{}_fold_{}_sigma_{}_C_{}_epochs_{}_{}{}{}_{}{}{}'.format(args.dataset,
                                                                                                  args.mode, fold,
                                                                                                  args.ns,
                                                                                                  args.clip,
                                                                                                  args.epochs,
                                                                                                  current_time.day,
                                                                                                  current_time.month,
                                                                                                  current_time.year,
                                                                                                  current_time.hour,
                                                                                                  current_time.minute,
                                                                                                  current_time.second)
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
