import pickle

def get_gaussian_noise(clipping_val, noise_scale):
    return noise_scale * clipping_val

def bound(args):
    return args.lr * (2 * args.clip + 8 * args.clip ** 2 * args.ns ** 2) / 2

def save_res(fold, args, dct, current_time):
    save_name = args.res_path+'{}_{}_fold_{}_{}{}{}_{}{}{}'.format(args.dataset, args.mode, fold, current_time.day, current_time.month,
                                             current_time.year, current_time.hour, current_time.minute,
                                             current_time.second)
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)
