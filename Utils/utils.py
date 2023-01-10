import pickle
import random
import os
import numpy as np
import torch
from sklearn.metrics import log_loss
from Model.models import *
from torch.utils.data import DataLoader
from Data.datasets import Data
import pandas as pd


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
    name = get_name(args=args, current_date=current_time, fold=fold)
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
    sensitivity = num_feat ** 2 / 4 + num_feat
    coff_0 = 1.0
    coff_1 = np.sum(X / 2 - X * y, axis=0).astype(np.float32)
    coff_2 = (1 / 8) * np.dot(X.T, X).astype(np.float32)
    noise_1 = np.random.laplace(0.0, sensitivity / epsilon, coff_1.shape).astype(np.float32)
    noise_2 = np.random.laplace(0.0, sensitivity / epsilon, coff_2.shape).astype(np.float32)
    if mode == 'scipy':
        return coff_0, (1 / num_data_point) * coff_1.reshape(-1, 1), (1 / num_data_point) * coff_2
    elif mode == 'torch' or mode == 'fair':
        return coff_0, (1 / num_data_point) * torch.from_numpy(coff_1.reshape(-1, 1)), (
                1 / num_data_point) * torch.from_numpy(coff_2)
    elif mode == 'func' or mode == 'fairdp' or mode == 'func_org' or mode == 'func_org':
        coff_1 = coff_1 + noise_1
        coff_2 = coff_2 + noise_2
        coff_2 = 1 / 2 + (coff_2 + coff_2.T)
        coff_2 = coff_2 + 5 * np.sqrt(2) * sensitivity * (1 / epsilon) * np.eye(num_feat)
        w, V = np.linalg.eig(coff_2)
        indx = np.where(w > 1e-8)[0]
        w = w[indx].astype(np.float32)
        V = V[:, indx].astype(np.float32)
        coff_2 = np.diag(w)
        coff_1 = np.dot(V.T, coff_1).astype(np.float32)
        return coff_0, (1 / num_data_point) * torch.from_numpy(coff_1.reshape(-1, 1)), (
                1 / num_data_point) * torch.from_numpy(coff_2), V,


def icml_bound(args, d):
    return d / (np.sqrt(2 * np.pi))


def get_name(args, current_date, fold):
    if args.mode == 'func':
        return '{}_{}_fold_{}_submode_{}_eps_{}_epochs_{}_{}{}{}_{}{}{}'.format(args.dataset,
                                                                                args.mode, fold,
                                                                                args.submode,
                                                                                args.tar_eps,
                                                                                args.epochs,
                                                                                current_date.day,
                                                                                current_date.month,
                                                                                current_date.year,
                                                                                current_date.hour,
                                                                                current_date.minute,
                                                                                current_date.second)

    elif args.mode == 'ratio':
        return '{}_{}_fold_{}_sigma_{}_C_{}_epochs_{}_ratio_{}_{}{}{}_{}{}{}'.format(args.dataset,
                                                                            args.mode, fold,
                                                                            args.ns,
                                                                            args.clip,
                                                                            args.epochs,
                                                                            args.ratio,
                                                                            current_date.day,
                                                                            current_date.month,
                                                                            current_date.year,
                                                                            current_date.hour,
                                                                            current_date.minute,
                                                                            current_date.second)
    else:
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


def bound_kl(args, num_ep):
    M = (args.bs_male + args.bs_female) / (args.ns ** 2)
    epochs = np.arange(num_ep)
    return np.sqrt(1 - np.exp(-1 * (M * epochs)))


def get_Mt(args, norm_grad):
    M = norm_grad / (args.clip ** 2 * (args.ns ** 2))
    return M


def bound_kl_emp(M):
    return np.sqrt(1 - np.exp(-1 * (M)))
    # pass


def init_model(args):
    if args.model_type == 'NormNN':
        return NormNN(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'NN':
        return NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'Logit':
        return NormLogit(args.input_dim, args.n_hid, args.output_dim)


def init_data(args, fold, train_df, test_df, male_df, female_df):
    if args.mode == 'clean':
        df_train = train_df[train_df.fold != fold]
        df_valid = train_df[train_df.fold == fold]

        # Defining DataSet
        train_dataset = Data(
            X=df_train[args.feature].values,
            y=df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            X=df_valid[args.feature].values,
            y=df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            X=test_df[args.feature].values,
            y=test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=4
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader
    elif args.mode == 'dpsgd':
        df_train = train_df[train_df.fold != fold]
        df_valid = train_df[train_df.fold == fold]

        # Defining DataSet
        train_dataset = Data(
            X=df_train[args.feature].values,
            y=df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            X=df_valid[args.feature].values,
            y=df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            X=test_df[args.feature].values,
            y=test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            num_workers=0
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader
    elif args.mode == 'func':
        df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
            drop=True)
        df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
            drop=True)
        df_train_mal = male_df[male_df.fold != fold]
        df_train_fem = female_df[female_df.fold != fold]
        df_val_mal = male_df[male_df.fold == fold]
        df_val_fem = female_df[female_df.fold == fold]

        df_train['bias'] = 1
        df_valid['bias'] = 1
        df_train_mal['bias'] = 1
        df_train_fem['bias'] = 1
        df_val_mal['bias'] = 1
        df_val_fem['bias'] = 1

        # female
        X_fem = df_train_fem[args.feature].values
        y_fem = df_train_fem[args.target].values.reshape(-1, 1)
        X_fem = X_fem / np.linalg.norm(X_fem, ord=2, axis=1).reshape(-1, 1)

        # male
        X_mal = df_train_mal[args.feature].values
        y_mal = df_train_mal[args.target].values.reshape(-1, 1)
        X_mal = X_mal / np.linalg.norm(X_mal, ord=2, axis=1).reshape(-1, 1)

        # train
        X_train = df_train[args.feature].values
        y_train = df_train[args.target].values.reshape(-1, 1)
        X_train = X_train / np.linalg.norm(X_train, ord=2, axis=1).reshape(-1, 1)

        # valid
        X_mal_val = df_val_mal[args.feature].values
        y_mal_val = df_val_mal[args.target].values.reshape(-1, 1)
        X_mal_val = X_mal_val / np.linalg.norm(X_mal_val, ord=2, axis=1).reshape(-1, 1)

        X_fem_val = df_val_fem[args.feature].values
        y_fem_val = df_val_fem[args.target].values.reshape(-1, 1)
        X_fem_val = X_fem_val / np.linalg.norm(X_fem_val, ord=2, axis=1).reshape(-1, 1)

        X_valid = df_valid[args.feature].values
        y_valid = df_valid[args.target].values.reshape(-1, 1)
        X_valid = X_valid / np.linalg.norm(X_valid, ord=2, axis=1).reshape(-1, 1)

        # test

        X_test = test_df[args.feature].values
        y_test = test_df[args.target].values.reshape(-1, 1)
        X_test = X_test / np.linalg.norm(X_test, ord=2, axis=1).reshape(-1, 1)
        return X_train, X_valid, X_test, X_mal, X_fem, X_mal_val, X_fem_val, y_train, y_valid, y_test, y_mal, y_fem, y_mal_val, y_fem_val
    elif args.mode == 'fair':
        df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
            drop=True)
        df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
            drop=True)
        df_train_mal = male_df[male_df.fold != fold]
        df_train_fem = female_df[female_df.fold != fold]
        df_val_mal = male_df[male_df.fold == fold]
        df_val_fem = female_df[female_df.fold == fold]

        # Defining DataSet
        train_male_dataset = Data(
            X=df_train_mal[args.feature].values,
            y=df_train_mal[args.target].values,
            ismale=df_train_mal[args.z].values
        )

        train_female_dataset = Data(
            df_train_fem[args.feature].values,
            df_train_fem[args.target].values,
            ismale=df_train_fem[args.z].values
        )

        valid_male_dataset = Data(
            df_val_mal[args.feature].values,
            df_val_mal[args.target].values,
            ismale=df_val_mal[args.z].values
        )

        valid_female_dataset = Data(
            df_val_fem[args.feature].values,
            df_val_fem[args.target].values,
            ismale=df_val_fem[args.z].values
        )

        train_dataset = Data(
            df_train[args.feature].values,
            df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            df_valid[args.feature].values,
            df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            test_df[args.feature].values,
            test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        # sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
        train_male_loader = DataLoader(
            train_male_dataset,
            batch_size=int(args.sampling_rate * len(train_male_dataset)),
            pin_memory=True,
            drop_last=True,
            num_workers=0
        )

        # sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
        train_female_loader = DataLoader(
            train_female_dataset,
            batch_size=int(args.sampling_rate * len(train_female_dataset)),
            pin_memory=True,
            drop_last=True,
            num_workers=0
        )

        valid_male_loader = torch.utils.data.DataLoader(
            valid_male_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_female_loader = torch.utils.data.DataLoader(
            valid_female_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(args.sampling_rate * len(train_dataset)),
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        args.n_batch = len(train_male_loader)
        args.bs_male = int(args.sampling_rate * len(train_male_dataset))
        args.bs_female = int(args.sampling_rate * len(train_female_dataset))
        args.bs = int(args.sampling_rate * len(train_dataset))
        args.num_val_male = len(df_val_mal)
        args.num_val_female = len(df_val_fem)
        return train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader
    elif args.mode == 'fairdp':
        df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
            drop=True)
        df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
            drop=True)
        df_train_mal = male_df[male_df.fold != fold]
        df_train_fem = female_df[female_df.fold != fold]
        df_val_mal = male_df[male_df.fold == fold]
        df_val_fem = female_df[female_df.fold == fold]

        # Defining DataSet
        train_male_dataset = Data(
            X=df_train_mal[args.feature].values,
            y=df_train_mal[args.target].values,
            ismale=df_train_mal[args.z].values
        )

        train_female_dataset = Data(
            df_train_fem[args.feature].values,
            df_train_fem[args.target].values,
            ismale=df_train_fem[args.z].values
        )

        valid_male_dataset = Data(
            df_val_mal[args.feature].values,
            df_val_mal[args.target].values,
            ismale=df_val_mal[args.z].values
        )

        valid_female_dataset = Data(
            df_val_fem[args.feature].values,
            df_val_fem[args.target].values,
            ismale=df_val_fem[args.z].values
        )

        train_dataset = Data(
            df_train[args.feature].values,
            df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            df_valid[args.feature].values,
            df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            test_df[args.feature].values,
            test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
        train_male_loader = DataLoader(
            train_male_dataset,
            batch_size=int(args.sampling_rate * len(train_male_dataset)),
            pin_memory=True,
            sampler=sampler_male,
            drop_last=True,
            num_workers=0
        )

        sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
        train_female_loader = DataLoader(
            train_female_dataset,
            batch_size=int(args.sampling_rate * len(train_female_dataset)),
            pin_memory=True,
            sampler=sampler_female,
            drop_last=True,
            num_workers=0
        )

        valid_male_loader = torch.utils.data.DataLoader(
            valid_male_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_female_loader = torch.utils.data.DataLoader(
            valid_female_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        sampler_ = torch.utils.data.RandomSampler(train_dataset, replacement=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(args.sampling_rate * len(train_dataset)),
            num_workers=0,
            sampler=sampler_,
            # shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        args.n_batch = len(train_male_loader)
        args.bs_male = int(args.sampling_rate * len(train_male_dataset))
        args.bs_female = int(args.sampling_rate * len(train_female_dataset))
        args.bs = int(args.sampling_rate * len(train_dataset))
        args.num_val_male = len(df_val_mal)
        args.num_val_female = len(df_val_fem)
        return train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader
    elif args.mode in ['fairdp', 'fairdp_epoch', 'fairdp_track', 'onebatch', 'proposed', 'smooth', 'fairdp_baseline',
                       'ratio']:
        df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
            drop=True)
        df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
            drop=True)
        df_train_mal = male_df[male_df.fold != fold]
        df_train_fem = female_df[female_df.fold != fold]
        df_val_mal = male_df[male_df.fold == fold]
        df_val_fem = female_df[female_df.fold == fold]

        # Defining DataSet
        train_male_dataset = Data(
            X=df_train_mal[args.feature].values,
            y=df_train_mal[args.target].values,
            ismale=df_train_mal[args.z].values
        )

        train_female_dataset = Data(
            df_train_fem[args.feature].values,
            df_train_fem[args.target].values,
            ismale=df_train_fem[args.z].values
        )

        valid_male_dataset = Data(
            df_val_mal[args.feature].values,
            df_val_mal[args.target].values,
            ismale=df_val_mal[args.z].values
        )

        valid_female_dataset = Data(
            df_val_fem[args.feature].values,
            df_val_fem[args.target].values,
            ismale=df_val_fem[args.z].values
        )

        train_dataset = Data(
            df_train[args.feature].values,
            df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            df_valid[args.feature].values,
            df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            test_df[args.feature].values,
            test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
        train_male_loader = DataLoader(
            train_male_dataset,
            batch_size=int(args.sampling_rate * len(train_male_dataset)),
            pin_memory=True,
            drop_last=True,
            sampler=sampler_male,
            num_workers=0
        )

        sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
        train_female_loader = DataLoader(
            train_female_dataset,
            batch_size=int(args.sampling_rate * len(train_female_dataset)),
            pin_memory=True,
            drop_last=True,
            sampler=sampler_female,
            num_workers=0
        )

        valid_male_loader = torch.utils.data.DataLoader(
            valid_male_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_female_loader = torch.utils.data.DataLoader(
            valid_female_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        args.n_batch = len(train_male_loader)
        args.bs_male = int(args.sampling_rate * len(train_male_dataset))
        args.bs_female = int(args.sampling_rate * len(train_female_dataset))
        args.num_val_male = len(df_val_mal)
        args.num_val_female = len(df_val_fem)
        return train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader
    elif args.mode == 'fair_test':
        df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
            drop=True)
        df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
            drop=True)
        df_train_mal = male_df[male_df.fold != fold]
        df_train_fem = female_df[female_df.fold != fold]
        df_val_mal = male_df[male_df.fold == fold]
        df_val_fem = female_df[female_df.fold == fold]

        # Defining DataSet
        train_male_dataset = Data(
            X=df_train_mal[args.feature].values,
            y=df_train_mal[args.target].values,
            ismale=df_train_mal[args.z].values
        )

        train_female_dataset = Data(
            df_train_fem[args.feature].values,
            df_train_fem[args.target].values,
            ismale=df_train_fem[args.z].values
        )

        valid_male_dataset = Data(
            df_val_mal[args.feature].values,
            df_val_mal[args.target].values,
            ismale=df_val_mal[args.z].values
        )

        valid_female_dataset = Data(
            df_val_fem[args.feature].values,
            df_val_fem[args.target].values,
            ismale=df_val_fem[args.z].values
        )

        train_dataset = Data(
            df_train[args.feature].values,
            df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            df_valid[args.feature].values,
            df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            test_df[args.feature].values,
            test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        batch_size = min(int(args.sampling_rate * len(train_male_dataset)),
                         int(args.sampling_rate * len(train_female_dataset)))

        # Defining DataLoader with BalanceClass Sampler
        sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
        train_male_loader = DataLoader(
            train_male_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler_male,
            num_workers=0
        )

        sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
        train_female_loader = DataLoader(
            train_female_dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler_female,
            num_workers=0
        )

        valid_male_loader = torch.utils.data.DataLoader(
            valid_male_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_female_loader = torch.utils.data.DataLoader(
            valid_female_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        args.n_batch = len(train_male_loader)
        args.bs_male = batch_size
        args.bs_female = batch_size
        args.num_val_male = len(df_val_mal)
        args.num_val_female = len(df_val_fem)
        return train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader

# def choose_data(args, df_1, df_2):
#
#     if len(df_1) > len(df_2):
#         df = df_2.copy()
#         df_2 = df_1.copy()
#         df_1 = df.copy()
