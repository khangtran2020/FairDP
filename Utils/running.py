import torch
from Data.datasets import Adult
from torch.utils.data import DataLoader
from Model.models import NeuralNetwork, NormNN
from Utils.train_eval import *
from Utils.plottings import *
from sklearn.metrics import accuracy_score
from Utils.metrics import *
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from opacus import PrivacyEngine
# from pyvacy import optim, analysis

def run_clean(fold, train_df, test_df, args, device, current_time):
    df_train = train_df[train_df.fold != fold]
    df_valid = train_df[train_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
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

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=3, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)

        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        es(acc_score, model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_dpsgd(fold, train_df, test_df, args, device, current_time):
    df_train = train_df[train_df.fold != fold]
    df_valid = train_df[train_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
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

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
    }

    # THE ENGINE LOOP
    i = 0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn_dpsgd(train_loader, model, criterion, optimizer, device,
                                                              scheduler=None, clipping=args.clip,
                                                              noise_scale=args.ns)
        # train_fn_dpsgd(train_loader, model,criterion, optimizer, device,scheduler=None,epoch=epoch, clipping=args.clip, noise_scale=args.ns)
        # return
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        es(acc_score, model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair(fold, male_df, female_df, test_df, args, device, current_time):
    df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
        drop=True)
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
        drop=True)
    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
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

    valid_male_loader = DataLoader(
        valid_male_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = DataLoader(
        valid_female_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'prob_male': [],
        'prob_female': [],
        'demo_parity': [],
        'male_tpr': [],
        'female_tpr': [],
        'equal_odd': [],
        'test_history_loss': [],
        'test_history_acc': [],
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader, model=model, device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['prob_male'].append(prob_male)
        history['prob_female'].append(prob_female)
        history['demo_parity'].append(demo_p)
        history['male_tpr'].append(male_tpr)
        history['female_tpr'].append(female_tpr)
        history['equal_odd'].append(equal_odd)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        es(acc_score, model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair(fold,history, epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd(fold, male_df, female_df, test_df, args, device, current_time):
    df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
        drop=True)
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
        drop=True)
    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
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
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Device

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'prob_male': [],
        'prob_female': [],
        'demo_parity': [],
        'male_tpr': [],
        'female_tpr': [],
        'equal_odd': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn_dpsgd(train_loader, model, criterion, optimizer, device,
                                                              scheduler=None, clipping=args.clip,
                                                              noise_scale=args.ns)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader, model=model, device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['prob_male'].append(prob_male)
        history['prob_female'].append(prob_female)
        history['demo_parity'].append(demo_p)
        history['male_tpr'].append(male_tpr)
        history['female_tpr'].append(female_tpr)
        history['equal_odd'].append(equal_odd)

        es(acc_score, model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

# def run_norm(fold, df, args, device, current_time):
#     df_train = df[df.fold != fold]
#     df_valid = df[df.fold == fold]
#
#     # Defining DataSet
#     train_dataset = Adult(
#         df_train[args.feature].values,
#         df_train[args.target].values
#     )
#
#     valid_dataset = Adult(
#         df_valid[args.feature].values,
#         df_valid[args.target].values
#     )
#
#     # Defining DataLoader with BalanceClass Sampler
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         pin_memory=True,
#         drop_last=True,
#         num_workers=4
#     )
#
#     valid_loader = torch.utils.data.DataLoader(
#         valid_dataset,
#         batch_size=args.batch_size,
#         num_workers=4,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False,
#     )
#
#     # Defining Model for specific fold
#     model = NormNN(args.input_dim, args.n_hid, args.output_dim)
#     model.to(device)
#
#     # DEfining criterion
#     criterion = torch.nn.BCELoss()
#     criterion.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#
#     # Defining LR SCheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
#                                                            factor=0.1, patience=10, verbose=True,
#                                                            threshold=0.0001, threshold_mode='rel',
#                                                            cooldown=0, min_lr=0, eps=1e-08)
#     # DEfining Early Stopping Object
#     es = EarlyStopping(patience=args.patience, verbose=False)
#
#     # History dictionary to store everything
#     history = {
#         'train_history_loss': [],
#         'train_history_acc': [],
#         'val_history_loss': [],
#         'val_history_acc': [],
#     }
#
#     # THE ENGINE LOOP
#     tk0 = tqdm(range(args.epochs), total=args.epochs)
#     for epoch in tk0:
#         train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
#                                                         scheduler=None)
#
#         val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
#
#         train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
#         acc_score = accuracy_score(targets, np.round(np.array(outputs)))
#
#         scheduler.step(acc_score)
#
#         tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
#                         Valid_ACC_SCORE=acc_score)
#
#         history['train_history_loss'].append(train_loss)
#         history['train_history_acc'].append(train_acc)
#         history['val_history_loss'].append(val_loss)
#         history['val_history_acc'].append(acc_score)
#
#         es(acc_score, model, args.save_path+f'model_{fold}.bin')
#
#         if es.early_stop:
#             print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
#             break
#
#     print_history(fold,history,epoch+1, args, current_time)
#     save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd_alg2(fold, male_df, female_df, test_df, args, device, current_time):
    df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
        drop=True)
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
        drop=True)
    df_train_mal = male_df[male_df.fold != fold]
    df_train_fem = female_df[female_df.fold != fold]
    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    train_male_dataset = Adult(
        df_train_mal[args.feature].values,
        df_train_mal[args.target].values
    )

    train_female_dataset = Adult(
        df_train_fem[args.feature].values,
        df_train_fem[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
    )

    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        sampler=sampler_male,
        num_workers=0
    )

    sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.batch_size,
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

    # Defining Model for specific fold
    model_male = NormNN(args.input_dim, args.n_hid, args.output_dim)
    model_female = NormNN(args.input_dim, args.n_hid, args.output_dim)
    global_model = NormNN(args.input_dim, args.n_hid, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
                                                                factor=0.1, patience=3, verbose=True,
                                                                threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=0, eps=1e-08)
    scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
                                                                  factor=0.1, patience=3, verbose=True,
                                                                  threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience,verbose=False)

    # History dictionary to store everything
    history = {
        'train_male_history_loss': [],
        'train_female_history_loss': [],
        'train_male_history_acc': [],
        'train_female_history_acc': [],
        'val_male_history_loss': [],
        'val_female_history_loss': [],
        'val_male_history_acc': [],
        'val_female_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'prob_male': [],
        'prob_female': [],
        'demo_parity': [],
        'male_tpr': [],
        'female_tpr': [],
        'equal_odd': [],
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:

        global_dict = global_model.state_dict()
        model_male.load_state_dict(global_dict)
        model_female.load_state_dict(global_dict)

        # dataloader, model, criterion, optimizer, device, scheduler, epoch, clipping, noise_scale
        train_male_loss, train_male_out, train_male_targets = train_fn_dpsgd(dataloader=train_male_loader,
                                                                             model=model_male,
                                                                             criterion=criterion,
                                                                             optimizer=optimizer_male,
                                                                             device=device,
                                                                             scheduler=None,
                                                                             clipping=args.clip,
                                                                             noise_scale=args.ns)

        train_female_loss, train_female_out, train_female_targets = train_fn_dpsgd(dataloader=train_female_loader,
                                                                                   model=model_female,
                                                                                   criterion=criterion,
                                                                                   optimizer=optimizer_female,
                                                                                   device=device,
                                                                                   scheduler=None,
                                                                                   clipping=args.clip,
                                                                                   noise_scale=args.ns)

        # update global model
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)

        global_model.load_state_dict(global_dict)

        val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, global_model, criterion, device)
        val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, global_model, criterion, device)
        train_loss, train_output, train_target = eval_fn(train_loader, global_model, criterion, device)
        valid_loss, valid_output, valid_target = eval_fn(valid_loader, global_model, criterion, device)
        test_loss, test_output, test_target = eval_fn(test_loader, global_model, criterion, device)

        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=global_model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader,
                                                          model=global_model, device=device)

        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=len(df_val_mal),
                                                  num_female=len(df_val_fem),
                                                  device=device)

        train_male_acc = accuracy_score(train_male_targets, np.round(np.array(train_male_out)))
        train_female_acc = accuracy_score(train_female_targets, np.round(np.array(train_female_out)))

        acc_male_score = accuracy_score(targets_male, np.round(np.array(outputs_male)))
        acc_female_score = accuracy_score(targets_female, np.round(np.array(outputs_female)))

        train_acc = accuracy_score(train_target, np.round(np.array(train_output)))
        val_acc = accuracy_score(valid_target, np.round(np.array(valid_output)))
        test_acc = accuracy_score(test_target, np.round(np.array(test_output)))

        scheduler_male.step(acc_male_score)
        scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Male_Loss=train_male_loss, Train_Male_ACC_SCORE=train_male_acc,
                        Train_Female_Loss=train_female_loss,
                        Train_Female_ACC_SCORE=train_female_acc, Valid_Male_Loss=val_male_loss,
                        Valid_Male_ACC_SCORE=acc_male_score,
                        Valid_Female_Loss=val_female_loss, Valid_Female_ACC_SCORE=acc_female_score)

        history['train_male_history_loss'].append(train_male_loss)
        history['train_female_history_loss'].append(train_female_loss)
        history['train_male_history_acc'].append(train_male_acc)
        history['train_female_history_acc'].append(train_female_acc)
        history['val_male_history_loss'].append(val_male_loss)
        history['val_female_history_loss'].append(val_female_loss)
        history['val_male_history_acc'].append(acc_male_score)
        history['val_female_history_acc'].append(acc_female_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(valid_loss)
        history['val_history_acc'].append(val_acc)
        history['prob_male'].append(prob_male)
        history['prob_female'].append(prob_female)
        history['demo_parity'].append(demo_p)
        history['male_tpr'].append(male_tpr)
        history['female_tpr'].append(female_tpr)
        history['equal_odd'].append(equal_odd)
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        es(val_acc,global_model,args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair_v4(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_v2(fold, male_df, female_df, test_df, args, device, current_time):
    df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis=0).reset_index(
        drop=True)
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
        drop=True)

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)

    df_train_mal = male_df[male_df.fold != fold]
    df_train_fem = female_df[female_df.fold != fold]

    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    train_male_dataset = Adult(
        df_train_mal[args.feature].values,
        df_train_mal[args.target].values
    )

    train_female_dataset = Adult(
        df_train_fem[args.feature].values,
        df_train_fem[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
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

    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.batch_size,
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

    # Defining Model for specific fold
    model_male = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model_female = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    global_model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)
    optimizer_global = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler_global = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_global, mode='max',
                                                                  factor=0.1, patience=10, verbose=True,
                                                                  threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)
    scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
                                                                factor=0.1, patience=10, verbose=True,
                                                                threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=0, eps=1e-08)
    scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
                                                                  factor=0.1, patience=10, verbose=True,
                                                                  threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    # es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_global_history_loss': [],
        'train_male_history_loss': [],
        'train_female_history_loss': [],
        'train_global_history_acc': [],
        'train_male_history_acc': [],
        'train_female_history_acc': [],
        'val_global_history_loss': [],
        'val_male_history_loss': [],
        'val_female_history_loss': [],
        'val_global_history_acc': [],
        'val_male_history_acc': [],
        'val_female_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:

        # dataloader, model, criterion, optimizer, device, scheduler, epoch, clipping, noise_scale
        # global
        train_global_loss, train_global_out, train_global_targets = train_fn(train_loader, global_model, criterion,
                                                                             optimizer_global, device, scheduler=None)
        # male
        train_male_loss, train_male_out, train_male_targets = train_fn(train_male_loader, model_male, criterion,
                                                                       optimizer_male, device, scheduler=None)
        # female
        train_female_loss, train_female_out, train_female_targets = train_fn(train_female_loader, model_female,
                                                                             criterion, optimizer_female, device,
                                                                             scheduler=None)

        val_global_loss, outputs_global, targets_global = eval_fn(valid_loader, global_model, criterion, device)
        val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, model_male, criterion, device)
        val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, model_female, criterion, device)
        test_loss, test_out, test_tar = eval_fn(test_loader, global_model, criterion, device)

        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=len(df_val_mal),
                                                  num_female=len(df_val_fem),
                                                  device=device)

        train_male_acc = accuracy_score(train_male_targets, np.round(np.array(train_male_out)))
        train_female_acc = accuracy_score(train_female_targets, np.round(np.array(train_female_out)))
        train_global_acc = accuracy_score(train_global_targets, np.round(np.array(train_global_out)))

        acc_male_score = accuracy_score(targets_male, np.round(np.array(outputs_male)))
        acc_female_score = accuracy_score(targets_female, np.round(np.array(outputs_female)))
        acc_global_score = accuracy_score(targets_global, np.round(np.array(outputs_global)))

        test_acc = accuracy_score(test_tar, np.round(np.array(test_out)))

        scheduler_male.step(acc_male_score)
        scheduler_female.step(acc_female_score)
        scheduler_global.step(acc_global_score)

        tk0.set_postfix(Train_Male_Loss=train_male_loss, Train_Male_ACC_SCORE=train_male_acc,
                        Train_Female_Loss=train_female_loss,
                        Train_Female_ACC_SCORE=train_female_acc, Valid_Male_Loss=val_male_loss,
                        Valid_Male_ACC_SCORE=acc_male_score,
                        Valid_Female_Loss=val_female_loss, Valid_Female_ACC_SCORE=acc_female_score)

        history['train_global_history_loss'].append(train_global_loss)
        history['train_male_history_loss'].append(train_male_loss)
        history['train_female_history_loss'].append(train_female_loss)
        history['train_global_history_acc'].append(train_global_acc)
        history['train_male_history_acc'].append(train_male_acc)
        history['train_female_history_acc'].append(train_female_acc)
        history['val_global_history_loss'].append(val_global_loss)
        history['val_male_history_loss'].append(val_male_loss)
        history['val_female_history_loss'].append(val_female_loss)
        history['val_global_history_acc'].append(acc_global_score)
        history['val_male_history_acc'].append(acc_male_score)
        history['val_female_history_acc'].append(acc_female_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        # es(acc_global_score, global_model, args.save_path+f'model_{fold}.bin')
        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    print_history_fair_v2(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_v3(fold, male_df, female_df, test_df, args, device, current_time):
    # df_train = pd.concat([male_df[male_df.fold != fold], female_df[female_df.fold != fold]], axis = 0).reset_index(drop=True)
    df_valid = pd.concat([male_df[male_df.fold == fold], female_df[female_df.fold == fold]], axis=0).reset_index(
        drop=True)

    # df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)

    df_train_mal = male_df[male_df.fold != fold]
    df_train_fem = female_df[female_df.fold != fold]

    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    # train_dataset = Adult(
    #     df_train[args.feature].values,
    #     df_train[args.target].values
    # )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    train_male_dataset = Adult(
        df_train_mal[args.feature].values,
        df_train_mal[args.target].values
    )

    train_female_dataset = Adult(
        df_train_fem[args.feature].values,
        df_train_fem[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=0
    # )

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

    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.batch_size,
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

    # Defining Model for specific fold
    model_male = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model_female = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    global_model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)
    # optimizer_global = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
                                                                factor=0.1, patience=10, verbose=True,
                                                                threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=0, eps=1e-08)
    scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
                                                                  factor=0.1, patience=10, verbose=True,
                                                                  threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        # 'train_global_history_loss': [],
        'train_male_history_loss': [],
        'train_female_history_loss': [],
        # 'train_global_history_acc': [],
        'train_male_history_acc': [],
        'train_female_history_acc': [],
        'val_global_history_loss': [],
        'val_male_history_loss': [],
        'val_female_history_loss': [],
        'val_global_history_acc': [],
        'val_male_history_acc': [],
        'val_female_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        global_dict = global_model.state_dict()
        model_male.load_state_dict(global_dict)
        model_female.load_state_dict(global_dict)

        # dataloader, model, criterion, optimizer, device, scheduler, epoch, clipping, noise_scale
        # train_global_loss,train_global_out,train_global_targets = train_fn(train_loader, global_model,criterion, optimizer_global, device,scheduler=None,epoch=epoch)
        train_male_loss, train_male_out, train_male_targets = train_fn(train_male_loader, model_male, criterion,
                                                                       optimizer_male, device, scheduler=None)
        train_female_loss, train_female_out, train_female_targets = train_fn(train_female_loader, model_female,
                                                                             criterion, optimizer_female, device,
                                                                             scheduler=None)
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)
        global_model.load_state_dict(global_dict)

        # val_global_loss, outputs_global, targets_global = eval_fn(valid_male_loader, global_model, criterion, device)
        val_global_loss, outputs_global, targets_global = eval_fn(valid_loader, global_model, criterion, device)
        val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, model_male, criterion, device)
        val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, model_female, criterion, device)
        test_loss, test_out, test_tar = eval_fn(valid_loader, global_model, criterion, device)

        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=len(df_val_mal),
                                                  num_female=len(df_val_fem),
                                                  device=device)

        train_male_acc = accuracy_score(train_male_targets, np.round(np.array(train_male_out)))
        train_female_acc = accuracy_score(train_female_targets, np.round(np.array(train_female_out)))
        # train_global_acc = accuracy_score(train_global_targets, np.round(np.array(train_global_out)))

        acc_male_score = accuracy_score(targets_male, np.round(np.array(outputs_male)))
        acc_female_score = accuracy_score(targets_female, np.round(np.array(outputs_female)))
        acc_global_score = accuracy_score(targets_global, np.round(np.array(outputs_global)))
        test_acc = accuracy_score(test_tar, np.round(np.array(test_out)))

        # scheduler_male.step(acc_male_score)
        # scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Male_Loss=train_male_loss, Train_Male_ACC_SCORE=train_male_acc,
                        Train_Female_Loss=train_female_loss,
                        Train_Female_ACC_SCORE=train_female_acc, Valid_Male_Loss=val_male_loss,
                        Valid_Male_ACC_SCORE=acc_male_score,
                        Valid_Female_Loss=val_female_loss, Valid_Female_ACC_SCORE=acc_female_score)

        # history['train_global_history_loss'].append(train_global_loss)
        history['train_male_history_loss'].append(train_male_loss)
        history['train_female_history_loss'].append(train_female_loss)
        # history['train_global_history_acc'].append(train_male_acc)
        history['train_male_history_acc'].append(train_male_acc)
        history['train_female_history_acc'].append(train_female_acc)
        history['val_global_history_loss'].append(val_global_loss)
        history['val_male_history_loss'].append(val_male_loss)
        history['val_female_history_loss'].append(val_female_loss)
        history['val_global_history_acc'].append(acc_global_score)
        history['val_male_history_acc'].append(acc_male_score)
        history['val_female_history_acc'].append(acc_female_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        es(acc_global_score, global_model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair_v3(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd_alg1(fold, male_df, female_df, test_df, args, device, current_time):
    df_train_mal = male_df[male_df.fold != fold]
    df_train_fem = female_df[female_df.fold != fold]
    df_val_mal = male_df[male_df.fold == fold]
    df_val_fem = female_df[female_df.fold == fold]

    # Defining DataSet
    train_male_dataset = Adult(
        df_train_mal[args.feature].values,
        df_train_mal[args.target].values
    )

    train_female_dataset = Adult(
        df_train_fem[args.feature].values,
        df_train_fem[args.target].values
    )

    valid_male_dataset = Adult(
        df_val_mal[args.feature].values,
        df_val_mal[args.target].values
    )

    valid_female_dataset = Adult(
        df_val_fem[args.feature].values,
        df_val_fem[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        sampler=sampler_male,
        num_workers=0
    )

    sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.batch_size,
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model_male = NormNN(args.input_dim, args.n_hid, args.output_dim)
    model_female = NormNN(args.input_dim, args.n_hid, args.output_dim)
    global_model = NormNN(args.input_dim, args.n_hid, args.output_dim)

    global_dict = global_model.state_dict()
    model_male.load_state_dict(global_dict)
    model_female.load_state_dict(global_dict)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
                                                                factor=0.1, patience=10, verbose=True,
                                                                threshold=0.0001, threshold_mode='rel',
                                                                cooldown=0, min_lr=0, eps=1e-08)
    scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
                                                                  factor=0.1, patience=10, verbose=True,
                                                                  threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    # es = EarlyStopping(patience=args.patience,verbose=False)

    # History dictionary to store everything
    history = {
        'train_male_history_loss': [],
        'train_female_history_loss': [],
        'train_male_history_acc': [],
        'train_female_history_acc': [],
        'val_male_history_loss': [],
        'val_female_history_loss': [],
        'val_male_history_acc': [],
        'val_female_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'prob_male': [],
        'prob_female': [],
        'demo_parity': [],
        'male_tpr': [],
        'female_tpr': [],
        'equal_odd': [],
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:

        # dataloader, model, criterion, optimizer, device, scheduler, epoch, clipping, noise_scale
        train_male_loss, train_male_out, train_male_targets = train_fn_dpsgd(dataloader=train_male_loader,
                                                                             model=model_male,
                                                                             criterion=criterion,
                                                                             optimizer=optimizer_male,
                                                                             device=device,
                                                                             scheduler=None,
                                                                             clipping=args.clip,
                                                                             noise_scale=args.ns)

        train_female_loss, train_female_out, train_female_targets = train_fn_dpsgd(dataloader=train_female_loader,
                                                                                   model=model_female,
                                                                                   criterion=criterion,
                                                                                   optimizer=optimizer_female,
                                                                                   device=device,
                                                                                   scheduler=None,
                                                                                   clipping=args.clip,
                                                                                   noise_scale=args.ns)

        # update global model
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)

        global_model.load_state_dict(global_dict)

        val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, global_model, criterion, device)
        val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, global_model, criterion, device)
        test_loss, test_out, test_tar = eval_fn(test_loader, global_model, criterion, device)

        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=global_model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader,
                                                          model=global_model, device=device)

        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=len(df_val_mal),
                                                  num_female=len(df_val_fem),
                                                  device=device)

        train_male_acc = accuracy_score(train_male_targets, np.round(np.array(train_male_out)))
        train_female_acc = accuracy_score(train_female_targets, np.round(np.array(train_female_out)))

        acc_male_score = accuracy_score(targets_male, np.round(np.array(outputs_male)))
        acc_female_score = accuracy_score(targets_female, np.round(np.array(outputs_female)))
        test_acc = accuracy_score(test_tar, np.round(np.array(test_out)))

        scheduler_male.step(acc_male_score)
        scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Male_Loss=train_male_loss, Train_Male_ACC_SCORE=train_male_acc,
                        Train_Female_Loss=train_female_loss,
                        Train_Female_ACC_SCORE=train_female_acc, Valid_Male_Loss=val_male_loss,
                        Valid_Male_ACC_SCORE=acc_male_score,
                        Valid_Female_Loss=val_female_loss, Valid_Female_ACC_SCORE=acc_female_score)

        history['train_male_history_loss'].append(train_male_loss)
        history['train_female_history_loss'].append(train_female_loss)
        history['train_male_history_acc'].append(train_male_acc)
        history['train_female_history_acc'].append(train_female_acc)
        history['val_male_history_loss'].append(val_male_loss)
        history['val_female_history_loss'].append(val_female_loss)
        history['val_male_history_acc'].append(acc_male_score)
        history['val_female_history_acc'].append(acc_female_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['prob_male'].append(prob_male)
        history['prob_female'].append(prob_female)
        history['demo_parity'].append(demo_p)
        history['male_tpr'].append(male_tpr)
        history['female_tpr'].append(female_tpr)
        history['equal_odd'].append(equal_odd)
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        # es(acc_score,model,args.save_path+f'model_{fold}.bin')

        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    print_history_fair_alg1(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_opacus(fold, train_df, test_df, args, device, current_time):
    df_train = train_df[train_df.fold != fold]
    df_valid = train_df[train_df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    test_dataset = Adult(
        test_df[args.feature].values,
        test_df[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
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

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.tar_eps,
        target_delta=args.tar_delt,
        max_grad_norm=args.clip,
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.clip}")

    # Defining LR SCheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                        factor=0.1, patience=10, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
    }

    # THE ENGINE LOOP
    i = 0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_opacus(train_loader, model, criterion, optimizer, device,args)
        # train_fn_dpsgd(train_loader, model,criterion, optimizer, device,scheduler=None,epoch=epoch, clipping=args.clip, noise_scale=args.ns)
        # return
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        # scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        es(acc_score, model, args.save_path+f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold,history,epoch+1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)