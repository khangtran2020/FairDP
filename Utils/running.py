import numpy as np
import torch
from Data.datasets import Data
from torch.utils.data import DataLoader
from Model.models import NeuralNetwork, NormNN
from Utils.train_eval import *
from Utils.plottings import *
from sklearn.metrics import accuracy_score
from Utils.metrics import *
from Utils.utils import count_parameters, sigmoid, logloss, get_coefficient
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.accountants import create_accountant


# from pyvacy import optim, analysis

def run_clean(fold, train_df, test_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    train_loader, valid_loader, test_loader = init_data(args=args, fold=fold, train_df=train_df, test_df=test_df,
                                                        male_df=None, female_df=None)
    model = init_model(args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=5e-4, eps=1e-08)
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
        'best_test': 0
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

        # scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)

        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    print_history(fold, history, epoch + 1, args, current_time)
    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    history['best_test'] = test_acc
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_dpsgd(fold, train_df, test_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    train_loader, valid_loader, test_loader = init_data(args=args, fold=fold, train_df=train_df, test_df=test_df,
                                                        male_df=None, female_df=None)

    # Defining Model for specific fold
    model = init_model(args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                        factor=0.1, patience=3, verbose=True,
    #                                                        threshold=0.0005, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=0.0005, eps=1e-08)
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
        'best_test': 0
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

        # scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        # es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)
        torch.save(model.state_dict(), args.save_path + model_name)
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    history['best_test'] = test_acc
    print_history(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_fair(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)
    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    # Defining Model for specific fold
    model = init_model(args=args)
    model_male = init_model(args=args)
    model_female = init_model(args=args)

    model.to(device)
    model_male.to(device)
    model_female.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # # Defining LR SCheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                        factor=0.1, patience=10, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
    #                                                             factor=0.1, patience=10, verbose=True,
    #                                                             threshold=0.0001, threshold_mode='rel',
    #                                                             cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
    #                                                               factor=0.1, patience=10, verbose=True,
    #                                                               threshold=0.0001, threshold_mode='rel',
    #                                                               cooldown=0, min_lr=1e-4, eps=1e-08)

    # DEfining Early Stopping Object
    # es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_acc_parity': 0,
        'best_equal_odd': 0,
        'best_disp_imp': 0,
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        _, _, _ = train_fn(train_male_loader, model_male, criterion, optimizer_male, device,
                           scheduler=None)
        _, _, _ = train_fn(train_female_loader, model_female, criterion, optimizer_female, device,
                           scheduler=None)
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=model, device=device)
        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=model, device=device)
        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=args.num_val_male,
                                                  num_female=args.num_val_female,
                                                  device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        # scheduler.step(acc_score)
        # scheduler_male.step(male_acc_score)
        # scheduler_female.step(female_acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['demo_parity'].append(demo_p)
        history['acc_parity'].append(acc_par)
        history['equal_odd'].append(equal_odd)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        torch.save(model.state_dict(), args.save_path + model_name)
        # es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)
        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    # model.load_state_dict(torch.load(args.save_path + model_name))
    # test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
    # test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    # _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
    #                            model=model, device=device)
    # _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
    #                                   female_loader=valid_female_loader, model=model, device=device)
    # male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
    #                                           female_loader=valid_female_loader,
    #                                           global_model=model,
    #                                           male_model=model_male,
    #                                           female_model=model_female,
    #                                           num_male=len(df_val_mal),
    #                                           num_female=len(df_val_fem),
    #                                           device=device)
    # history['best_test'] = test_acc
    # history['best_demo_parity'] = demo_p
    # history['best_equal_odd'] = equal_odd
    # history['best_disp_imp'] = max(male_norm, female_norm)
    print_history_fair(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_fair_dpsgd(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)
    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    # Defining Model for specific fold
    model = init_model(args=args)
    model_male = init_model(args=args)
    model_female = init_model(args=args)
    model.to(device)
    model_male.to(device)
    model_female.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                        factor=0.1, patience=10, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
    #                                                             factor=0.1, patience=10, verbose=True,
    #                                                             threshold=0.0001, threshold_mode='rel',
    #                                                             cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
    #                                                               factor=0.1, patience=10, verbose=True,
    #                                                               threshold=0.0001, threshold_mode='rel',
    #                                                               cooldown=0, min_lr=1e-4, eps=1e-08)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_equal_odd': 0,
        'best_disp_imp': 0,
        'best_acc_parity': 0,
    }

    # THE ENGINE LOOP
    i = 0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        _, _, _ = train_fn_dpsgd_one_batch(dataloader=train_male_loader,
                                           model=model_male,
                                           criterion=criterion,
                                           optimizer=optimizer_male,
                                           device=device,
                                           scheduler=None,
                                           clipping=args.clip,
                                           noise_scale=args.ns)
        _, _, _ = train_fn_dpsgd_one_batch(dataloader=train_female_loader,
                                           model=model_female,
                                           criterion=criterion,
                                           optimizer=optimizer_female,
                                           device=device,
                                           scheduler=None,
                                           clipping=args.clip,
                                           noise_scale=args.ns)
        train_loss, train_out, train_targets = train_fn_dpsgd_one_batch(dataloader=train_loader,
                                                                        model=model,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer,
                                                                        device=device,
                                                                        scheduler=None,
                                                                        clipping=args.clip,
                                                                        noise_scale=args.ns)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=model, device=device)
        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=args.num_val_male,
                                                  num_female=args.num_val_female,
                                                  device=device)
        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=model, device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        # scheduler.step(acc_score)
        # scheduler_male.step(male_acc_score)
        # scheduler_female.step(female_acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['demo_parity'].append(demo_p)
        history['acc_parity'].append(acc_par)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['equal_odd'].append(equal_odd)

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        torch.save(model.state_dict(), args.save_path + model_name)
        # es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)

        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break
    # model.load_state_dict(torch.load(args.save_path + model_name))
    # test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
    # test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    # _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
    #                            model=model, device=device)
    # _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
    #                                   female_loader=valid_female_loader, model=model, device=device)
    # male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
    #                                           female_loader=valid_female_loader,
    #                                           global_model=model,
    #                                           male_model=model_male,
    #                                           female_model=model_female,
    #                                           num_male=len(df_val_mal),
    #                                           num_female=len(df_val_fem),
    #                                           device=device)
    # history['best_test'] = test_acc
    # history['best_demo_parity'] = demo_p
    # history['best_equal_odd'] = equal_odd
    # history['best_disp_imp'] = max(male_norm, female_norm)
    print_history_fair(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_fair_dpsgd_track_grad(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    print(args.n_batch, args.bs_male + args.bs_female)
    print(bound_kl(args=args, num_ep=args.epochs))

    # Defining Model for specific fold
    model_male = init_model(args=args)
    model_female = init_model(args=args)
    global_model = init_model(args=args)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    # scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
    #                                                             factor=0.1, patience=10, verbose=True,
    #                                                             threshold=0.0001, threshold_mode='rel',
    #                                                             cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
    #                                                               factor=0.1, patience=10, verbose=True,
    #                                                               threshold=0.0001, threshold_mode='rel',
    #                                                               cooldown=0, min_lr=1e-4, eps=1e-08)

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
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_disp_imp': 0,
        'best_epoch': 0,
        'empi_bound': []
    }

    # THE ENGINE LOOP
    M = 0.0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        global_dict = global_model.state_dict()
        model_male.load_state_dict(global_dict)
        model_female.load_state_dict(global_dict)

        _, _, _, male_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_male_loader,
                                                                model=model_male,
                                                                criterion=criterion,
                                                                optimizer=optimizer_male,
                                                                device=device,
                                                                scheduler=None,
                                                                clipping=args.clip,
                                                                noise_scale=args.ns)

        _, _, _, female_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_female_loader,
                                                                  model=model_female,
                                                                  criterion=criterion,
                                                                  optimizer=optimizer_female,
                                                                  device=device,
                                                                  scheduler=None,
                                                                  clipping=args.clip,
                                                                  noise_scale=args.ns)
        grad_norm = 0
        for p in global_model.named_parameters():
            grad_norm += (male_par[p[0]] - female_par[p[0]]).norm(p=2) ** 2
        print(grad_norm.item() / (args.clip ** 2))
        M_t = get_Mt(args=args, norm_grad=grad_norm.item())
        M = M_t
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)

        global_model.load_state_dict(global_dict)

        # val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, global_model, criterion, device)
        # val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, global_model, criterion, device)
        train_loss, train_output, train_target = eval_fn(train_loader, global_model, criterion, device)
        valid_loss, valid_output, valid_target = eval_fn(valid_loader, global_model, criterion, device)
        test_loss, test_output, test_target = eval_fn(test_loader, global_model, criterion, device)

        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=global_model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=global_model, device=device)
        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=args.num_val_male,
                                                  num_female=args.num_val_female,
                                                  device=device)

        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=global_model, device=device)
        train_acc = accuracy_score(train_target, np.round(np.array(train_output)))
        val_acc = accuracy_score(valid_target, np.round(np.array(valid_output)))
        test_acc = accuracy_score(test_target, np.round(np.array(test_output)))

        # scheduler_male.step(acc_male_score)
        # scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=valid_loss,
                        Valid_ACC_SCORE=val_acc)

        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(valid_loss)
        history['val_history_acc'].append(val_acc)
        history['demo_parity'].append(demo_p)
        history['equal_odd'].append(equal_odd)
        history['acc_parity'].append(acc_par)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['empi_bound'].append(bound_kl_emp(M))

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        torch.save(global_model.state_dict(), args.save_path + model_name)
        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break
    global_model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, global_model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                               model=global_model, device=device)
    male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                              female_loader=valid_female_loader,
                                              global_model=global_model,
                                              male_model=model_male,
                                              female_model=model_female,
                                              num_male=args.num_val_male,
                                              num_female=args.num_val_female,
                                              device=device)
    history['best_test'] = test_acc
    history['best_demo_parity'] = demo_p
    history['best_disp_imp'] = max(male_norm, female_norm)
    history['best_epoch'] = es.best_epoch
    print_history_track_grad(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_functional_mechanism_logistic_regression(fold, train_df, test_df, male_df, female_df, args, device,
                                                 current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)
    X_train, X_valid, X_test, X_mal, X_fem, X_mal_val, X_fem_val, y_train, y_valid, y_test, y_mal, y_fem, y_mal_val, y_fem_val = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)
    if args.submode == 'func' or args.submode == 'fairdp':
        f_coff_0, f_coff_1, f_coff_2, Q_f = get_coefficient(X=X_fem, y=y_fem, epsilon=args.tar_eps, lbda=args.lamda,
                                                            mode=args.submode)
        m_coff_0, m_coff_1, m_coff_2, Q_m = get_coefficient(X=X_mal, y=y_mal, epsilon=args.tar_eps, lbda=args.lamda,
                                                            mode=args.submode)
    else:
        f_coff_0, f_coff_1, f_coff_2 = get_coefficient(X=X_fem, y=y_fem, epsilon=args.tar_eps, lbda=args.lamda,
                                                       mode=args.submode)
        m_coff_0, m_coff_1, m_coff_2 = get_coefficient(X=X_mal, y=y_mal, epsilon=args.tar_eps, lbda=args.lamda,
                                                       mode=args.submode)
    model_mal = torch.randn((len(args.feature), 1), requires_grad=True).float()
    model_fem = torch.randn((len(args.feature), 1), requires_grad=True).float()

    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'demo_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'acc_imp': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_equal_odd': 0,
        'best_disp_imp': 0,
    }

    # training process
    i = 0
    noise_mal = []
    noise_fem = []

    # tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in range(args.epochs):
        loss_mal = 0
        loss_fem = 0
        if args.submode == 'fair':
            # print(model_mal.requires_grad, model_fem.requires_grad)
            for i in range(args.num_draws):
                noise_m = torch.normal(0, args.ns_, size=model_mal.size(), requires_grad=True).float()
                noise_f = torch.normal(0, args.ns_, size=model_fem.size(), requires_grad=True).float()
                noise_mal.append(noise_m)
                noise_fem.append(noise_f)
                loss_m = update_one_step(args=args, model=model_mal, model_=model_fem,
                                         coff=(m_coff_0, m_coff_1, m_coff_2),
                                         Q=None, Q_=None, noise=noise_m)
                loss_f = update_one_step(args=args, model=model_fem, model_=model_mal,
                                         coff=(f_coff_0, f_coff_1, f_coff_2),
                                         Q=None, Q_=None, noise=noise_f)
                loss_mal += loss_m
                loss_fem += loss_f

        elif args.submode == 'fairdp':

            for i in range(args.num_draws):
                noise_m = torch.normal(0, args.ns_, size=model_mal.size(), requires_grad=False).float()
                noise_f = torch.normal(0, args.ns_, size=model_fem.size(), requires_grad=False).float()
                noise_mal.append(noise_m)
                noise_fem.append(noise_f)
                loss_m = update_one_step(args=args, model=model_mal, model_=model_fem,
                                         coff=(m_coff_0, m_coff_1, m_coff_2),
                                         Q=Q_m, Q_=Q_f, noise=noise_m)
                loss_f = update_one_step(args=args, model=model_fem, model_=model_mal,
                                         coff=(f_coff_0, f_coff_1, f_coff_2),
                                         Q=Q_f, Q_=Q_m, noise=noise_f)
                loss_mal += loss_m
                loss_fem += loss_f

        elif args.submode == 'func':
            loss_m = update_one_step(args=args, model=model_mal, model_=model_fem,
                                     coff=(m_coff_0, m_coff_1, m_coff_2),
                                     Q=Q_m, Q_=None, noise=None)
            loss_f = update_one_step(args=args, model=model_fem, model_=model_mal,
                                     coff=(f_coff_0, f_coff_1, f_coff_2),
                                     Q=Q_f, Q_=None, noise=None)
            loss_mal = loss_m
            loss_fem = loss_f

        elif args.submode == 'torch':
            loss_m = update_one_step(args=args, model=model_mal, model_=model_fem,
                                     coff=(m_coff_0, m_coff_1, m_coff_2),
                                     Q=None, Q_=None, noise=None)
            loss_f = update_one_step(args=args, model=model_fem, model_=model_mal,
                                     coff=(f_coff_0, f_coff_1, f_coff_2),
                                     Q=None, Q_=None, noise=None)
            loss_mal = loss_m
            loss_fem = loss_f

        model_mal = model_mal - args.lr * model_mal.grad
        model_fem = model_fem - args.lr * model_fem.grad
        global_model = (model_mal + model_fem) / 2

        if args.submode == 'func' or args.submode == 'torch':
            train_acc, train_loss, _ = fair_evaluate(args=args, model=global_model, noise=None, X=X_train, y=y_train)
            valid_acc, valid_loss, _ = fair_evaluate(args=args, model=global_model, noise=None, X=X_valid, y=y_valid)
            test_acc, test_loss, _ = fair_evaluate(args=args, model=global_model, noise=None, X=X_test, y=y_test)
            male_acc, _, male_pred, male_tpr, male_prob = fair_evaluate(args=args, model=global_model, noise=None, X=X_mal_val,
                                                                 y=y_mal_val, fair=True)
            female_acc, _, female_pred, female_tpr, female_prob = fair_evaluate(args=args, model=global_model, noise=None,
                                                                       X=X_fem_val,
                                                                       y=y_fem_val, fair=True)
            _, _, male_pred_m = fair_evaluate(args=args, model=model_mal, noise=None, X=X_mal_val,
                                              y=y_mal_val, fair=False)
            _, _, female_pred_f = fair_evaluate(args=args, model=model_fem, noise=None, X=X_fem_val,
                                                y=y_fem_val, fair=False)

            male_norm = torch.norm(male_pred - male_pred_m, p=1).item() / male_pred.size(0)
            female_norm = torch.norm(female_pred - female_pred_f, p=1).item() / female_pred.size(0)

        else:
            acc, loss, _, tpr, prob, norm = fair_evaluate(args=args, model=(global_model, model_mal, model_fem),
                                                          noise=(noise_mal, noise_fem),
                                                          X=(X_train, X_valid, X_test, X_mal_val, X_fem_val),
                                                          y=(y_train, y_valid, y_test, y_mal_val, y_fem_val))
            train_acc, valid_acc, test_acc, male_acc, female_acc = acc
            train_loss, valid_loss, test_loss = loss
            male_tpr, female_tpr = tpr
            male_prob, female_prob = prob
            male_norm, female_norm = norm

        model_mal.grad = torch.zeros(model_mal.size())
        model_fem.grad = torch.zeros(model_fem.size())

        print("Epoch {}: train loss {}, train acc {}, valid loss {}, valid acc {}, loss on male {}, female {}".format(
            epoch, train_loss, train_acc,
            valid_loss, valid_acc, loss_mal, loss_fem))

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(valid_loss)
        history['val_history_acc'].append(valid_acc)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['demo_parity'].append(np.abs(male_prob - female_prob))
        history['equal_odd'].append(np.abs(male_tpr - female_tpr))
        history['disp_imp'].append(max(male_norm, female_norm))
        history['acc_imp'].append(np.abs(male_acc - female_acc))

        torch.save(global_model, args.save_path + model_name)
        torch.save(model_mal, args.save_path + 'male_{}'.format(model_name))
        torch.save(model_fem, args.save_path + 'female_{}'.format(model_name))

    print_history_func(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd_test(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    print(args.n_batch, args.bs_male, args.bs_female)
    # print(bound_kl(args=args, num_ep=args.epochs))

    # Defining Model for specific fold
    model_male = init_model(args=args)
    model_female = init_model(args=args)
    global_model = init_model(args=args)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

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
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_disp_imp': 0,
        'best_epoch': 0,
        'empi_bound': []
    }

    # THE ENGINE LOOP
    M = 0.0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        global_dict = global_model.state_dict()
        model_male.load_state_dict(global_dict)
        model_female.load_state_dict(global_dict)

        _, _, _, male_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_male_loader,
                                                                model=model_male,
                                                                criterion=criterion,
                                                                optimizer=optimizer_male,
                                                                device=device,
                                                                scheduler=None,
                                                                clipping=args.clip,
                                                                noise_scale=args.ns)

        _, _, _, female_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_female_loader,
                                                                  model=model_female,
                                                                  criterion=criterion,
                                                                  optimizer=optimizer_female,
                                                                  device=device,
                                                                  scheduler=None,
                                                                  clipping=args.clip,
                                                                  noise_scale=args.ns)
        grad_norm = 0
        for p in global_model.named_parameters():
            grad_norm += (male_par[p[0]] - female_par[p[0]]).norm(p=2) ** 2

        M_t = get_Mt(args=args, norm_grad=grad_norm.item())
        M = M_t
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)

        global_model.load_state_dict(global_dict)

        # val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, global_model, criterion, device)
        # val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, global_model, criterion, device)
        train_loss, train_output, train_target = eval_fn(train_loader, global_model, criterion, device)
        valid_loss, valid_output, valid_target = eval_fn(valid_loader, global_model, criterion, device)
        test_loss, test_output, test_target = eval_fn(test_loader, global_model, criterion, device)

        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=global_model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=global_model, device=device)
        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=args.num_val_male,
                                                  num_female=args.num_val_female,
                                                  device=device)

        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=global_model, device=device)
        train_acc = accuracy_score(train_target, np.round(np.array(train_output)))
        val_acc = accuracy_score(valid_target, np.round(np.array(valid_output)))
        test_acc = accuracy_score(test_target, np.round(np.array(test_output)))

        # scheduler_male.step(acc_male_score)
        # scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=valid_loss,
                        Valid_ACC_SCORE=val_acc)

        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(valid_loss)
        history['val_history_acc'].append(val_acc)
        history['demo_parity'].append(demo_p)
        history['equal_odd'].append(equal_odd)
        history['acc_parity'].append(acc_par)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['empi_bound'].append(bound_kl_emp(M))

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        torch.save(global_model.state_dict(), args.save_path + model_name)
        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break
    global_model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, global_model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                               model=global_model, device=device)
    male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                              female_loader=valid_female_loader,
                                              global_model=global_model,
                                              male_model=model_male,
                                              female_model=model_female,
                                              num_male=args.num_val_male,
                                              num_female=args.num_val_female,
                                              device=device)
    history['best_test'] = test_acc
    history['best_demo_parity'] = demo_p
    history['best_disp_imp'] = max(male_norm, female_norm)
    history['best_epoch'] = es.best_epoch
    print_history_track_grad(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_smooth(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)
    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    model_male = init_model(args)
    model_female = init_model(args)
    global_model = init_model(args)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)

    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                        factor=0.1, patience=20, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=5e-4, eps=1e-08)

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
        'disp_imp': [],
        'norm_model': [],
        'best_test': 0
    }
    global_dict = global_model.state_dict()
    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        _, _, _ = train_smooth_classifier(dataloader=train_male_loader, model=model_male, model_=model_female,
                                          criterion=criterion, optimizer=optimizer_male, device=device,
                                          scheduler=None, num_draws=args.num_draws)
        _, _, _ = train_smooth_classifier(dataloader=train_female_loader, model=model_female, model_=model_male,
                                          criterion=criterion, optimizer=optimizer_female, device=device,
                                          scheduler=None, num_draws=args.num_draws)

        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)
        global_model.load_state_dict(global_dict)

        train_loss, train_outputs, train_targets = eval_smooth_classifier(train_loader, global_model, criterion, device,
                                                                          num_draws=args.num_draws)
        val_loss, outputs, targets = eval_smooth_classifier(valid_loader, global_model, criterion, device,
                                                            num_draws=args.num_draws)
        test_loss, test_outputs, test_targets = eval_smooth_classifier(test_loader, global_model, criterion, device,
                                                                       num_draws=args.num_draws)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_outputs)))
        test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        male_norm, female_norm = disperate_impact_smooth(male_loader=valid_male_loader,
                                                         female_loader=valid_female_loader,
                                                         global_model=global_model,
                                                         male_model=model_male,
                                                         female_model=model_female,
                                                         num_male=args.num_val_male,
                                                         num_female=args.num_val_female,
                                                         device=device, num_draws=args.num_draws)
        # scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)
        params_ = model_female.state_dict()
        l2_norm = 0.0
        for p in model_male.named_parameters():
            l2_norm += torch.norm(p[1] - params_[p[0]], p=2) ** 2

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['norm_model'].append(np.sqrt(l2_norm.item()))

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        es(epoch=epoch, epoch_score=acc_score, model=global_model, model_path=args.save_path + model_name)

        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    print_history_fair(fold, history, epoch + 1, args, current_time)
    global_model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_smooth_classifier(test_loader, model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    history['best_test'] = test_acc
    save_res(fold=fold, args=args, dct=history, current_time=current_time)


def run_fair_dpsgd_track_grad_baseline(fold, train_df, test_df, male_df, female_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)

    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df)

    print(args.n_batch, args.bs_male + args.bs_female)
    print(bound_kl(args=args, num_ep=args.epochs))

    # Defining Model for specific fold
    model_male = init_model(args=args)
    model_female = init_model(args=args)
    global_model = init_model(args=args)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.lr)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.lr)

    # Defining LR SCheduler
    # scheduler_male = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_male, mode='max',
    #                                                             factor=0.1, patience=10, verbose=True,
    #                                                             threshold=0.0001, threshold_mode='rel',
    #                                                             cooldown=0, min_lr=1e-4, eps=1e-08)
    # scheduler_female = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_female, mode='max',
    #                                                               factor=0.1, patience=10, verbose=True,
    #                                                               threshold=0.0001, threshold_mode='rel',
    #                                                               cooldown=0, min_lr=1e-4, eps=1e-08)

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
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'disp_imp': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_disp_imp': 0,
        'best_epoch': 0,
        'empi_bound': []
    }

    # THE ENGINE LOOP
    M = 0.0
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        global_dict = global_model.state_dict()
        # model_male.load_state_dict(global_dict)
        # model_female.load_state_dict(global_dict)

        _, _, _, male_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_male_loader,
                                                                model=model_male,
                                                                criterion=criterion,
                                                                optimizer=optimizer_male,
                                                                device=device,
                                                                scheduler=None,
                                                                clipping=args.clip,
                                                                noise_scale=args.ns)

        _, _, _, female_par = train_fn_dpsgd_one_batch_track_grad(dataloader=train_female_loader,
                                                                  model=model_female,
                                                                  criterion=criterion,
                                                                  optimizer=optimizer_female,
                                                                  device=device,
                                                                  scheduler=None,
                                                                  clipping=args.clip,
                                                                  noise_scale=args.ns)
        grad_norm = 0
        for p in global_model.named_parameters():
            grad_norm += (male_par[p[0]] - female_par[p[0]]).norm(p=2) ** 2
        # print(grad_norm.item() / (args.clip ** 2))
        M_t = get_Mt(args=args, norm_grad=grad_norm.item())
        M = M_t
        male_dict = model_male.state_dict()
        female_dict = model_female.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.div(deepcopy(male_dict[key]) + deepcopy(female_dict[key]), 2)

        global_model.load_state_dict(global_dict)

        # val_male_loss, outputs_male, targets_male = eval_fn(valid_male_loader, global_model, criterion, device)
        # val_female_loss, outputs_female, targets_female = eval_fn(valid_female_loader, global_model, criterion, device)
        train_loss, train_output, train_target = eval_fn(train_loader, global_model, criterion, device)
        valid_loss, valid_output, valid_target = eval_fn(valid_loader, global_model, criterion, device)
        test_loss, test_output, test_target = eval_fn(test_loader, global_model, criterion, device)

        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=global_model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=global_model, device=device)
        male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                                  female_loader=valid_female_loader,
                                                  global_model=global_model,
                                                  male_model=model_male,
                                                  female_model=model_female,
                                                  num_male=args.num_val_male,
                                                  num_female=args.num_val_female,
                                                  device=device)

        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=global_model, device=device)
        train_acc = accuracy_score(train_target, np.round(np.array(train_output)))
        val_acc = accuracy_score(valid_target, np.round(np.array(valid_output)))
        test_acc = accuracy_score(test_target, np.round(np.array(test_output)))

        # scheduler_male.step(acc_male_score)
        # scheduler_female.step(acc_female_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=valid_loss,
                        Valid_ACC_SCORE=val_acc)

        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(valid_loss)
        history['val_history_acc'].append(val_acc)
        history['demo_parity'].append(demo_p)
        history['equal_odd'].append(equal_odd)
        history['acc_parity'].append(acc_par)
        history['disp_imp'].append(max(male_norm, female_norm))
        history['empi_bound'].append(bound_kl_emp(M))

        torch.save(model_male.state_dict(), args.save_path + 'male_{}'.format(model_name))
        torch.save(model_female.state_dict(), args.save_path + 'female_{}'.format(model_name))
        torch.save(global_model.state_dict(), args.save_path + model_name)
        #
        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break
    global_model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(test_loader, global_model, criterion, device)
    test_acc = accuracy_score(test_targets, np.round(np.array(test_outputs)))
    _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                               model=global_model, device=device)
    male_norm, female_norm = disperate_impact(male_loader=valid_male_loader,
                                              female_loader=valid_female_loader,
                                              global_model=global_model,
                                              male_model=model_male,
                                              female_model=model_female,
                                              num_male=args.num_val_male,
                                              num_female=args.num_val_female,
                                              device=device)
    history['best_test'] = test_acc
    history['best_demo_parity'] = demo_p
    history['best_disp_imp'] = max(male_norm, female_norm)
    history['best_epoch'] = es.best_epoch
    print_history_track_grad(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)