import torch
from Data.datasets import Adult
from torch.utils.data import DataLoader
from Model.models import NeuralNetwork, NormNN
from train_eval import *
from plottings import *
from sklearn.metrics import accuracy_score
from metrics import *
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

def run_clean(fold, df, args, device, current_time):
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)

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
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)

        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)

        es(acc_score, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold, history, num_epochs=epoch + 1)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_dpsgd(fold, df, args, device, current_time):
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR)

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
    }

    # THE ENGINE LOOP
    i = 0
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn_dpsgd(train_loader, model, criterion, optimizer, device,
                                                              scheduler=None, clipping=args.clip,
                                                              noise_scale=args.ns)
        # train_fn_dpsgd(train_loader, model,criterion, optimizer, device,scheduler=None,epoch=epoch, clipping=args.clip, noise_scale=args.ns)
        # return
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)

        es(acc_score, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold, history, num_epochs=epoch + 1)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_fair(fold, male_df, female_df, args, device, current_time):
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
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_male_loader = DataLoader(
        valid_male_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = DataLoader(
        valid_female_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)

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
        'equal_odd': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader, model=model, device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
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

        es(acc_score, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair(fold, history, num_epochs=epoch + 1)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd(fold, male_df, female_df, args, device, current_time):
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
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Device

    # Defining Model for specific fold
    model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)

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
        'equal_odd': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn_dpsgd(train_loader, model, criterion, optimizer, device,
                                                              scheduler=None, clipping=args.clip,
                                                              noise_scale=args.ns)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        prob_male, prob_female, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                                     model=model, device=device)
        male_tpr, female_tpr, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                                          female_loader=valid_female_loader, model=model, device=device)
        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
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

        es(acc_score, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair(fold, history, num_epochs=epoch + 1)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_norm(fold, df, args, device, current_time):
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    # Defining DataSet
    train_dataset = Adult(
        df_train[args.feature].values,
        df_train[args.target].values
    )

    valid_dataset = Adult(
        df_valid[args.feature].values,
        df_valid[args.target].values
    )

    # Defining DataLoader with BalanceClass Sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model = NormNN(args.input_dim, args.hidden_dim, args.output_dim)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)

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
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)

        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)

        train_acc = accuracy_score(train_targets, np.round(np.array(train_out)))
        acc_score = accuracy_score(targets, np.round(np.array(outputs)))

        scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)

        es(acc_score, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history(fold, history, num_epochs=epoch + 1)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_fair_dpsgd_alg2(fold, male_df, female_df, args, device, current_time):
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

    # Defining DataLoader with BalanceClass Sampler
    sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        sampler=sampler_male,
        num_workers=0
    )

    sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        sampler=sampler_female,
        num_workers=0
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model_male = NormNN(args.input_dim, args.hidden_dim, args.output_dim)
    model_female = NormNN(args.input_dim, args.hidden_dim, args.output_dim)
    global_model = NormNN(args.input_dim, args.hidden_dim, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.LR)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.LR)

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
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
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
        history['prob_male'].append(prob_male)
        history['prob_female'].append(prob_female)
        history['demo_parity'].append(demo_p)
        history['male_tpr'].append(male_tpr)
        history['female_tpr'].append(female_tpr)
        history['equal_odd'].append(equal_odd)
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        # es(acc_score,model,f'model_{fold}.bin')

        # if es.early_stop:
        #     print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
        #     break

    print_history_fair_(fold, history, num_epochs=epoch + 1, args=args)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_fair_v2(fold, male_df, female_df, args, device, current_time):
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
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model_male = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model_female = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    global_model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.LR)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.LR)
    optimizer_global = torch.optim.Adam(global_model.parameters(), lr=args.LR)

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
    es = EarlyStopping(patience=args.patience, verbose=False)

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
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
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
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        es(acc_global_score, global_model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair_v2(fold, history, num_epochs=epoch + 1, args=args)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)

def run_fair_v3(fold, male_df, female_df, args, device, current_time):
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
    #     batch_size=args.BATCH_SIZE,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=0
    # )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Model for specific fold
    model_male = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    model_female = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)
    global_model = NeuralNetwork(args.input_dim, args.hidden_dim, args.output_dim)

    model_male.to(device)
    model_female.to(device)
    global_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer_male = torch.optim.Adam(model_male.parameters(), lr=args.LR)
    optimizer_female = torch.optim.Adam(model_female.parameters(), lr=args.LR)
    # optimizer_global = torch.optim.Adam(global_model.parameters(), lr=args.LR)

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
        'male_norm': [],
        'female_norm': []
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.EPOCHS), total=args.EPOCHS)
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
        history['male_norm'].append(male_norm)
        history['female_norm'].append(female_norm)

        es(acc_global_score, global_model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    print_history_fair_v3(fold, history, num_epochs=epoch + 1, args=args)
    save_res(folds=fold, args=args, dct=history, current_time=current_time)