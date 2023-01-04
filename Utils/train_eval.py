import numpy as np
import torch
from Utils.utils import get_gaussian_noise
from opacus.utils.batch_memory_manager import BatchMemoryManager
from copy import deepcopy
from Utils.utils import logloss
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep = 100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        if self.run_mode == 'func' and epoch < self.skip_ep:
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score

class ReduceOnPlatau:
    def __init__(self, mode="max", delta=1e-4, verbose=False, args=None, min_lr = 5e-5):
        self.patience = args.lr_patience
        self.counter = 0
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.args = args
        self.min_lr = min_lr
        self.step = args.lr_step
        self.best_score = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                if self.args.lr - self.step < self.min_lr:
                    self.args.lr = self.min_lr
                else:
                    self.args.lr -= self.step
                print("Reduce learning rate to {}".format(self.args.lr))
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0
        return self.args

def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()

        output = model(features)
        output = torch.squeeze(output)

        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets

def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    # num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            # num_data_point += features.size(dim=0)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss, fin_outputs, fin_targets

def train_fn_dpsgd(dataloader, model, criterion, optimizer, device, scheduler, clipping, noise_scale):
    model.to(device)
    model.train()
    noise_std = get_gaussian_noise(clipping, noise_scale)
    train_targets = []
    train_outputs = []
    train_loss = 0
    num_data_point = 0
    for bi, d in enumerate(dataloader):
        features, target, _ = d
        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        optimizer.zero_grad()
        temp_par = {}
        for p in model.named_parameters():
            temp_par[p[0]] = torch.zeros_like(p[1])
        bz = features.size(dim=0)
        for i in range(bz):
            for p in model.named_parameters():
                p[1].grad = torch.zeros_like(p[1])
            feat = features[i]
            targ = target[i]
            output = model(feat)
            output = torch.squeeze(output)
            loss = criterion(output, targ)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping, norm_type=2)
            # total_l2_norm = 0
            for p in model.named_parameters():
                # total_l2_norm += p[1].grad.detach().norm(p=2)**2
                temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
            # print(np.sqrt(total_l2_norm) <= clipping ,np.sqrt(total_l2_norm), clipping)
            output = output.cpu().detach().numpy()
            train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
            train_outputs.append(output)
            # model.zero_grad()
            num_data_point += 1

        for p in model.named_parameters():
            p[1].grad = deepcopy(temp_par[p[0]]) + torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(
                device)
            p[1].grad = p[1].grad / bz
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return train_loss / num_data_point, train_outputs, train_targets

def train_fn_dpsgd_one_batch_track_grad(dataloader, model, criterion, optimizer, device, scheduler, clipping, noise_scale):
    model.to(device)
    model.train()
    noise_std = get_gaussian_noise(clipping, noise_scale)
    train_targets = []
    train_outputs = []
    train_loss = 0
    num_data_point = 0
    # for bi, d in enumerate(dataloader):
    features, target, _ = next(iter(dataloader))
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    optimizer.zero_grad()
    temp_par = {}
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
    bz = features.size(dim=0)
    for i in range(bz):
        for p in model.named_parameters():
            p[1].grad = torch.zeros_like(p[1])
        feat = features[i]
        targ = target[i]
        output = model(feat)
        output = torch.squeeze(output)
        loss = criterion(output, targ)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping, norm_type=2)
        total_l2_norm = 0
        for p in model.named_parameters():
            total_l2_norm += p[1].grad.detach().norm(p=2) ** 2
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
        # print(np.sqrt(total_l2_norm) <= clipping ,np.sqrt(total_l2_norm), clipping)
        output = output.cpu().detach().numpy()
        train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
        train_outputs.append(output)
        # model.zero_grad()
        num_data_point += 1

    for p in model.named_parameters():
        p[1].grad = deepcopy(temp_par[p[0]]) + torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(
            device)
        p[1].grad = p[1].grad / bz
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return train_loss / num_data_point, train_outputs, train_targets, temp_par

def update_one_step(args, model, model_, coff, Q, Q_, noise):
    if args.submode == 'func':
        coff_0, coff_1, coff_2 = coff
        Q = torch.from_numpy(Q)
        loss = coff_0 + torch.mm(torch.mm(Q, model).T, coff_1) + torch.mm(
            torch.mm(torch.mm(Q, model).T.float(), coff_2.float()), torch.mm(Q, model))
        model.retain_grad()
        loss.backward()
    elif args.submode == 'torch':
        coff_0, coff_1, coff_2 = coff
        loss = coff_0 + torch.mm(model.T, coff_1) + torch.mm(
            torch.mm(model.T.float(), coff_2.float()), model)
        model.retain_grad()
        loss.backward()
    elif args.submode == 'fairdp':
        coff_0, coff_1, coff_2 = coff
        Q = torch.from_numpy(Q)
        # print(Q.size(), Q_.size(), model.size(), model_.size())
        loss = (1 / args.num_draws) * (coff_0 + torch.mm(torch.mm(Q, model + noise).T, coff_1) + torch.mm(
            torch.mm(torch.mm(Q, model + noise).T.float(), coff_2.float()),
            torch.mm(Q, model + noise))) + (1 / args.num_draws) * args.alpha * torch.norm(
            model-model_, p=2)
        model.retain_grad()
        loss.backward()
    elif args.submode == 'func_org':
        coff_0, coff_1, coff_2 = coff
        Q = torch.from_numpy(Q)
        loss = coff_0 + torch.mm(torch.mm(Q, model).T, coff_1) + torch.mm(
            torch.mm(torch.mm(Q, model).T.float(), coff_2.float()), torch.mm(Q, model))
        model.retain_grad()
        loss.backward()
    elif args.submode == 'fair':
        coff_0, coff_1, coff_2 = coff
        # print(model.requires_grad)
        loss = (1 / args.num_draws) * (
                coff_0 + torch.mm((model + noise).T, coff_1) + torch.mm(torch.mm((model + noise).T, coff_2),
                                                                        (model + noise))) + (
                       1 / args.num_draws) * args.alpha * torch.norm(
            model - model_, p=2)
        model.retain_grad()
        loss.backward()
    # print(model.grad)
    return loss.item()

def fair_evaluate(args, model, noise, X, y, fair=False):
    if args.submode == 'func' or args.submode == 'torch' or args.submode == 'func_org':
        pred = torch.sigmoid(torch.mm(torch.from_numpy(X.astype(np.float32)), model))
        loss = logloss(y=y, pred=pred.detach().numpy())
        acc = accuracy_score(y_true=y, y_pred=np.round(pred.detach().numpy()))
        if fair:
            tn, fp, fn, tp = confusion_matrix(y, np.round(pred.detach().numpy())).ravel()
            tpr = tp / (tp + fn)
            prob = np.sum(np.round(pred.detach().numpy())) / pred.shape[0]
            return acc, loss, pred, tpr, prob
        else:
            return acc, loss, pred
    else:
        noise_m, noise_f = noise
        X_train, X_valid, X_test, X_mal, X_fem = X
        y_train, y_valid, y_test, y_mal, y_fem = y
        pred_tr, pred_va, pred_te, pred_m, pred_f, pred_mg, pred_fg = (0.0 for i in range(7))
        model_m, model_f, model_g = model
        for i in range(args.num_draws):
            temp_g = model_g + (1/2)*(noise_m[i] + noise_f[i])
            temp_m = model_m + noise_m[i]
            temp_f = model_f + noise_f[i]
            pred_tr = pred_tr + torch.sigmoid(torch.mm(torch.from_numpy(X_train.astype(np.float32)), temp_g))
            pred_va = pred_va + torch.sigmoid(torch.mm(torch.from_numpy(X_valid.astype(np.float32)), temp_g))
            pred_te = pred_te + torch.sigmoid(torch.mm(torch.from_numpy(X_test.astype(np.float32)), temp_g))
            pred_mg = pred_mg + torch.sigmoid(torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_g))
            pred_fg = pred_fg + torch.sigmoid(torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_g))
            pred_m = pred_m + torch.sigmoid(torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp_m))
            pred_f = pred_f + torch.sigmoid(torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp_f))
        pred_tr = (1 / args.num_draws) * pred_tr
        pred_va = (1 / args.num_draws) * pred_va
        pred_te = (1 / args.num_draws) * pred_te
        pred_m = (1 / args.num_draws) * pred_m
        pred_f = (1 / args.num_draws) * pred_f
        pred_mg = (1 / args.num_draws) * pred_mg
        pred_fg = (1 / args.num_draws) * pred_fg

        acc_tr = accuracy_score(y_true=y_train, y_pred=np.round(pred_tr.detach().numpy()))
        acc_va = accuracy_score(y_true=y_valid, y_pred=np.round(pred_va.detach().numpy()))
        acc_te = accuracy_score(y_true=y_test, y_pred=np.round(pred_te.detach().numpy()))
        loss_tr = logloss(y=y_train, pred=pred_tr.detach().numpy())
        loss_va = logloss(y=y_valid, pred=pred_va.detach().numpy())
        loss_te = logloss(y=y_test, pred=pred_te.detach().numpy())

        tn, fp, fn, tp = confusion_matrix(y_mal, np.round(pred_mg.detach().numpy())).ravel()
        male_tpr = tp / (tp + fn)
        male_prob = np.sum(np.round(pred_mg.detach().numpy())) / pred_mg.shape[0]

        tn, fp, fn, tp = confusion_matrix(y_fem, np.round(pred_fg.detach().numpy())).ravel()
        female_tpr = tp / (tp + fn)
        female_prob = np.sum(np.round(pred_fg.detach().numpy())) / pred_fg.shape[0]

        male_norm = torch.norm(pred_mg - pred_m, p=2).item()/pred_m.size(0)
        female_norm = torch.norm(pred_fg - pred_f, p=2).item()/pred_f.size(0)
        norm = (male_norm, female_norm)
        return (acc_tr, acc_va, acc_te), (loss_tr, loss_va, loss_te), (pred_tr, pred_va, pred_te, pred_m, pred_f), (
        male_tpr, female_tpr), (male_prob, female_prob), norm

def train_smooth_classifier(dataloader, model, model_, criterion, optimizer, device, scheduler, num_draws):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0
    params_ = model_.state_dict()

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()
        output = 0.0
        for i in range(num_draws):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.normal(mean=0.0, std=1.0, size=p.size(), requires_grad=False))
            output = output + model(features)
        output = output/num_draws
        output = torch.squeeze(output)
        l2_norm = 0.0
        for p in model.named_parameters():
            l2_norm += torch.norm(p[1] - params_[p[0]], p=2)
        loss = criterion(output, target) + l2_norm
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets

def eval_smooth_classifier(data_loader, model, criterion, device, num_draws):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    # num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            # num_data_point += features.size(dim=0)
            outputs = 0.0
            for i in range(num_draws):
                for p in model.parameters():
                    p.add_(torch.normal(mean=0.0, std=1.0, size=p.size(), requires_grad=False))
                outputs = outputs + model(features)
            outputs = outputs / num_draws
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss, fin_outputs, fin_targets