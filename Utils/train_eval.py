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


def train_fn_dpsgd_one_batch(dataloader, model, criterion, optimizer, device, scheduler, clipping, noise_scale):
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


def train_fn_track_grad(dataloader, dataloader_, model, criterion, optimizer, device, scheduler, clipping, noise_scale):
    model.to(device)
    model.train()
    noise_std = get_gaussian_noise(clipping, noise_scale)
    train_targets = []
    train_outputs = []
    male_norm = []
    female_norm = []
    train_loss = 0
    num_data_point = 0
    # for bi, d in enumerate(dataloader):
    features, target, ismale = next(iter(dataloader))
    features_, target_, ismale_ = next(iter(dataloader_))
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    features_ = features_.to(device, dtype=torch.float)
    target_ = target_.to(device, dtype=torch.float)
    optimizer.zero_grad()
    temp_par = {}
    male_par = {}
    female_par = {}
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
        male_par[p[0]] = torch.zeros_like(p[1])
        female_par[p[0]] = torch.zeros_like(p[1])
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
            # total_l2_norm += p[1].grad.detach().norm(p=2) ** 2
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
            male_par[p[0]] = male_par[p[0]] + deepcopy(p[1].grad)
        # print(np.sqrt(total_l2_norm) <= clipping ,np.sqrt(total_l2_norm), clipping)
        # if ismale[i].item() == 1:
        #     male_norm.append(np.sqrt(total_l2_norm.item()))
        # else:
        #     female_norm.append(np.sqrt(total_l2_norm.item()))
        output = output.cpu().detach().numpy()
        train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
        train_outputs.append(output)
        # model.zero_grad()
        num_data_point += 1

    bz = features_.size(dim=0)
    for i in range(bz):
        for p in model.named_parameters():
            p[1].grad = torch.zeros_like(p[1])
        feat = features_[i]
        targ = target_[i]
        output = model(feat)
        output = torch.squeeze(output)
        loss = criterion(output, targ)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping, norm_type=2)
        # total_l2_norm = 0
        for p in model.named_parameters():
            # total_l2_norm += p[1].grad.detach().norm(p=2) ** 2
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
            female_par[p[0]] = female_par[p[0]] + deepcopy(p[1].grad)
        # print(np.sqrt(total_l2_norm) <= clipping ,np.sqrt(total_l2_norm), clipping)
        # if ismale_[i].item() == 1:
        #     male_norm.append(np.sqrt(total_l2_norm.item()))
        # else:
        #     female_norm.append(np.sqrt(total_l2_norm.item()))
        output = output.cpu().detach().numpy()
        train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
        train_outputs.append(output)
        # model.zero_grad()
        num_data_point += 1

    grad_norm = 0
    for p in model.named_parameters():
        grad_norm += (male_par[p[0]] - female_par[p[0]]).norm(p=2)**2

    for p in model.named_parameters():
        p[1].grad = deepcopy(temp_par[p[0]]) + torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(
            device)
        p[1].grad = p[1].grad / num_data_point
    optimizer.step()

    # male_norm = np.array(male_norm)
    # female_norm = np.array(female_norm)

    if scheduler is not None:
        scheduler.step()

    return train_loss / num_data_point, train_outputs, train_targets, grad_norm


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
    if args.submode == 'func' or args.submode == 'torch':
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
        pred_tr, pred_va, pred_te, pred_m, pred_f = (0.0 for i in range(5))
        for i in range(args.num_draws):
            temp = model + (1/2)*(noise_m[i] + noise_f[i])
            pred_tr = pred_tr + torch.sigmoid(torch.mm(torch.from_numpy(X_train.astype(np.float32)), temp))
            pred_va = pred_va + torch.sigmoid(torch.mm(torch.from_numpy(X_valid.astype(np.float32)), temp))
            pred_te = pred_te + torch.sigmoid(torch.mm(torch.from_numpy(X_test.astype(np.float32)), temp))
            pred_m = pred_m + torch.sigmoid(torch.mm(torch.from_numpy(X_mal.astype(np.float32)), temp))
            pred_f = pred_f + torch.sigmoid(torch.mm(torch.from_numpy(X_fem.astype(np.float32)), temp))
        pred_tr = (1 / args.num_draws) * pred_tr
        pred_va = (1 / args.num_draws) * pred_va
        pred_te = (1 / args.num_draws) * pred_te
        pred_m = (1 / args.num_draws) * pred_m
        pred_f = (1 / args.num_draws) * pred_f
        acc_tr = accuracy_score(y_true=y_train, y_pred=np.round(pred_tr.detach().numpy()))
        acc_va = accuracy_score(y_true=y_valid, y_pred=np.round(pred_va.detach().numpy()))
        acc_te = accuracy_score(y_true=y_test, y_pred=np.round(pred_te.detach().numpy()))
        loss_tr = logloss(y=y_train, pred=pred_tr.detach().numpy())
        loss_va = logloss(y=y_valid, pred=pred_va.detach().numpy())
        loss_te = logloss(y=y_test, pred=pred_te.detach().numpy())

        tn, fp, fn, tp = confusion_matrix(y_mal, np.round(pred_m.detach().numpy())).ravel()
        male_tpr = tp / (tp + fn)
        male_prob = np.sum(np.round(pred_m.detach().numpy())) / pred_m.shape[0]

        tn, fp, fn, tp = confusion_matrix(y_fem, np.round(pred_f.detach().numpy())).ravel()
        female_tpr = tp / (tp + fn)
        female_prob = np.sum(np.round(pred_f.detach().numpy())) / pred_f.shape[0]

        return (acc_tr, acc_va, acc_te), (loss_tr, loss_va, loss_te), (pred_tr, pred_va, pred_te, pred_m, pred_f), (
        male_tpr, female_tpr), (male_prob, female_prob)

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

# def train_opacus(dataloader, model, criterion, optimizer, device, args):
#     model.to(device)
#     model.train()
#     train_targets = []
#     train_outputs = []
#     train_loss = 0
#
#     with BatchMemoryManager(
#             data_loader=dataloader,
#             max_physical_batch_size=args.MAX_PHYSICAL_BATCH_SIZE,
#             optimizer=optimizer
#     ) as memory_safe_data_loader:
#         for bi, d in enumerate(memory_safe_data_loader):
#             optimizer.zero_grad()
#             features, target = d
#             features = features.to(device)
#             target = target.to(device)
#
#             # compute output
#             output = model(features)
#             output = torch.squeeze(output)
#             loss = criterion(output, target)
#             loss.backward()
#
#             output = output.cpu().detach().numpy()
#             train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
#             train_outputs.extend(output)
#
#             train_loss += loss.item()
#             optimizer.step()
#     return train_loss, train_outputs, train_targets
#
# def train_fn_dpsgd_without_optimizer(dataloader, model, criterion, device, scheduler, clipping, noise_scale, lr):
#     model.to(device)
#     model.train()
#     noise_std = get_gaussian_noise(clipping, noise_scale)
#     train_targets = []
#     train_outputs = []
#     train_loss = 0
#     num_data_point = 0
#     for bi, d in enumerate(dataloader):
#         features, target = d
#         features = features.to(device, dtype=torch.float)
#         target = target.to(device, dtype=torch.float)
#         temp_par = {}
#         for p in model.named_parameters():
#             temp_par[p[0]] = torch.zeros_like(p[1])
#         bz = features.size(dim=0)
#         for i in range(bz):
#             for p in model.named_parameters():
#                 p[1].grad = torch.zeros_like(p[1])
#             feat = features[i]
#             targ = target[i]
#             output = model(feat)
#             output = torch.squeeze(output)
#             loss = criterion(output, targ)
#             train_loss += loss.item()
#             loss.backward()
#             for p in model.named_parameters():
#                 torch.nn.utils.clip_grad_norm_(p[1].grad, clipping, norm_type=2)
#                 temp_par[p[0]] += deepcopy(p[1].grad)
#             output = output.cpu().detach().numpy()
#             train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
#             train_outputs.append(output)
#             # model.zero_grad()
#             num_data_point += 1
#
#         for p in model.named_parameters():
#             p[1].grad = deepcopy(temp_par[p[0]]) + torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(
#                 device)
#             p[1].grad = p[1].grad / bz
#         for p in model.parameters():
#             p = p - lr*p.grad
#
#     if scheduler is not None:
#         scheduler.step()
#
#     return train_loss / num_data_point, train_outputs, train_targets
#
#
