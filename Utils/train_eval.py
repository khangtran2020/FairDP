import numpy as np
import torch
from Utils.utils import get_gaussian_noise
from opacus.utils.batch_memory_manager import BatchMemoryManager
from copy import deepcopy



class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

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
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0

    for bi, d in enumerate(dataloader):
        features, target = d

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
            features, target = d

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
    # print(noise_std)
    train_targets = []
    train_outputs = []
    train_loss = 0
    num_data_point = 0
    for bi, d in enumerate(dataloader):
        features, target = d
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
            for p in model.named_parameters():
                torch.nn.utils.clip_grad_norm_(p[1].grad, clipping, norm_type=2)
                temp_par[p[0]] += deepcopy(p[1].grad)
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

def train_opacus(dataloader, model, criterion, optimizer, device, args):
    model.to(device)
    model.train()
    train_targets = []
    train_outputs = []
    train_loss = 0

    with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=args.MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:
        for bi, d in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            features, target = d
            features = features.to(device)
            target = target.to(device)

            # compute output
            output = model(features)
            output = torch.squeeze(output)
            loss = criterion(output, target)
            loss.backward()

            output = output.cpu().detach().numpy()
            train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            train_outputs.extend(output)

            train_loss += loss.item()
            optimizer.step()
    return train_loss, train_outputs, train_targets