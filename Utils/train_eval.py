import numpy as np
import torch
from Utils.utils import get_gaussian_noise

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

    ep_loss = 0

    for bi, d in enumerate(dataloader):
        features, target = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()

        output = model(features)
        output = torch.squeeze(output)

        loss = criterion(output, target)
        ep_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return ep_loss, train_outputs, train_targets

def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    ep_loss = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            ep_loss += loss_eval.item()

            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return ep_loss, fin_outputs, fin_targets

def train_fn_dpsgd(dataloader, model, criterion, optimizer, device, scheduler, clipping, noise_scale):
    model.to(device)
    model.train()
    noise_std = get_gaussian_noise(clipping, noise_scale)
    # print(noise_std)
    train_targets = []
    train_outputs = []

    batch = next(iter(dataloader))
    features, target = batch
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    optimizer.zero_grad()
    model.zero_grad()
    temp_par = {}
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
    bz = features.size(dim=0)
    train_loss = 0
    for i in range(bz):
        feat = features[i]
        targ = target[i]
        output = model(feat)
        output = torch.squeeze(output)
        loss = criterion(output, targ)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        for p in model.named_parameters():
            temp_par[p[0]] += p[1].grad / bz
        output = output.cpu().detach().numpy()
        train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
        train_outputs.append(output)
        model.zero_grad()

    for p in model.named_parameters():
        p[1].grad = temp_par[p[0]] + torch.normal(0, noise_std, temp_par[p[0]].size()).to(device) / bz
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return train_loss / bz, train_outputs, train_targets