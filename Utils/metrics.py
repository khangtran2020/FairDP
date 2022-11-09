import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def demo_parity(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    female_outputs = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)

        for bi, d in enumerate(female_loader):
            features, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)

    male_outputs = np.round(np.array(male_outputs))
    female_outputs = np.round(np.array(female_outputs))
    prob_male = np.sum(male_outputs) / len(male_outputs)
    prob_female = np.sum(female_outputs) / len(female_outputs)
    return prob_male, prob_female, np.abs(prob_male - prob_female)

def equality_of_odd(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    male_target = []
    female_outputs = []
    female_target = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, target = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)
            male_target.extend(target.cpu().detach().numpy().astype(int).tolist())

        for bi, d in enumerate(female_loader):
            features, target = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)
            female_target.extend(target.cpu().detach().numpy().astype(int).tolist())

    male_outputs = np.round(np.array(male_outputs))
    tn, fp, fn, tp = confusion_matrix(male_target, male_outputs).ravel()
    male_tpr = tp / (tp + fn)
    female_outputs = np.round(np.array(female_outputs))
    tn, fp, fn, tp = confusion_matrix(female_target, female_outputs).ravel()
    female_tpr = tp / (tp + fn)
    return male_tpr, female_tpr, np.abs(male_tpr - female_tpr)

def disperate_impact(male_loader, female_loader, global_model, male_model, female_model, num_male, num_female, device):
    global_model.to(device)
    male_model.to(device)
    female_model.to(device)

    glob_male_out = []
    glob_female_out = []
    male_outputs = []
    female_outputs = []

    global_model.eval()
    male_model.eval()
    female_model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = global_model(features)
            male_out = male_model(features)

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_male_out.extend(glob_out)

            male_out = torch.squeeze(male_out, dim=-1)
            male_out = male_out.cpu().detach().numpy()
            male_outputs.extend(male_out)

        for bi, d in enumerate(female_loader):
            features, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = global_model(features)
            female_out = female_model(features)

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_female_out.extend(glob_out)

            female_out = torch.squeeze(female_out, dim=-1)
            female_out = female_out.cpu().detach().numpy()
            female_outputs.extend(female_out)

    male_outputs = np.array(male_outputs)
    glob_male_out = np.array(glob_male_out)
    female_outputs = np.array(female_outputs)
    glob_female_out = np.array(glob_female_out)

    male_norm = np.sum(np.abs(male_outputs - glob_male_out))
    female_norm = np.sum(np.abs(female_outputs - glob_female_out))
    return male_norm / num_male, female_norm / num_female