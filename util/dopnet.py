import numpy
import pandas
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.chem as chem
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import scale


class DopedMat:
    def __init__(self, comp, host_feat, dop_feats, conds, target, idx):
        self.comp = comp
        self.host_feat = host_feat
        self.dop_feats = dop_feats
        self.conds = conds
        self.target = target
        self.idx = idx


class DopDataset(Dataset):
    def __init__(self, host_feats, dop_feats, targets, max_dops):
        self.host_feats = host_feats
        self.dop_feats = dop_feats
        self.targets = targets
        self.max_dops = max_dops

    def __len__(self):
        return self.host_feats.shape[0]

    def __getitem__(self, idx):
        idx_dop = self.max_dops * idx

        return self.host_feats[idx, :], self.dop_feats[idx_dop:idx_dop + self.max_dops, :], self.targets[idx, :]


class DopNet(nn.Module):
    def __init__(self, dim_host_feats, dim_dop_feats, dim_out, max_dops):
        super(DopNet, self).__init__()
        self.max_dops = max_dops
        self.emb_host_feats = nn.Linear(dim_host_feats, 256)
        self.emb_dop_feats = nn.Linear(dim_dop_feats, 256)
        self.fc1 = nn.Linear((self.max_dops + 1) * 256, 512)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 16)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(16, dim_out)

    def forward(self, host_feats, dop_feats):
        emb_host_feats = F.relu(self.emb_host_feats(host_feats))
        emb_dop_feats = self.slice_dop_feats(F.relu(self.emb_dop_feats(dop_feats)), emb_host_feats.shape[0])
        h = self.dp1(F.relu(self.fc1(torch.cat([emb_host_feats, emb_dop_feats], dim=1))))
        h = self.dp2(F.relu(self.fc2(h)))
        out = self.fc3(h)

        return out

    def emb(self, host_feats, dop_feats):
        emb_host_feats = F.relu(self.emb_host_feats(host_feats))
        emb_dop_feats = self.slice_dop_feats(F.relu(self.emb_dop_feats(dop_feats)), emb_host_feats.shape[0])
        h = self.dp1(F.relu(self.fc1(torch.cat([emb_host_feats, emb_dop_feats], dim=1))))
        embs = self.dp2(F.relu(self.fc2(h)))

        return embs

    def slice_dop_feats(self, dop_feats, n_mats):
        list_dop_feats = list()

        for i in range(0, n_mats):
            list_dop_feats.append(dop_feats[i, :, :].view(1, -1))

        return torch.cat(list_dop_feats, dim=0)


def train(model, data_loader, optimizer, criterion):
    model.train()
    sum_train_losses = 0

    for host_feats, dop_feats, targets in data_loader:
        host_feats = host_feats.cuda()
        dop_feats = dop_feats.cuda()
        targets = targets.cuda()

        preds = model(host_feats, dop_feats)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_losses += loss.item()

    return sum_train_losses / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for host_feats, dop_feats, _ in data_loader:
            host_feats = host_feats.cuda()
            dop_feats = dop_feats.cuda()

            preds = model(host_feats, dop_feats)

            list_preds.append(preds)

    return torch.cat(list_preds, dim=0)


def emb(model, data_loader):
    model.eval()
    list_embs = list()

    with torch.no_grad():
        for host_feats, dop_feats, _ in data_loader:
            host_feats = host_feats.cuda()
            dop_feats = dop_feats.cuda()

            embs = model.emb(host_feats, dop_feats)

            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def load_dataset(dataset_file_name, comp_idx, target_idx, max_dops, cond_idx=None, norm_target=False):
    elem_feats = chem.load_elem_feats()
    data = numpy.array(pandas.read_excel(dataset_file_name))
    targets = data[:, target_idx].astype(float).reshape(-1, 1)
    comps = data[:, comp_idx]
    dataset = list()

    if cond_idx is not None:
        norm_conds = scale(data[:, cond_idx])

    if norm_target:
        target_mean = numpy.mean(targets)
        target_std = numpy.std(targets)
        targets = scale(targets)

    for i in tqdm(range(0, comps.shape[0])):
        host_feat, dop_feats = calc_atom_feats(elem_feats, comps[i], max_dops)
        conds = None

        if cond_idx is not None:
            host_feat = numpy.hstack([host_feat, norm_conds[i]])
            conds = data[i, cond_idx]

        dataset.append(DopedMat(comps[i], host_feat, dop_feats, conds, targets[i], idx=i))

    if norm_target:
        return dataset, target_mean, target_std
    else:
        return dataset


def calc_atom_feats(elem_feats, comp, max_dops):
    host_feats = list()
    dop_feats = numpy.zeros((max_dops, elem_feats.shape[1] + 1))
    elems = ast.literal_eval(str(chem.parse_formula(comp)))
    e_sum = numpy.sum([float(elems[key]) for key in elems])
    w_sum_vec = numpy.zeros(elem_feats.shape[1])
    n_dops = 0

    for e in elems:
        atom_vec = elem_feats[chem.atom_nums[e] - 1, :]
        ratio = float(elems[e])

        if ratio <= 0.1:
            dop_feats[n_dops, :] = numpy.hstack([numpy.log10(ratio), atom_vec])
            n_dops += 1
        else:
            w_sum_vec += (ratio / e_sum) * atom_vec
            host_feats.append(atom_vec)

    host_feat = numpy.hstack([w_sum_vec, numpy.std(host_feats, axis=0), numpy.min(host_feats, axis=0), numpy.max(host_feats, axis=0)])

    return host_feat, dop_feats


def get_dataset(list_data, max_dops):
    host_feats = list()
    dop_feats = list()
    targets = list()

    for x in list_data:
        host_feats.append(x.host_feat)
        dop_feats.append(x.dop_feats)
        targets.append(x.target)

    host_feats = torch.tensor(numpy.vstack(host_feats), dtype=torch.float)
    dop_feats = torch.tensor(numpy.vstack(dop_feats), dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float).view(-1, 1)

    return DopDataset(host_feats, dop_feats, targets, max_dops)
