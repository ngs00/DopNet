import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, dim_in, dim_latent):
        super(Autoencoder, self).__init__()
        self.enc_fc1 = nn.Linear(dim_in, 256)
        self.dp1 = nn.Dropout(p=0.3)
        self.enc_fc2 = nn.Linear(256, dim_latent)
        self.dpz = nn.Dropout(p=0.1)
        self.dec_fc1 = nn.Linear(dim_latent, 256)
        self.dp2 = nn.Dropout(p=0.3)
        self.dec_fc2 = nn.Linear(256, dim_in)

    def forward(self, x):
        z = self.enc(x)
        x_p = self.dec(z)

        return x_p

    def enc(self, x):
        h = F.leaky_relu(self.enc_fc1(x))
        z = F.leaky_relu(self.enc_fc2(h))

        return z

    def dec(self, z):
        h = F.leaky_relu(self.dec_fc1(z))
        x_p = self.dec_fc2(h)

        return x_p


def train(model, data_loader, optimizer):
    model.train()
    criterion = nn.MSELoss()
    sum_train_losses = 0

    for host_feats, _, _ in data_loader:
        host_feats = host_feats.cuda()

        x_p = model(host_feats)
        loss = criterion(host_feats, x_p)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_losses += loss.item()

    return sum_train_losses / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_embs = list()

    with torch.no_grad():
        for host_feats, _, _ in data_loader:
            list_embs.append(model.enc(host_feats.cuda()))

    return torch.cat(list_embs, dim=0)
