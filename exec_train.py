import numpy
import torch
import itertools
import pandas
import util.autoencoder as ae
import util.dopnet as dp
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from util.ml import get_k_folds_list

# experiment settings
dataset_path = 'dataset/mrl.xlsx'
target_idx = 5
max_dops = 3
init_lr = 1e-1
n_folds = 3

# dataset loading
dataset = dp.load_dataset(dataset_path, comp_idx=0, target_idx=6, max_dops=max_dops, cond_idx=[1])
rand_idx = numpy.random.permutation(len(dataset))
rand_dataset = [dataset[idx] for idx in rand_idx]
k_folds = get_k_folds_list(rand_dataset, k=n_folds)

# list objects storing prediction results
list_test_mae = list()
list_test_rmse = list()
list_test_r2 = list()
list_preds = list()
list_embs = list()

# train and evaluate DopNet for k-fold dataset
for k in range(0, n_folds):
    print('---------------------- Fold [{}/{}] ----------------------'.format(k + 1, n_folds))

    # load training dataset
    dataset_train = list(itertools.chain(*(k_folds[:k] + k_folds[k + 1:])))
    comps_train = [x.comp for x in dataset_train]
    targets_train = numpy.array([x.target for x in dataset_train]).reshape(-1, 1)
    dop_dataset_train = dp.get_dataset(dataset_train, max_dops)
    data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
    data_loader_calc = DataLoader(dop_dataset_train, batch_size=32)

    # load test dataset
    dataset_test = k_folds[k]
    comps_test = [x.comp for x in dataset_test]
    targets_test = numpy.array([x.target for x in dataset_test]).reshape(-1, 1)
    dop_dataset_test = dp.get_dataset(dataset_test, max_dops)
    data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

    # define host embedding network and its optimizer
    emb_host = ae.Autoencoder(dataset[0].host_feat.shape[0], 64).cuda()
    optimizer_emb = torch.optim.Adam(emb_host.parameters(), lr=1e-3, weight_decay=1e-5)

    # train the host embedding network
    for epoch in range(0, 300):
        train_loss = ae.train(emb_host, data_loader_train, optimizer_emb)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 300, train_loss))

    # calculate host embeddings
    host_embs_train = ae.test(emb_host, data_loader_calc)
    host_embs_test = ae.test(emb_host, data_loader_test)

    # load dataset for DopNet
    dop_dataset_train.host_feats = host_embs_train
    dop_dataset_test.host_feats = host_embs_test
    data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
    data_loader_calc = DataLoader(dop_dataset_train, batch_size=32)
    data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

    # define DopNet and its optimizer
    pred_model = dp.DopNet(host_embs_train.shape[1], dataset[0].dop_feats.shape[1], dim_out=1, max_dops=max_dops).cuda()
    optimizer = torch.optim.SGD(pred_model.parameters(), lr=init_lr, weight_decay=1e-7)
    criterion = torch.nn.L1Loss()

    # train DopNet
    for epoch in range(0, 600):
        if (epoch + 1) % 200 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5

        train_loss = dp.train(pred_model, data_loader_train, optimizer, criterion)
        preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
        test_loss = mean_absolute_error(targets_test, preds_test)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest loss: {:.4f}'.format(epoch + 1, 600, train_loss, test_loss))

    # calculate predictions, embeddings, and evaluation metrics
    preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
    embs_test = dp.emb(pred_model, data_loader_test).cpu().numpy()
    list_test_mae.append(mean_absolute_error(targets_test, preds_test))
    list_test_rmse.append(numpy.sqrt(mean_squared_error(targets_test, preds_test)))
    list_test_r2.append(r2_score(targets_test, preds_test))

    # save prediction and embedding results to the list objects
    idx_test = numpy.array([x.idx for x in dataset_test]).reshape(-1, 1)
    list_preds.append(numpy.hstack([idx_test, targets_test, preds_test]))
    list_embs.append(numpy.hstack([idx_test, targets_test, embs_test]))

# save prediction end embedding results as files
pandas.DataFrame(numpy.vstack(list_preds)).to_excel('save/pred/preds_dopnet.xlsx', header=None, index=None)
pandas.DataFrame(numpy.vstack(list_embs)).to_excel('save/emb/embs_dopnet.xlsx', header=None, index=None)

# print evaluation results
print('Test MAE: ' + str(numpy.mean(list_test_mae)))
print('Test RMSE: ' + str(numpy.mean(list_test_rmse)))
print('Test R2: ' + str(numpy.mean(list_test_r2)))
