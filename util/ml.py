import numpy
import torch
import joblib
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor


def sep_list_dataset(dataset, ratio):
    n_train = int(ratio * len(dataset))
    rand_idx = numpy.random.permutation(len(dataset))
    dataset_train = [dataset[idx] for idx in rand_idx[:n_train]]
    dataset_test = [dataset[idx] for idx in rand_idx[n_train:]]

    return dataset_train, dataset_test, rand_idx[:n_train], rand_idx[n_train:]


def sep_arr_dataset(dataset, ratio):
    n_train = int(ratio * dataset.shape[0])
    rand_idx = numpy.random.permutation(len(dataset))

    return dataset[rand_idx[:n_train], :], dataset[rand_idx[n_train:], :], rand_idx[:n_train], rand_idx[n_train:]


def get_k_folds_numpy(dataset, k):
    size_fold = int(dataset.shape[0] / k)
    pos = 0
    folds = list()

    for i in range(0, k - 1):
        folds.append(dataset[pos:pos + size_fold, :])
        pos += size_fold
    folds.append(dataset[pos:, :])

    return folds


def get_k_folds_list(dataset, k):
    size_fold = int(len(dataset) / k)
    pos = 0
    folds = list()

    for i in range(0, k - 1):
        folds.append(dataset[pos:pos + size_fold])
        pos += size_fold
    folds.append(dataset[pos:])

    return folds


def get_emb_gnn(model, data_loader):
    model.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            embs = model.emb(batch)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def run_svr(dataset_train, dataset_test):
    n_train = int(0.8 * dataset_train.shape[0])

    train_x = dataset_train[:n_train, :-1]
    train_y = dataset_train[:n_train, -1].reshape(-1, 1)
    val_x = dataset_train[n_train:, :-1]
    val_y = dataset_train[n_train:, -1].reshape(-1, 1)
    test_x = dataset_test[:, :-1]
    test_y = dataset_test[:, -1].reshape(-1, 1)
    min_val_error = 1e+8

    for c in [0.01, 0.1, 0.5, 1.0]:
        for e in [0.1, 0.2, 0.4]:
            model = SVR(C=c, epsilon=e).fit(train_x, train_y)
            preds = model.predict(val_x).reshape(-1, 1)
            val_error = numpy.mean(numpy.abs(val_y - preds))

            if val_error < min_val_error:
                min_val_error = val_error
                opt_c = c
                opt_e = e

    model = SVR(C=opt_c, epsilon=opt_e).fit(train_x, train_y)
    preds = model.predict(test_x).reshape(-1, 1)
    test_mae = numpy.mean(numpy.abs(test_y - preds))
    test_rmse = numpy.sqrt(numpy.mean((test_y - preds) ** 2))
    test_r2 = r2_score(test_y, preds)

    return preds, test_mae, test_rmse, test_r2


def run_gpr(train_x, train_y, test_x, test_y):
    kernel = DotProduct() + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel).fit(train_x, train_y)
    preds = model.predict(test_x).reshape(-1, 1)
    test_mae = numpy.mean(numpy.abs(test_y - preds))
    test_rmse = numpy.sqrt(numpy.mean((test_y - preds)**2))
    test_r2 = r2_score(test_y, preds)

    return preds, test_mae, test_rmse, test_r2


def run_gbtr(dataset_train, dataset_test, path_model=None):
    n_train = int(0.8 * dataset_train.shape[0])

    train_x = dataset_train[:n_train, :-1]
    train_y = dataset_train[:n_train, -1]
    val_x = dataset_train[n_train:, :-1]
    val_y = dataset_train[n_train:, -1]
    test_x = dataset_test[:, :-1]
    test_y = dataset_test[:, -1]
    min_val_error = 1e+8

    for d in range(3, 9):
        for n in [100, 200, 300, 400]:
            model = GradientBoostingRegressor(max_depth=d, n_estimators=n)
            model.fit(train_x, train_y)
            preds = model.predict(val_x)
            val_error = numpy.mean(numpy.abs(val_y - preds))
            print('d={}\tn={}\tMAE: {:.4f}'.format(d, n, val_error))

            if val_error < min_val_error:
                min_val_error = val_error
                opt_d = d
                opt_n = n

    model = GradientBoostingRegressor(max_depth=opt_d, n_estimators=opt_n)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    test_mae = numpy.mean(numpy.abs(test_y - preds))
    test_rmse = numpy.sqrt(numpy.mean((test_y - preds) ** 2))
    test_r2 = r2_score(test_y, preds)

    if path_model is not None:
        joblib.dump(model, path_model)

    return preds, test_mae, test_rmse, test_r2, model


def train_ae(model, data_loader, optimizer, criterion):
    sum_train_losses = 0

    for data, _ in data_loader:
        data = data.cuda()

        x_p = model(data)
        loss = criterion(data, x_p)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_losses += loss.item()

    return sum_train_losses / len(data_loader)
