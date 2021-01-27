import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.utils import resample
from numpy.linalg import det, inv


# ==========feature preprocessing===========
def feature_nominal2scalar(nominal_features):
    """
    convert the nominal feature description into inputable scalar
    """
    scalar_features = nominal_features.copy(deep=True)
    unique_nominals = scalar_features.unique()
    for index, nominal in enumerate(unique_nominals):
        scalar_features[nominal_features == nominal] = index
    return scalar_features.astype('int64')


def one_hot_tensor(feature_column):
    feature_tensor = torch.LongTensor(feature_column)
    feature_tensor = feature_tensor.view(feature_tensor.size()[0], 1)
    # feature_one_hot = torch.zeros(len(feature_column), feature_column.unique().shape[0])
    feature_one_hot = torch.LongTensor(len(feature_column), feature_column.unique().shape[0])
    feature_one_hot.zero_()
    feature_one_hot.scatter_(1, feature_tensor, 1)
    return feature_one_hot.type(torch.float32)


def numeric_tensor(feature_column):
    feature_tensor = torch.FloatTensor(feature_column)
    feature_tensor = feature_tensor.view(feature_tensor.size()[0], 1)
    return feature_tensor


# ==========dataset splite==================
def dataset_splite(Dataset, num4train, num_val, num_test):
    """
    num4train = train_num + num_val
    val_num = num_val
    test_num = num_test
    len(dataset) = num4train + num_test
    """
    train_db, test_db = Data.random_split(Dataset, [num4train, num_test])
    train_db, val_db = Data.random_split(train_db, [num4train - num_val, num_val])
    return train_db, val_db, test_db


# ================plot=======================
def plot_loss(loss_record, save_path, epoch_num):
    loss_record_copy = loss_record.copy(deep=True)
    loss_record_copy['epoch'] = ''
    for i in range(epoch_num):
        loss_record_copy.loc[i, 'epoch'] = i + 1
    # =======plot==========
    plt.plot(loss_record_copy['epoch'], loss_record_copy['Multi_loss_train'], c='black', linewidth=0.6,
             label='multi_loss_train')
    plt.plot(loss_record_copy['epoch'], loss_record_copy['mse_loss'], c='yellow', linewidth=0.6, label='mse loss')
    plt.plot(loss_record_copy['epoch'], loss_record_copy['crossEntropy_loss'], c='violet', linewidth=0.6,
             label='cross entropy loss')
    plt.plot(loss_record_copy['epoch'], loss_record_copy['KLD'], c='darkgrey', linewidth=0.6,
             label='KLD')
    plt.plot(loss_record_copy['epoch'], loss_record_copy['Multi_loss_val'], c='orangered', linewidth=0.6,
             label='multi_loss_val')
    plt.title('multi loss and loss items performance')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path, dpi=200)


# ==============reweighing calculation=============
def reweighing_calculate(df, feature_name, class_name):
    att_statistic = df[feature_name].value_counts()
    class_statistic = df[class_name].value_counts()
    combine_statistic = df.groupby(by=[feature_name, class_name]).size()
    weight_df = pd.DataFrame(index=[att_statistic.index], columns=[class_statistic.index])
    ds_size = len(df)
    for i in range(2):
        for j in range(2):
            exp = (att_statistic[att_statistic.index[i]] / ds_size) * (class_statistic[class_statistic.index[j]] / ds_size)
            obs = combine_statistic[att_statistic.index[i]][class_statistic.index[j]] / ds_size
            weight_df.loc[att_statistic.index[i], class_statistic.index[j]] = round(exp / obs, 2)
    print(weight_df)
    return weight_df


def weight_tensor_generator(dataset, weight_df):
    weight_tensor = torch.zeros(len(dataset))
    male_less = torch.tensor([0, 0, 1]).type(torch.float32)
    male_more = torch.tensor([1, 0, 1]).type(torch.float32)
    female_less = torch.tensor([0, 1, 0]).type(torch.float32)
    female_more = torch.tensor([1, 1, 0]).type(torch.float32)

    for num, (_, label) in enumerate(dataset):
        if torch.equal(label, male_less) is True:
            weight_tensor[num] = weight_df.iloc[0, 0]
        elif torch.equal(label, male_more) is True:
            weight_tensor[num] = weight_df.iloc[0, 1]
        elif torch.equal(label, female_less) is True:
            weight_tensor[num] = weight_df.iloc[1, 0]
        elif torch.equal(label, female_more) is True:
            weight_tensor[num] = weight_df.iloc[1, 1]
    return weight_tensor.reshape((len(dataset), 1))


# ============ upsampling =================
def Upsampling(dataset, weight_df):
    """
    dataset(tensor)
    weight_df(dataframe)
    """
    array = dataset.numpy()
    X = array[:, 0:-1]
    Y = array[:, -1]

    # ===== gender and income index ======
    id_female = np.where(X[:, 60:62] == np.array([1, 0]))[0]
    id_male = np.where(X[:, 60:62] == np.array([0, 1]))[0]
    id_less = np.where(Y[:] == 0)[0]
    id_more = np.where(Y[:] == 1)[0]

    id_fl = np.intersect1d(id_female, id_less)
    id_fm = np.intersect1d(id_female, id_more)
    id_ml = np.intersect1d(id_male, id_less)
    id_mm = np.intersect1d(id_male, id_more)
    print("fl:fm:ml:mm:", len(id_fl), len(id_fm), len(id_ml), len(id_mm))
    dataset_fl = np.hstack((X[id_fl], Y[id_fl].reshape(len(id_fl), 1)))
    dataset_fm = np.hstack((X[id_fm], Y[id_fm].reshape(len(id_fm), 1)))
    dataset_ml = np.hstack((X[id_ml], Y[id_ml].reshape(len(id_ml), 1)))
    dataset_mm = np.hstack((X[id_mm], Y[id_mm].reshape(len(id_mm), 1)))
    # ======= oversampling =======
    dataset_fl_au = resample(dataset_fl, replace=True, n_samples=11797, random_state=123)
    dataset_fm_au = resample(dataset_fm, replace=True, n_samples=30532, random_state=123)
    dataset_mm_au = resample(dataset_mm, replace=True, n_samples=10964, random_state=123)

    dataset_au = np.vstack((dataset_fl_au, dataset_fm_au, dataset_ml, dataset_mm_au))
    np.random.shuffle(dataset_au)

    label_array = dataset_au[:, -1]
    data_array = dataset_au[:, 0:-1]
    sex_array = dataset_au[:, 60:62]

    return torch.from_numpy(data_array), torch.from_numpy(sex_array), torch.from_numpy(label_array)


# =======JS-distance =======
def kl_divergence(p, q):
    """
    :param p (array): probability distribution 1
    :param q (array): probability distribution 2
    :return: kl divergence of two distribution
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def KL_divergence(a, b):
    """
    :param a (tensor): probability distribution 1 of one batch
    :param b (tensor): probability distribution 2 of one batch
    :return: kl divergence of two probability distribution
    """
    cov_a = np.cov(a)
    cov_b = np.cov(b)
    mean_a = a.mean(axis=1)
    mean_b = b.mean(axis=1)

    KL_ab = 0.5 * (np.log(det(cov_b) / det(cov_a)) - a.shape[0] + np.trace(np.dot(inv(cov_b), cov_a)) + np.dot(
        np.dot((mean_b - mean_a).T, inv(cov_b)), mean_b - mean_a))
    return KL_ab

def js_distance(p, q):
    """
    :param p (array): probability distribution 1
    :param q (array): probability distribution 2
    :return: js distance of two distribution
    """
    return 0.5 * kl_divergence(p, 0.5 * (p + q)) + 0.5 * kl_divergence(q, 0.5 * (p + q))
