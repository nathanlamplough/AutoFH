import numpy as np
from scipy import stats as stats
from sklearn.metrics import mutual_info_score as mi


def compute_n_features(X):
    return X.shape[1]


def compute_n_classes(y):
    return len(np.unique(y))


def compute_dimensionality(n_instances, n_features):
    return n_instances / n_features


def compute_mean_correlation(data):
    return data.corr().sum().sum() / (data.shape[1] * 2)


def compute_mean_skewness(data):
    return data.skew().sum() / (data.shape[1])


def compute_mean_kurtosis(X):
    return stats.kurtosis(X).sum() / (X.shape[1])


def compute_mean_feature_entropy(X):
    sum_entropy = 0
    for i in range(X.shape[1]):
        sum_entropy = sum_entropy + compute_entropy(X[:, i])
    return sum_entropy / X.shape[1]


def compute_entropy(y):
    u, c = np.unique(y, return_counts=True)
    entro = ((c / c.sum()) * np.log2(c / c.sum())).sum()
    return -entro;


def compute_mutual_info(X, y):
    m_infos = []
    for i in range(X.shape[1]):
        m_infos.append(mi(X[:, i], y))
    mean_mi = sum(m_infos) / X.shape[1]
    max_mi = max(m_infos)
    return (mean_mi, max_mi)


def compute_equiv_n_features(class_entropy, mean_mutual_info):
    return class_entropy / mean_mutual_info


def compute_noise_signal_ratio(feature_entropy, mean_mutual_info):
    return (feature_entropy - mean_mutual_info) / mean_mutual_info

def to_string(n):
    str_n = str(n)
    str_n = str_n.replace(',', '')
    str_n = str_n.replace('[', '')
    str_n = str_n.replace(']', '')
    return str_n

def compute_meta_features(data, to_string_bool):
    mf = []

    mf.append(compute_n_features(data))
    mf.append(compute_n_classes(data))
    mf.append(compute_dimensionality(mf[0], mf[1]))
    mf.append(compute_mean_correlation(data))
    mf.append(compute_mean_skewness(data))
    mf.append(compute_mean_kurtosis(data))
    mf.append(compute_mean_feature_entropy(data[0]))
    mf.append(compute_entropy(data[1]))
    mean_info, max_info = compute_mutual_info(data[0], data[1])
    mf.append(mean_info)
    mf.append(max_info)
    mf.append(compute_equiv_n_features(mf[7], mean_info))
    mf.append(compute_noise_signal_ratio(mf[6], mean_info))
    if(to_string_bool == False):
        return np.asarray(mf)
    str_mf = to_string(mf)
    return str_mf
