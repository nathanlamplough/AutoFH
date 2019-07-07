import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import MetaFunctions as mf

#Reading in the iris data to test the meta functions with.
#Iris was chosen as it is one the most well known datasets.
iris = pd.read_csv('../Data/iris.csv', delimiter= ',', quoting= 3)
y = np.asarray(iris.iloc[:,-1])
X = np.asarray(iris.drop(iris.columns[-1], axis=1))
#This are the already known meta values of iris that were calculated using excel
no_features = 4
no_instances = 150
class_entropy = 1.584962500721156
mean_mutual_info = 0.7399384184312428
feature_entropy = 4.483305040525429

def test_compute_n_features():
    global X
    assert  mf.compute_n_features(X) == 4

def test_compute_n_classes():
    global y
    assert mf.compute_n_classes(y) == 3


def test_compute_dimensionality():
    global y
    assert mf.compute_dimensionality(150, 4) == 37.5

#Testing the correlation between each feature and the target
#then taking the mean
def test_compute_mean_correlation():
    global iris
    assert mf.compute_mean_correlation(iris) == 1.3069313703926795

#testing the skewnesss of the data
def test_compute_mean_skewness():
    global iris
    assert mf.compute_mean_skewness(iris) == 0.05390056083822681

#testing the kurtosis of the data
def test_compute_mean_kurtosis():
    global X
    assert mf.compute_mean_kurtosis(X) == -0.7656823989531729

#testing the entropy of each feature than return the average
def test_compute_mean_feature_entropy():
    global X
    assert mf.compute_mean_feature_entropy(X) == 4.483305040525429

#testsing the entropy of the targets
def test_compute_entropy():
    global y
    assert mf.compute_entropy(y) == 1.584962500721156

#testing the mean and max mutual info
def test_compute_mutual_info():
    global X
    global y
    mi = mf.compute_mutual_info(X,y)
    assert mi[0] == 0.7399384184312428
    assert mi[1] == 1.002510220562348

#testing the equivalent number of features
def test_compute_equiv_n_features():
    global class_entropy
    global mean_mutual_info
    assert mf.compute_equiv_n_features(class_entropy,
                                       mean_mutual_info) == 2.142019472487271

#testing the noise signal ratio
def test_compute_noise_signal_ratio():
    global feature_entropy
    global mean_mutual_info
    assert mf.compute_noise_signal_ratio(feature_entropy,
                                         mean_mutual_info) == 5.059024546975905


