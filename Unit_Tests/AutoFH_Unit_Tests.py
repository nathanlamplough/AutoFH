import numpy as np
import sys
sys.path.append("..")
import AutoFH as af

#Testing that a class 1D array is turned into the matrix one hot form
def test_to_one_hot():
    test = [0, 1, 0, 1]
    result = np.asarray([[1 , 0],
              [0, 1],
              [1, 0],
              [0, 1]])
    test_result = af.to_one_hot(test, 2)

    for i in range(len(result)):
        for j in range(len(result[i])):
            assert test_result[i, j] == result[i,j]

#Testing that a string of floats seperated by spaces is turned
#into an array of floats
def test_to_float():
    db_strings = np.empty(0)
    db_strings = np.append(db_strings, test_database_class('0.1 10 1234 5.432 72.1'))
    db_strings = np.append(db_strings, test_database_class('0.1 10 1234 5.432 72.1'))

    result = np.asarray([[0.1, 10.0, 1234.0, 5.432, 72.1],
              [0.1, 10.0, 1234.0, 5.432, 72.1]], dtype=float)

    test_result = af.to_float(db_strings)

    for i in range(len(result)):
        for j in range(len(result[i])):
            assert test_result[i, j] == result[i, j]

#A dummy db class which is used in the test_get_closest test
class test_database_class():
    def __init__(self, meta_str):
        self.l_meta_features = meta_str

#Testing that the inputted meta feature array finds the closest one
#out of the two in meta_features in euclidean space
def test_get_closest():
    meta_features = np.asarray([[0.1, 10.0, 1234.0, 5.432, 72.1],
                    [0.4, 11.0, 1454.0, 7.432, 83.1]])

    meta_feature_input = np.asarray([0.2, 10.5, 1235.0, 6.432, 75.1])

    result = np.asarray([0.1, 10.0, 1234.0, 5.432, 72.1])

    assert af.Get_Closest(meta_feature_input, meta_features) == 0
