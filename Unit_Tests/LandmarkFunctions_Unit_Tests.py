import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import LandmarkMetaFunctions as lmf

#Reading in the iris data to test the landmark meta functions with.
#Iris was chosen as it is one the most well known datasets.
iris = pd.read_csv('../Data/iris.csv', delimiter= ',', quoting= 3)
y = np.asarray(iris.iloc[:,-1])
X = np.asarray(iris.drop(iris.columns[-1], axis=1))

#Testing the error produced from a one node
#decision tree
def test_decision_node():
    global X
    global y
    assert lmf.Decision_Node(X,y) == 0.38

#Testing the error produced from a one random node
#decision tree
def test_random_node():
    global X
    global y
    assert lmf.Random_Node(X,y) == 0.38

#Testing the error produced from a naive bayes classifier
def test_naive_bayes():
    global X
    global y
    assert lmf.Naive_Bayes(X ,y) == 0.040000000000000036

#Testing the error produced from a 1-K K nearest neighbour
def test_nearest_neighbour():
    global X
    global y
    assert lmf.Nearest_Neighbour(X , y) == 0.040000000000000036

#Testing the error produced from LDA
def test_lda():
    global X
    global y
    assert lmf.LDA(X,y) == 0.020000000000000018