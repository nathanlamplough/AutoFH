from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def Decision_Node(X, Y):
    clf = DecisionTreeClassifier(random_state=0, max_depth=1, criterion='entropy')
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return 1 - accuracy_score(pred, y_te)

def Random_Node(X, Y):
    clf = DecisionTreeClassifier(random_state= 0, max_depth=1, criterion='entropy', splitter='random')
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return 1 - accuracy_score(pred, y_te)

def Naive_Bayes(X,Y):
    clf = GaussianNB()
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return 1 - accuracy_score(pred, y_te)

def Nearest_Neighbour(X, Y):
    clf = KNeighborsClassifier(n_neighbors=1)
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return 1 - accuracy_score(pred, y_te)

def LDA(X,Y):
    clf = LinearDiscriminantAnalysis()
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return 1 - accuracy_score(pred, y_te)

def to_string(n):
    str_n = str(n)
    str_n = str_n.replace(',', '')
    str_n = str_n.replace('[', '')
    str_n = str_n.replace(']', '')
    return str_n

def compute_l_meta_features(data, to_string_bool):
    lmf = []
    lmf.append(Decision_Node(data[0], data[1]))
    lmf.append((Random_Node(data[0], data[1])))
    lmf.append((Naive_Bayes(data[0], data[1])))
    lmf.append((Nearest_Neighbour(data[0], data[1])))
    lmf.append((LDA(data[0], data[1])))
    if(to_string_bool == False):
        return np.asarray(lmf)
    str_lmf = str(lmf)
    str_lmf = str_lmf.replace(',', '')
    str_lmf = str_lmf.replace('[', '')
    str_lmf = str_lmf.replace(']', '')
    return str_lmf

