import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def run_network(epochs, X, Y, model, l_rate, cv, l_reg, lamb, momentum, d_out):
    indices = np.random.permutation(X.shape[0])
    bins = np.array_split(indices, cv)

    if d_out < 3 :
        loss_func = torch.nn.BCELoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=l_rate, momentum= momentum)

    y_predictions = np.empty(0)
    y_fold_order = np.empty(0)
    for fold in range(cv):
        foldTrain = np.delete(bins, fold, axis=0)  # list to save current indices for training
        foldTrain = np.hstack(foldTrain)
        foldTest = bins[fold]  # list to save current indices for testing
        for epoch in range(epochs):
            X_fold_train = X[foldTrain]
            y_pred = model(X_fold_train)
            l_regularisation = torch.tensor(0, dtype = torch.float)
            if l_reg != 0:
                for param in model.parameters():
                    l_regularisation += torch.norm(param, l_reg)
            y_fold_train = Y[foldTrain]
            loss = loss_func(y_pred, y_fold_train) + (l_regularisation * lamb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        X_fold_test = X[foldTest]
        model.eval()
        y_pred = model(X_fold_test)
        pred = y_pred.cpu().data.numpy()
        pred_values = np.argmax(pred, axis = 1)
        y_predictions = np.append(y_predictions,pred_values)
        if d_out == 2:
            y_fold = np.argmax(Y[foldTest], axis = 1)
            y_fold_order = np.append(y_fold_order, y_fold)
        else:
            y_fold_order = np.append(y_fold_order, Y[foldTest])
    return print_metrics(y_fold_order, y_predictions)

def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

class Net(nn.Module):

    def __init__(self, d_in, h, d_out, l, p):
        super(Net, self).__init__()
        #number of hidden layer nodes
        self.h = h
        #number of input nodes
        self.in_layer = nn.Linear(d_in, h)
        #list to store hidden layers
        self.layers = nn.ModuleList()
        #output layer
        self.out_layer = nn.Linear(h, d_out)
        #dropout layer with p being the probability for dropout
        self.drop_layer = nn.Dropout(p=p)
        #calls a function to add l number of layers
        self.add_layers(l)
        #number of hidden layers
        self.l = l
        #dimensions of the output layer/number of classes
        self.d_out = d_out

    #a function to add l number of hidden layers to a layers list
    def add_layers(self, l):
        for i in range(l):
            self.layers.append(nn.Linear(self.h, self.h))

    #forward propagation function, each layer is looped through
    #the output of one layer is passed to the next
    #return the output of the network as y_pred
    #sigmoid is performed for binary classification and softmax for multi-classification
    def forward(self, x):
        out = torch.relu(self.in_layer(x))
        for i in range(self.l):
             out = torch.relu(self.layers[i](out))
        if(self.d_out == 2):
            y_pred = torch.sigmoid(self.out_layer(out))
        else:
            y_pred = F.softmax(self.out_layer(out), dim= 1)
        return y_pred


def print_metrics(y, y_pred):
    f1 = f1_score(y, y_pred, average='weighted', labels=np.unique(y_pred))
    return f1

