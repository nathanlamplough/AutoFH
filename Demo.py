import AutoFH as af
import numpy as np
import pandas as pd


iris = pd.read_csv('Data/iris.csv', delimiter= ',', quoting= 3)
y = np.asarray(iris.iloc[:,-1])
X = np.asarray(iris.drop(iris.columns[-1], axis=1))

print(af.AutoFH(X, y , 100, True))