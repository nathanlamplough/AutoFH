# AutoFH
AutoFH is a tool for automated feature and hyperparameter selection for artficial neural networks. Currently AutoFH supports binary and multi-classifications problems. AutoFH by completing the following steps. A more detailed explanation can be seen in the report.

1. A dataset *D* (`X , y_`) is inputted by the user along with iterations `iter` for Bayesian Optimisation runs.

2. Feature selection is performed using 3 different techniques (genetic algorithm, univariate, correlation), the feature subsets are        stored.

3. Meta features are extracted from *D* and used to find the most similar dataset from the metafeature SQLite database using Euclidean        distance. The closest datasets optimum hyperparameters are then retrieved.

4. Bayesian Optimisation is then warm started using the hyperparameters from step 3. (The feature selection subsets are selected as        hyperparameter).

5. Bayesian Optimisation is run for `iter` iterations and outputs the optimum hyperparameters. These hyperparameters are stored with *D*s    meta-features in the SQLite database.

6. The optimum hyperparameters are returned to the user.


## Getting Started

These instructions will get you a copy of AutoFH up and running on your local machine.

### Prerequisites

Anaconda - https://www.anaconda.com/distribution/#windows (Includes sci-kit learn, numpy and pandas)

Bayes Opt - `pip install bayesian-optimization`

PyTorch - `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` (Anaconda will automatically install prequisites for PyTorch)

SQLAlchemy - `pip install SQLAlchemy`

Progress Bar - `pip install progress`

### Installing

To use, clone the repository and import the AutoFH.py file into your project. 

### AutoFH Usage
The AutoFH function takes 4 parameters. `def AutoFH(X, y_, iter, norm):`
* `X` - the training data (numpy array)
* `y_` - the labels/classes (numpy array)
* `iter` - the number of iterations to run Bayesian Optimisation (int)
* `norm` - bool value that decides if 0-1 normalisation is done on X

X values must be numerical and > 0 this is because the Univariate feature selection can not handle negative values. Therefore it is advised to normalise your data before using AutoFH or setting `norm` to `True`

### The Hyperparameters
AutoFH optmises 9 hyperparameters with the following default ranges (min , max):

* Learning Rate (0.0001,0.1)
* Dropout (0,1)
* Epochs (1,1500)
* Momentum (0,1)
* Layers (0,5)
* Nodes (#features,#classes)
* L Regularisation (0,2.99)
* Regularisation Parameter (0,1)
* Feature Selection Algorithm (0,3.99)

The reasoning for these defaults can be seen in the report.

These defaults can be change by changing the pbounds dictionary inside the Bayesian_optmisation function inside the Auto_FH.py file.

For example if you wanted to change the epoch range from (1,1500) to (1,10000) and the layers from (0,5) to (0,10) the pbounds dictionary would change from:
```
pbounds = {'l_rate': (0.0001, 0.1), 'dropout': (0, 1), 'epochs' : (1, 1500), 'momentum' : (0, 1), 'layers' : (0, 5), 'nodes' : (d_out, d_in), 'l_reg' : (0,2.99), 'reg_lambda' : (0,1), 'FSA' : (0,3.99)}
```
*to*
```
pbounds = {'l_rate': (0.0001, 0.1), 'dropout': (0, 1), 'epochs' : (1, 10000), 'momentum' : (0, 1), 'layers' : (0, 10), 'nodes' : (d_out, d_in), 'l_reg' : (0,2.99), 'reg_lambda' : (0,1), 'FSA' : (0,3.99)}
```


### Feature Selection
The feature selection algorithm (FSA) is chosen through Bayesian Optmisation where 0 , 1 , 2 and 3 correspond to different FSAs. These are as follows:

* 0 - No FSA
* 1 - Genetic Algortihm
* 2 - Correlation FSA
* 3 - Univariate FSA

Additional FSAs can easily be added. 

To do this first add the FSA output to the `fsa_dict` in AutoFH in the AutoFH.py file.

Next extend the FSA range in the pbounds dictionary inside the Bayesian_optimisation function by 1.

Example: 

```
global fsa_dict
fsa_dict = fs.select_features(df, x, y_)
fsa_dict.append(new_fsa_set)
```

```
pbounds = {'l_rate': (0.0001, 0.1), 'dropout': (0, 1), 'epochs': (1, 1500), 'momentum': (0, 1), 'layers': (0, 5),
               'nodes': (d_out, d_in), 'l_reg': (0, 2.99), 'reg_lambda': (0, 1), 'FSA': (0, 3.99)}
```
changes to 
```
pbounds = {'l_rate': (0.0001, 0.1), 'dropout': (0, 1), 'epochs': (1, 1500), 'momentum': (0, 1), 'layers': (0, 5),
               'nodes': (d_out, d_in), 'l_reg': (0, 2.99), 'reg_lambda': (0, 1), 'FSA': (0, 4.99)}
```


### Demo
A demo.py file is included with this release to allow a quickstart and showcase the simiplicity of AutoFH. In the demo you will observe a high F1-Score after 1 iteration due to warm starting. You will then see the F1-Score be low for a few iterations as Bayesian Optmisation is exploring the hyperparameter space. After this exploration it will start to focus around the optimum parameters and produce higher scores.

demo.py contains the following:

```
import AutoFH as af
import numpy as np
import pandas as pd


iris = pd.read_csv('Data/iris.csv', delimiter= ',', quoting= 3)
y = np.asarray(iris.iloc[:,-1])
X = np.asarray(iris.drop(iris.columns[-1], axis=1))

af.AutoFH(X, y , 100, True)
```
The iris dataset is included so this code will run out of the box.

An `iter` of 100 is reccomended, increase iter according to the complexity of the problem

An important note is that the total iterations is iter + 11. This is because 1 iterations is done to warm start Bayesian Optmisation and 10 iterations are done as initial points for Bayesian Optimisation

AutoFH will return a list of dictionaries of the most optimum hyperparameters sets. 

### Unit Testing

This release comes with unit tests in the "Unit_Test" folder. This is for marking purposes to demonstrate testing was done throughout this project.

To run these tests first install pytest - `pip install -U pytest`

Next navigate to the "Unit_Test" folder in the terminal. 

To run a test file type `pytest AutoFH_Unit_Tests.py` (Replace `AutoFH_Unit_Tests.py` with the file you'd like to test)

## Built With

* [Bayes_Opt](https://github.com/fmfn/BayesianOptimization) - Bayesian Optmisation framework
* [PyTorch](https://pytorch.org/) - Machine learning framework
* [sci-kit learn](https://scikit-learn.org/) - Machine learning framework
* [SQLAlchemy](https://www.sqlalchemy.org/) - Database framework

## Version

1.0.0 using [SemVer](http://semver.org/) for versioning. 

## Author
**Nathan Lamplough** - [Git Repository](https://github.com/nathanlamplough)

* Designed as part of a final year project for Bsc Computer Science at Goldsmiths University of London

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/nathanlamplough/AutoFH/blob/master/LICENSE) file for details


