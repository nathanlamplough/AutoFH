import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
import sys
from progress.bar import Bar

warnings.filterwarnings('ignore')

class Genetic_Algorithm():
    def __init__(self, X , Y, p, pop_size, iter):
        #probability of turning off a feature
        self.p = p
        self.X = X
        self.Y = Y
        self.n_features = X.shape[1]
        #the size of the population
        self.pop_size = pop_size
        #the best feature set
        self.chromosomes_best = []
        #the best score
        self.scores_best = []
        #the average score per iteration
        self.scores_avg = []
        #the probability of changing a feature in the mutation phase
        self.mutation_rate = 1 / self.n_features
        #the number of iterations it is ran for
        self.iterations = iter

    #runs the genetic algorithm for specified iterations
    def compute_GA(self):
        population = np.empty(0)
        last_score = 0
        bar = Bar('Processing', max=50)
        for i in range(self.iterations):
            bar.next()
            self.generate()
            last_score = self.scores_best[i]

        bar.finish()

    #intialises the first population by switching off some
    #features
    def intialisation(self, p, n_features, pop_size):
        pop_switches = []
        while(len(pop_switches) != pop_size):
            switches = np.random.rand(n_features) < p
            if switches.any():
                pop_switches.append(switches)
        return np.asarray(pop_switches)

    #measures the fitness of each population using naive bayes
    def fitness(self, X, Y, pop_size, switches):
        clf = GaussianNB()
        fitnesses = np.empty(0)
        for i in range(pop_size):
            x = X[:,switches[i]]

            fitness_score = np.mean(cross_val_score(clf, x, Y,
                                cv=5,
                                scoring="f1_weighted"))

            fitnesses = np.append(fitnesses, fitness_score)

        sorted_indices = np.argsort(fitnesses)

        return fitnesses[sorted_indices], switches[sorted_indices]

    #selects a combination of the best and random features sets
    #from the population
    def selection(self, pop_size, switches_, fitness_scores):
        selected = []
        switches = switches_
        n = int(pop_size / 2)
        e = int(n / 2)
        r = n - e
        for i in range(e, 0 , -1):
            selected.append(switches[i])
            switches = np.delete(switches, i, axis = 0)
            fitness_scores = np.delete(fitness_scores, i, axis = 0)
        r_selection = self.roulette(switches, r, fitness_scores)
        for i in r_selection:
            selected.append(i)
        return selected

    #used as part of selection to choose random feature subsets
    #with a probability based on fitness
    def roulette(self, switches, n, fitness_scores):
        selected = []
        probs = []
        indices = []
        for i in range(n):
            #probabilities of each feature set from the population
            #calculated
            probs = fitness_scores / fitness_scores.sum()
            indices = np.arange(fitness_scores.shape[0])
            #a random index value is chosen with probabilties based on fiteness used
            choice = np.random.choice(indices, p = probs)
            selected.append(choice)
            #the chosen index value feature set is deleted so it cant be rechosen
            fitness_scores = np.delete(fitness_scores, choice, axis = 0)
        return switches[selected]

    #combined features subsets from the population for the next
    #iteration
    def crossover(self, switches, pop_size):
        population_next = []
        switches = list(switches)
        for i in range(int(len(switches)/2)):
            for j in range(pop_size):
                chromosome1, chromosome2 = switches[i], switches[len(switches) - 1 - i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)
                if len(population_next) == pop_size:
                    break
            if len(population_next) == pop_size:
                break
        return population_next

    #randomly turns off and on features that have crossovered
    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            switch = population[i]
            if np.random.rand() < self.mutation_rate:
                switch = np.random.rand(len(switch)) < 0.05
            population_next.append(switch)
        return population_next

    #runs all the functions together, this is 1 iteration
    def generate(self):
        # Selection, crossover and mutation
        population = self.intialisation(self.p, self.n_features, self.pop_size)
        scores_sorted, population_sorted = self.fitness(self.X, self.Y, self.pop_size, population)
        population = self.selection(self.pop_size, population_sorted, scores_sorted)
        population = self.crossover(population, self.pop_size)
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[-1])
        self.scores_avg.append(np.mean(scores_sorted))

        return population

    #sorts the best scores and returns the best feature set
    def get_best(self):
        best_index = np.argmax(self.scores_best)
        best_mask = self.chromosomes_best[best_index]
        return self.X[:,best_mask]


def correlation_feature_selection(data_df, threshold, y_index):
    corr_matrix = data_df.corr()
    y_values = np.asarray(corr_matrix.iloc[y_index])
    indices = np.where(abs(y_values[0]) < threshold)
    for i in indices:
        data_df = data_df.drop(data_df.columns[i], axis= 1)
    return data_df

def univariate_selection(X, y, dim):
    best_x = []
    best_score = 0
    clf = GaussianNB()
    for i in range(dim, 2 , -1):
        x_new = SelectKBest(score_func=chi2, k=i).fit_transform(X, y)

        fitness_score = np.mean(cross_val_score(clf, X, y,
                                                    cv=5,
                                                    scoring="f1_weighted"))

        if(fitness_score > best_score):
            best_x = x_new
            best_score = fitness_score
    return best_x

def recursive_feature_selection(X, Y):
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
          scoring='accuracy')
    rfecv.fit(X,Y)
    mask = rfecv.get_support()
    return X[:,mask]

def select_features(df, x, y):

    print('Performing Feature Selection')
    Y = y
    X = x

    ga = Genetic_Algorithm(X, Y, 0.3, X.shape[1]*10, 50)
    ga.compute_GA()
    X_ga = ga.get_best()

    X_corr = np.asarray(correlation_feature_selection(df, 0.3, -1))

    X_uni = univariate_selection(X, Y, X.shape[1])


    return (X, X_ga, X_corr, X_uni)


