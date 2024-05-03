import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier as DTC

X, y = load_digits(return_X_y=True)

class Random(object):
    def predict(self, X):
        return np.random.choice(10, size=X.shape[0])

class Constant(object):
    def __init__(self, val):
        self.val = val

    def predict(self, X):
        return np.ones(X.shape[0]) * self.val


models = [Random() for _ in range(5)]
#models = [Constant(j) for j in range(5)]

def fit(X_trn, y_trn):
    if X_trn.shape[0] == 0:
        return Random()
    if X_trn.shape[0] == 1:
        return Constant(y_trn[0])
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.2)
    models = []
    scores = []
    for d in range(1, 30):
        models.append(DTC(max_depth=d).fit(X_trn, y_trn))
        scores.append(models[-1].score(X_val, y_val))
    return models[np.argmax(scores)]


def assign(idxs, models, null=False):
    correct = []
    for model in models:
        correct.append(model.predict(X[idxs]) == y[idxs])
    correct = np.array(correct)
    assignments = np.zeros(len(idxs))
    for i in range(len(idxs)):
        candidates = np.where(correct[:, i] == True)[0]
        if candidates.size == 0:
            assignments[i] = -1 if null else np.random.choice(len(models))
        else:
            assignments[i] = np.random.choice(candidates)
    return assignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=30, help='number of customers sampled at each round')
    parser.add_argument('-T', type=int, default=40, help='number of total rounds')
    parser.add_argument('-t', type=int, default=0, help='number of monopoly rounds')
    args = parser.parse_args()

    data = np.zeros((len(models), len(X)))  # 1 if learner i has the data of customer j, 0 otherwise

    for _ in range(args.T):
        idxs = np.random.choice(X.shape[0], size=args.k, replace=False)
        idxs.sort()
        assignments = assign(idxs, models)

        if _ < args.t:
            assignments = np.zeros(idxs.size)

        for i in range(len(models)):
            data[i][idxs[np.where(assignments == i)[0]]] = 1
            model_idxs = np.where(data[i] == 1)[0]
            models[i] = fit(X[model_idxs], y[model_idxs])

    assignments = assign(np.arange(X.shape[0]), models, null=True)
    print('*' * 20)
    for i in range(-1, len(models)):
        counts = np.zeros(10)
        for label in y[np.where(assignments == i)]:
            counts[label] += 1
        if i == -1:
            print('all models wrong')
        else:
            print(str(models[i]) + ' correct')
        print(counts)
        print()
    print('*' * 20)
