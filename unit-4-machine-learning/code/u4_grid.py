# Task 5.1-5.2 - substation-state data, Naive Bayes from scratch, SVM comparison
import math, time
import numpy as np

FEATURES = ['load_pct', 'oil_temp', 'ambient', 'voltage_pu', 'fan_on']

def substation_data(n=2000, seed=7, noise=0.03):
    """Synthetic but physically motivated substation records. Hazard (1) when the
    transformer is overloaded AND hot, or when voltage leaves the 0.92-1.08 p.u.
    band (too low or too high) - a rule no single straight line can separate."""
    rng = np.random.RandomState(seed)
    load = rng.uniform(40, 140, n)
    ambient = rng.uniform(10, 48, n)
    fan = (rng.rand(n) < 0.8).astype(float)
    oil = 30 + 0.45 * load + 0.5 * ambient - 8 * fan + rng.normal(0, 5, n)
    volt = 1.0 - 0.0008 * (load - 90) + rng.normal(0, 0.045, n)
    hazard = ((load > 105) & (oil > 85)) | (np.abs(volt - 1.0) > 0.08)
    flip = rng.rand(n) < noise
    y = np.where(flip, ~hazard, hazard).astype(int)
    X = np.column_stack([load, oil, ambient, volt, fan])
    return X, y

class NaiveBayes:
    """Categorical NB on binned features. Parameters by maximum-likelihood counts with
    Laplace (additive) smoothing alpha. Works in log space."""
    def __init__(self, alpha=1.0, bins=6):
        self.alpha, self.bins = alpha, bins

    def _discretise(self, X):
        return np.column_stack([np.digitize(X[:, j], self.edges[j]) for j in range(X.shape[1])])

    def fit(self, X, y):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.bins + 1)[1:-1])) for j in range(X.shape[1])]
        D = self._discretise(X)
        self.classes = np.unique(y)
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        self.log_lik = []                                    # per feature: (classes x values)
        for j in range(D.shape[1]):
            V = len(self.edges[j]) + 1
            table = np.zeros((len(self.classes), V))
            for ci, c in enumerate(self.classes):
                counts = np.bincount(D[y == c, j], minlength=V).astype(float)
                with np.errstate(divide='ignore'):
                    table[ci] = np.log((counts + self.alpha) / (counts.sum() + self.alpha * V))
            self.log_lik.append(table)
        return self

    def joint_log(self, X):
        D = self._discretise(X)
        s = np.tile(self.log_prior, (len(D), 1))
        for j, table in enumerate(self.log_lik):
            s = s + table[:, D[:, j]].T
        return s

    def predict_proba(self, X):
        s = self.joint_log(X)
        m = np.max(s, axis=1, keepdims=True)
        with np.errstate(invalid='ignore'):
            p = np.exp(s - m); p = p / p.sum(axis=1, keepdims=True)
        p[~np.isfinite(p).all(axis=1)] = np.exp(self.log_prior)      # every class impossible: fall back to the prior
        return p

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

def log_loss(y, p1):
    p_true = np.where(np.asarray(y) == 1, p1, 1 - np.asarray(p1, float))
    with np.errstate(divide='ignore'):
        return float(np.mean(-np.log(p_true)))           # inf if a true class got probability 0

def toy_zero_count():
    """Three training examples; 'Fan=off' is never seen with Normal, so P(off|Normal)=0."""
    train = [('High', 'off', 'Hazard'), ('High', 'on', 'Normal'), ('Low', 'on', 'Normal')]
    query = ('High', 'off')
    out = {}
    for alpha in (0, 1):
        res = {}
        for c in ('Hazard', 'Normal'):
            rows = [r for r in train if r[2] == c]
            prior = len(rows) / len(train)
            p_load = (sum(r[0] == query[0] for r in rows) + alpha) / (len(rows) + 2 * alpha)
            p_fan = (sum(r[1] == query[1] for r in rows) + alpha) / (len(rows) + 2 * alpha)
            res[c] = (round(prior, 3), round(p_load, 3), round(p_fan, 3), round(prior * p_load * p_fan, 4))
        out[alpha] = res
    # query 2: a Hazard pattern never seen with Load=Low: both classes may hit zero
    q2 = ('Low', 'off')
    res = {}
    for c in ('Hazard', 'Normal'):
        rows = [r for r in train if r[2] == c]
        p = len(rows) / 3 * (sum(r[0] == q2[0] for r in rows) / len(rows)) * (sum(r[1] == q2[1] for r in rows) / len(rows))
        res[c] = p
    out['unseen'] = res
    return out
