# Task 3 - EM for a 1-D Gaussian mixture: faulty (unnormalised responsibilities) and repaired
import math
import numpy as np

def normal_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

def log_likelihood(X, pi, mu, var):
    dens = sum(pi[k] * normal_pdf(X, mu[k], var[k]) for k in range(len(pi)))
    return float(np.sum(np.log(np.maximum(dens, 1e-300))))

def em(X, pi, mu, var, T, repaired=True, var_floor=1e-6, trace=None):
    """One E-step and M-step per iteration. The only difference between the two
    versions is the normalisation of r[i][k] over k."""
    X = np.asarray(X, float)
    pi, mu, var = np.array(pi, float), np.array(mu, float), np.array(var, float)
    K = len(pi)
    hist = [dict(ll=log_likelihood(X, pi, mu, var), wsum=float(pi.sum()))]
    for t in range(T):
        r = np.stack([pi[k] * normal_pdf(X, mu[k], var[k]) for k in range(K)], axis=1)    # n x K
        if repaired:
            r = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-300)                     # THE REPAIR
        if trace is not None and t == 0:
            trace['r'] = r.copy()
        Nk = r.sum(axis=0)
        mu = (r * X[:, None]).sum(axis=0) / Nk
        var = np.maximum((r * (X[:, None] - mu) ** 2).sum(axis=0) / Nk, var_floor)
        pi = Nk / len(X)
        if trace is not None and t == 0:
            trace.update(Nk=Nk.copy(), mu=mu.copy(), var=var.copy(), pi=pi.copy())
        hist.append(dict(ll=log_likelihood(X, pi, mu, var), wsum=float(pi.sum())))
    return pi, mu, var, hist

def make_data(seed=0, n=300):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.normal(-2, 0.8, int(n * 0.4)), rng.normal(3, 1.2, n - int(n * 0.4))])

def random_init(X, K, rng):
    return np.full(K, 1.0 / K), rng.choice(X, K, replace=False), np.full(K, np.var(X))

def many_inits(X, K=2, T=60, runs=20, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for r in range(runs):
        pi0, mu0, var0 = random_init(X, K, rng)
        res = {}
        for name, rep in (('faulty', False), ('repaired', True)):
            pi, mu, var, h = em(X, pi0, mu0, var0, T, rep)
            lls = [x['ll'] for x in h]
            drops = sum(1 for a, b in zip(lls, lls[1:]) if b < a - 1e-9)
            res[name] = dict(final_ll=lls[-1], decreases=drops, wsum=h[-1]['wsum'], mu=np.sort(mu))
        rows.append(res)
    return rows
