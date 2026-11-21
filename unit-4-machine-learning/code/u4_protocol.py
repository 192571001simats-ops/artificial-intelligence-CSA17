# Task 1 - the faulty protocol versus the repaired protocol
import numpy as np
from u4_tree import TreeClassifier, stratified_split, kfold, accuracy, prf, confusion, majority

def faulty_protocol(X, y, model):
    """TrainAndTest as submitted: train and test on the same data."""
    model.fit(X, y)
    return accuracy(y, model.predict(X))

def cross_val_accuracy(X, y, make_model, k=5, seed=0):
    folds = kfold(y, k, seed)
    scores = []
    for i in range(k):
        val = folds[i]; trn = np.concatenate([folds[j] for j in range(k) if j != i])
        m = make_model().fit(X[trn], y[trn])
        scores.append(accuracy(y[val], m.predict(X[val])))
    return float(np.mean(scores)), float(np.std(scores))

def repaired_protocol(X, y, depths=(1, 2, 3, 4, 5, 6, 8, None), k=5, seed=42, positive=1):
    """Split once (80/20, stratified, fixed seed); choose max_depth by k-fold CV on the
    training part only; refit; evaluate once on the untouched test set; compare with
    the majority-class baseline."""
    train, test = stratified_split(y, [0.8, 0.2], seed)
    cv = {d: cross_val_accuracy(X[train], y[train], lambda d=d: TreeClassifier('entropy', max_depth=d), k, seed) for d in depths}
    best = max(depths, key=lambda d: (round(cv[d][0], 6), -(d or 99)))     # prefer the shallower tree on ties
    model = TreeClassifier('entropy', max_depth=best).fit(X[train], y[train])
    pred = model.predict(X[test])
    base_label = majority(list(y[train]))
    base = [base_label] * len(test)
    p, r, f = prf(list(y[test]), list(pred), positive)
    bp, br, bf = prf(list(y[test]), base, positive)
    return dict(cv=cv, best_depth=best, test_acc=accuracy(y[test], pred), precision=p, recall=r, f1=f,
                cm=confusion(list(y[test]), list(pred), [0, 1]), baseline_acc=accuracy(y[test], base),
                baseline_f1=bf, train_acc=accuracy(y[train], model.predict(X[train])))
