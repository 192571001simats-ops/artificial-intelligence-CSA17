# Tasks 1-2 - learning protocol and decision trees implemented from scratch
import math, random
from collections import Counter
import numpy as np

# ---------------- metrics (shared by all tasks) ----------------
def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m

def prf(y_true, y_pred, positive):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true))
    return sum(prf(y_true, y_pred, l)[2] for l in labels) / len(labels)

# ---------------- impurity ----------------
def entropy(labels):
    n = len(labels)
    return -sum(c / n * math.log2(c / n) for c in Counter(labels).values()) if n else 0.0

def gini(labels):
    n = len(labels)
    return 1 - sum((c / n) ** 2 for c in Counter(labels).values()) if n else 0.0

def majority(labels):
    c = Counter(labels)
    return max(sorted(c), key=lambda k: c[k])          # ties broken by label order (reproducible)

# ---------------- categorical ID3 (desk-check table) ----------------
def info_gain(rows, attr, target, impurity=entropy):
    base = impurity([r[target] for r in rows])
    rem = 0.0
    for v in sorted({r[attr] for r in rows}):
        sub = [r[target] for r in rows if r[attr] == v]
        rem += len(sub) / len(rows) * impurity(sub)
    return base - rem

def split_info(rows, attr):
    return entropy([r[attr] for r in rows])

def id3(rows, attrs, target, parent_majority=None, log=None, depth=0, criterion='gain', min_branch=1):
    """Recursive tree construction. Ties are broken by the order of `attrs`."""
    labels = [r[target] for r in rows]
    if not rows:
        return parent_majority                                     # no examples
    if len(set(labels)) == 1:
        return labels[0]                                           # all one class
    if not attrs:
        return majority(labels)                                    # no attributes left
    scores = []
    for a in attrs:
        branches = Counter(r[a] for r in rows)
        if min(branches.values()) < min_branch:
            continue                                               # forbidden: too-small branch
        g = info_gain(rows, a, target)
        s = g / split_info(rows, a) if criterion == 'ratio' and split_info(rows, a) > 0 else g
        scores.append((s, -attrs.index(a), a, g))
    if not scores:
        return majority(labels)
    best = max(scores)[2]
    if log is not None:
        log.append((depth, len(rows), round(entropy(labels), 3), [(a, round(g, 3), round(s, 3)) for s, _, a, g in scores], best))
    tree = {best: {}}
    for v in sorted({r[best] for r in rows}):
        sub = [r for r in rows if r[best] == v]
        tree[best][v] = id3(sub, [a for a in attrs if a != best], target, majority(labels), log, depth + 1, criterion, min_branch)
    return tree

def predict_id3(tree, row, default='Fail'):
    while isinstance(tree, dict):
        a = next(iter(tree))
        tree = tree[a].get(row[a], default)
    return tree

DESK = [dict(No=i + 1, Attendance=a, Assignments=s, Internal=m, Result=r) for i, (a, s, m, r) in enumerate([
    ('High', 'Done', 'Good', 'Pass'), ('High', 'Done', 'Poor', 'Pass'), ('High', 'Missed', 'Good', 'Pass'),
    ('High', 'Missed', 'Poor', 'Fail'), ('Low', 'Done', 'Good', 'Pass'), ('Low', 'Done', 'Poor', 'Fail'),
    ('Low', 'Missed', 'Good', 'Fail'), ('Low', 'Missed', 'Poor', 'Fail')])]
ATTRS = ['Attendance', 'Assignments', 'Internal']

# ---------------- numeric CART-style tree (binary threshold splits) ----------------
class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'label', 'n')
    def __init__(self, label=None, n=0):
        self.feature = self.threshold = self.left = self.right = None
        self.label, self.n = label, n

class TreeClassifier:
    """Binary decision tree on numeric features. criterion: 'entropy' or 'gini'.
    Stopping: pure node, no examples, depth limit, min_samples_split, or no split
    that reduces impurity."""
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf

    def _imp(self, counts, n):
        if n == 0: return 0.0
        p = counts / n
        if self.criterion == 'gini':
            return 1.0 - np.sum(p * p)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.root = self._grow(X, yi, 0)
        return self

    def _best_split(self, X, y):
        n, k = len(y), len(self.classes_)
        parent = self._imp(np.bincount(y, minlength=k).astype(float), n)
        best = (0.0, None, None)
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind='mergesort')
            xs, ys = X[order, f], y[order]
            left = np.zeros(k); right = np.bincount(ys, minlength=k).astype(float)
            for i in range(n - 1):
                left[ys[i]] += 1; right[ys[i]] -= 1
                if xs[i] == xs[i + 1]: continue
                nl = i + 1
                if nl < self.min_samples_leaf or n - nl < self.min_samples_leaf: continue
                gain = parent - (nl / n) * self._imp(left, nl) - ((n - nl) / n) * self._imp(right, n - nl)
                if gain > best[0] + 1e-12:
                    best = (gain, f, (xs[i] + xs[i + 1]) / 2)
        return best

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = Node(label=int(np.argmax(counts)), n=len(y))
        if len(y) == 0 or counts.max() == len(y): return node
        if self.max_depth is not None and depth >= self.max_depth: return node
        if len(y) < self.min_samples_split: return node
        gain, f, t = self._best_split(X, y)
        if f is None: return node
        mask = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left, node.right = self._grow(X[mask], y[mask], depth + 1), self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        out = []
        for x in np.asarray(X, float):
            node = self.root
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(self.classes_[node.label])
        return np.array(out)

    def depth(self, node=None):
        node = node or self.root
        return 0 if node.feature is None else 1 + max(self.depth(node.left), self.depth(node.right))

    def leaves(self, node=None):
        node = node or self.root
        return 1 if node.feature is None else self.leaves(node.left) + self.leaves(node.right)

def stratified_split(y, fracs, seed):
    """Indices for train/validation/test (fractions sum to 1), stratified by class."""
    rng = np.random.RandomState(seed)
    parts = [[] for _ in fracs]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        cuts = np.cumsum([int(round(f * len(idx))) for f in fracs[:-1]])
        for p, chunk in zip(parts, np.split(idx, cuts)):
            p.extend(chunk.tolist())
    return [np.array(sorted(p)) for p in parts]

def kfold(n_or_y, k, seed):
    y = np.asarray(n_or_y)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx): folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]
