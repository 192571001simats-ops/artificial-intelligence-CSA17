# Task 4 - fully connected network with back-propagation in NumPy
import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
ACT = {
    'sigmoid': (sigmoid, lambda z, a: a * (1 - a)),
    'relu': (lambda z: np.maximum(0, z), lambda z, a: (z > 0).astype(float)),
}

class MLP:
    """Layers l = 1..L. Column convention: z(l) = W(l) a(l-1) + b(l), a(l) = g(z(l)).
    W(l) has shape (n_l, n_{l-1}); a batch of m examples is stored as columns."""
    def __init__(self, sizes, hidden='sigmoid', output='sigmoid', loss='mse', seed=0, init_scale=None):
        rng = np.random.RandomState(seed)
        self.sizes, self.hidden, self.output, self.loss = sizes, hidden, output, loss
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            scale = init_scale if init_scale is not None else (np.sqrt(2.0 / n_in) if hidden == 'relu' else np.sqrt(1.0 / n_in))
            self.W.append(rng.randn(n_out, n_in) * scale)
            self.b.append(np.zeros((n_out, 1)))

    def forward(self, X):
        a, cache = X, [(None, X)]
        L = len(self.W)
        for l in range(L):
            z = self.W[l] @ a + self.b[l]
            if l == L - 1:
                if self.output == 'softmax':
                    e = np.exp(z - z.max(axis=0, keepdims=True)); a = e / e.sum(axis=0, keepdims=True)
                else:
                    a = sigmoid(z)
            else:
                a = ACT[self.hidden][0](z)
            cache.append((z, a))
        return a, cache

    def loss_value(self, A, Y):
        m = Y.shape[1]
        if self.loss == 'mse':
            return float(0.5 * np.sum((A - Y) ** 2) / m)
        return float(-np.sum(Y * np.log(np.maximum(A, 1e-12))) / m)

    def backward(self, cache, Y):
        """delta(L) from the loss; delta(l) = (W(l+1)^T delta(l+1)) * g'(z(l)) for l = L-1..1."""
        m = Y.shape[1]
        L = len(self.W)
        grads_W, grads_b, deltas = [None] * L, [None] * L, [None] * L
        zL, aL = cache[L]
        if self.loss == 'mse':
            delta = (aL - Y) * aL * (1 - aL)          # sigmoid output with squared error
        else:
            delta = aL - Y                             # softmax with cross-entropy
        for l in range(L - 1, -1, -1):
            a_prev = cache[l][1]
            deltas[l] = delta
            grads_W[l] = delta @ a_prev.T / m
            grads_b[l] = delta.sum(axis=1, keepdims=True) / m
            if l > 0:
                z, a = cache[l]
                delta = (self.W[l].T @ delta) * ACT[self.hidden][1](z, a)
        return grads_W, grads_b, deltas

    def step(self, gW, gb, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * gW[l]; self.b[l] -= lr * gb[l]

    def predict(self, X):
        A, _ = self.forward(X)
        return A.argmax(axis=0) if A.shape[0] > 1 else (A[0] >= 0.5).astype(int)

def gradient_check(net, X, Y, eps=1e-5):
    A, cache = net.forward(X)
    gW, gb, _ = net.backward(cache, Y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=['multi_index'])
            while not it.finished:
                i = it.multi_index; old = P[i]
                P[i] = old + eps; lp = net.loss_value(net.forward(X)[0], Y)
                P[i] = old - eps; lm = net.loss_value(net.forward(X)[0], Y)
                P[i] = old
                num = (lp - lm) / (2 * eps)
                rel = abs(num - G[i]) / max(1e-12, abs(num) + abs(G[i]))
                worst = max(worst, rel)
                it.iternext()
    return worst

def train(net, X, Y, lr, epochs, batch=32, seed=0, Xv=None, Yv=None, record_grads=False):
    """Mini-batch gradient descent; stops early if the loss becomes NaN/inf."""
    rng = np.random.RandomState(seed)
    m = X.shape[1]
    hist = dict(train_loss=[], val_loss=[], grad_norms=[])
    for ep in range(epochs):
        order = rng.permutation(m)
        norms = None
        for s in range(0, m, batch):
            idx = order[s:s + batch]
            A, cache = net.forward(X[:, idx])
            gW, gb, _ = net.backward(cache, Y[:, idx])
            if record_grads and norms is None:
                norms = [float(np.linalg.norm(g)) for g in gW]
            net.step(gW, gb, lr)
        tl = net.loss_value(net.forward(X)[0], Y)
        hist['train_loss'].append(tl)
        if Xv is not None:
            hist['val_loss'].append(net.loss_value(net.forward(Xv)[0], Yv))
        if record_grads: hist['grad_norms'].append(norms)
        if not np.isfinite(tl):
            break
    return hist

def perceptron(X, y, epochs=100, lr=1.0):
    """Single-layer perceptron (step activation) - X rows are examples."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            p = 1 if xi @ w + b > 0 else 0
            if p != yi:
                w += lr * (yi - p) * xi; b += lr * (yi - p); errors += 1
        if errors == 0: break
    return w, b, np.array([1 if xi @ w + b > 0 else 0 for xi in X])

def desk_check_network():
    net = MLP([2, 2, 1], 'sigmoid', 'sigmoid', 'mse')
    net.W[0] = np.array([[0.5, -0.4], [0.3, 0.8]]); net.b[0] = np.zeros((2, 1))
    net.W[1] = np.array([[0.6, -0.2]]); net.b[1] = np.zeros((1, 1))
    return net
