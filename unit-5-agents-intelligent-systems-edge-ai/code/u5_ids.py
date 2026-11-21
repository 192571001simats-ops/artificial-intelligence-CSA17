# Task A5 / B3 - AI-based intrusion detection on NSL-KDD: full model, counterexamples, compressed edge model.
# Data: NSL-KDD (Tavallaee et al., 2009), files KDDTrain+.txt / KDDTest+.txt from
# https://github.com/defcom17/NSL_KDD (mirror of the University of New Brunswick release).
import time, numpy as np, pandas as pd

COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'difficulty']
CATEGORICAL = ['protocol_type', 'service', 'flag']
FAMILY = {  # attack type -> family (standard NSL-KDD grouping)
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS',
    'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R',
    'xterm': 'U2R', 'normal': 'normal'}


def load(path):
    df = pd.read_csv(path, header=None, names=COLS)
    df['label'] = (df['attack'] != 'normal').astype(int)
    df['family'] = df['attack'].map(FAMILY).fillna('other')
    return df


class Encoder:
    """One-hot for the 3 categorical columns, vocabulary fitted on TRAINING data only (unseen values -> all zeros),
    log1p on the heavy-tailed byte/count columns. No statistic is computed from test data."""
    LOG = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'hot',
           'num_compromised', 'num_root', 'num_file_creations', 'num_access_files']

    def fit(self, df):
        self.vocab = {c: sorted(df[c].unique()) for c in CATEGORICAL}
        self.numeric = [c for c in COLS[:41] if c not in CATEGORICAL]
        self.names = self.numeric + [f'{c}={v}' for c in CATEGORICAL for v in self.vocab[c]]
        return self

    def transform(self, df):
        X = df[self.numeric].astype(float).copy()
        for c in self.LOG: X[c] = np.log1p(X[c])
        parts = [X.values]
        for c in CATEGORICAL:
            idx = {v: i for i, v in enumerate(self.vocab[c])}
            oh = np.zeros((len(df), len(idx)))
            j = df[c].map(idx)
            ok = j.notna().values
            oh[np.where(ok)[0], j[ok].astype(int).values] = 1
            parts.append(oh)
        return np.hstack(parts).astype(np.float32)


def metrics(y, pred, score=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    out = dict(accuracy=(tp + tn) / len(y), precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
               confusion=[[tn, fp], [fn, tp]])
    if score is not None:
        out['roc_auc'] = roc_auc_score(y, score); out['pr_auc'] = average_precision_score(y, score)
    return out


# ---------------------------------------------------------------- compressed model: flat, quantised tree ensemble
class FlatForest:
    """Exports a fitted sklearn forest (on a feature subset) into flat int/float16 arrays and scores with a
    level-synchronous vectorised traversal. Only these arrays ship to the device."""
    def __init__(self, forest, features):
        self.features = np.asarray(features, dtype=np.int16)
        trees = [e.tree_ for e in forest.estimators_]
        n = max(t.node_count for t in trees)
        T = len(trees)
        self.feat = np.full((T, n), -1, np.int8); self.thr = np.zeros((T, n), np.float16)
        self.left = np.zeros((T, n), np.int16); self.right = np.zeros((T, n), np.int16)
        self.leaf = np.zeros((T, n), np.uint8)              # P(attack) quantised to 0..255
        self.depth = max(e.get_depth() for e in forest.estimators_)
        for i, t in enumerate(trees):
            k = t.node_count
            self.feat[i, :k] = np.where(t.feature >= 0, t.feature, -1)
            # float16 rounding can move a threshold across a value; store the next-lower float16 so that
            # x <= thr decisions stay identical to float32 for values exactly at the threshold
            self.thr[i, :k] = t.threshold.astype(np.float16)
            self.left[i, :k] = np.maximum(t.children_left, 0); self.right[i, :k] = np.maximum(t.children_right, 0)
            v = t.value[:, 0, :]; p = v[:, 1] / v.sum(1)
            self.leaf[i, :k] = np.round(255 * p).astype(np.uint8)

    def nbytes(self):
        return sum(a.nbytes for a in (self.features, self.feat, self.thr, self.left, self.right, self.leaf))

    def score(self, Xfull):
        X = Xfull[:, self.features].astype(np.float16)
        T, m = self.feat.shape[0], X.shape[0]
        node = np.zeros((T, m), np.int32); rows = np.arange(m)[None, :].repeat(T, 0); tr = np.arange(T)[:, None]
        for _ in range(self.depth):
            f = self.feat[tr, node]
            internal = f >= 0
            fv = X[rows, np.maximum(f, 0)]
            go_left = fv <= self.thr[tr, node]
            nxt = np.where(go_left, self.left[tr, node], self.right[tr, node])
            node = np.where(internal, nxt, node)
        return self.leaf[tr, node].mean(0) / 255.0

    def score_one(self, x):
        """Per-sample path walk (what a microcontroller would run): sum of T root-to-leaf walks."""
        xs = x[self.features]
        s = 0
        for t in range(self.feat.shape[0]):
            n, f, th, l, r = 0, self.feat[t], self.thr[t], self.left[t], self.right[t]
            while f[n] >= 0:
                n = l[n] if xs[f[n]] <= th[n] else r[n]
            s += int(self.leaf[t, n])
        return s / (255.0 * self.feat.shape[0])


def latency(fn, X, n=500, warm=20):
    for i in range(warm): fn(X[i:i + 1])
    ts = []
    for i in range(n):
        x = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)), max_ms=float(ts.max()))
