# Task 3 - How tree structure and move order drive alpha-beta work
import random

def build(b, d, values):
    """Uniform tree as nested lists; leaves taken from `values` in order."""
    it = iter(values)
    def rec(k):
        return next(it) if k == 0 else [rec(k - 1) for _ in range(b)]
    return rec(d)

def minimax_value(node, maximizing):
    if not isinstance(node, list):
        return node
    vals = [minimax_value(c, not maximizing) for c in node]
    return max(vals) if maximizing else min(vals)

def reorder(node, maximizing, best_first):
    """Sort children by their true minimax value: best-first or worst-first."""
    if not isinstance(node, list):
        return node
    kids = [reorder(c, not maximizing, best_first) for c in node]
    kids.sort(key=lambda c: minimax_value(c, not maximizing),
              reverse=(maximizing == best_first))
    return kids

class Counter:
    def __init__(self):
        self.nodes = self.leaves = self.cutoffs = 0

def minimax_tree(node, maximizing, cnt):
    cnt.nodes += 1
    if not isinstance(node, list):
        cnt.leaves += 1
        return node
    vals = [minimax_tree(c, not maximizing, cnt) for c in node]
    return max(vals) if maximizing else min(vals)

def alphabeta(node, maximizing, cnt, alpha=float('-inf'), beta=float('inf')):
    cnt.nodes += 1
    if not isinstance(node, list):
        cnt.leaves += 1
        return node
    if maximizing:
        v = float('-inf')
        for i, c in enumerate(node):
            v = max(v, alphabeta(c, False, cnt, alpha, beta))
            alpha = max(alpha, v)
            if alpha >= beta:
                if i < len(node) - 1:
                    cnt.cutoffs += 1          # remaining siblings are never visited
                break
        return v
    v = float('inf')
    for i, c in enumerate(node):
        v = min(v, alphabeta(c, True, cnt, alpha, beta))
        beta = min(beta, v)
        if alpha >= beta:
            if i < len(node) - 1:
                cnt.cutoffs += 1
            break
    return v

def no_cutoff_tree(b, d):
    """Engineered so that no cutoff can occur: every child's whole subtree lies in
    its own slice of the parent's value window. MAX children rise, MIN children
    fall, so a new value never crosses the bound inherited from earlier siblings."""
    def gen(k, maximizing, lo, hi):
        if k == 0:
            return (lo + hi) / 2
        w = (hi - lo) / b
        parts = [(lo + i * w, lo + (i + 1) * w) for i in range(b)]
        if not maximizing:
            parts.reverse()
        return [gen(k - 1, not maximizing, p, q) for p, q in parts]
    t = gen(d, True, 0.0, 1.0)
    flat = []
    def walk(n):
        if isinstance(n, list):
            for c in n: walk(c)
        else:
            flat.append(n)
    walk(t)
    rank = {v: i for i, v in enumerate(sorted(flat))}       # integer leaf values
    def relabel(n):
        return [relabel(c) for c in n] if isinstance(n, list) else rank[n]
    return relabel(t)

def predicted_best(b, d):
    """Knuth-Moore minimal tree: level k holds b^ceil(k/2) + b^floor(k/2) - 1 nodes."""
    return sum(b ** ((k + 1) // 2) + b ** (k // 2) - 1 for k in range(d + 1)), b ** ((d + 1) // 2) + b ** (d // 2) - 1

def predicted_full(b, d):
    return (b ** (d + 1) - 1) // (b - 1), b ** d

def trees(b=4, d=4, seed=11):
    rng = random.Random(seed)
    leaves = list(range(b ** d)); rng.shuffle(leaves)
    base = build(b, d, leaves)
    return {
        'Best-first order': reorder(base, True, True),
        'Engineered no-cutoff': no_cutoff_tree(b, d),
        'Random order': base,
        'Equal leaf values': build(b, d, [5] * b ** d),
    }

def run_task3(b=4, d=4):
    rows = []
    for name, t in trees(b, d).items():
        m, a = Counter(), Counter()
        v1, v2 = minimax_tree(t, True, m), alphabeta(t, True, a)
        rows.append(dict(tree=name, value=v1, same=v1 == v2, mm_nodes=m.nodes, ab_nodes=a.nodes,
                         ab_leaves=a.leaves, cutoffs=a.cutoffs, saved=round(100 * (1 - a.nodes / m.nodes), 1)))
    return rows

def random_order_spread(b=4, d=4, trials=200):
    out = []
    for s in range(trials):
        rng = random.Random(1000 + s)
        lv = list(range(b ** d)); rng.shuffle(lv)
        c = Counter(); alphabeta(build(b, d, lv), True, c); out.append(c.nodes)
    out.sort()
    return dict(min=out[0], median=out[len(out) // 2], max=out[-1], mean=sum(out) / len(out))
