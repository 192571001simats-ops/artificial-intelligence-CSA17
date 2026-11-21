# Task 7 - Diagnose and redesign the Task 4 minimax
import random, time

def count_transpositions(game, s, depth, player):
    """How many of the leaf evaluations in plain minimax repeat a position?"""
    leaves, seen = 0, set()
    def walk(s, d, p):
        nonlocal leaves
        if game.winner(s) or not game.moves(s):
            return
        if d == 0:
            leaves += 1; seen.add(s); return
        for c in game.moves(s):
            walk(game.play(s, c, p), d - 1, 3 - p)
    walk(s, depth, player)
    return leaves, len(seen)

EXACT, LOWER, UPPER = 0, 1, 2

def ordered(game, s):
    """Centre-first move order: central columns take part in the most lines."""
    m = game.cols // 2
    return sorted(game.moves(s), key=lambda c: abs(c - m))

USE = dict(table=True, bounds=True, order=True)      # switches for the ablation study

def search(game, s, depth, alpha, beta, player, me, st, table):
    """Bound-passing minimax with a position table. Gives the same value as plain
    minimax to the same depth, but skips moves that cannot change the decision
    and reuses values of positions reached by a different move order."""
    st.nodes += 1
    key = (s, depth, player)
    hit = table.get(key) if USE['table'] else None
    if hit is not None:
        v, flag = hit
        if flag == EXACT or (flag == LOWER and v >= beta) or (flag == UPPER and v <= alpha):
            st.reused += 1
            return v
    w = game.winner(s)
    if w:
        return (WIN + depth) if w == me else -(WIN + depth)
    ms = ordered(game, s) if USE['order'] else game.moves(s)
    if not ms:
        return 0
    if depth == 0:
        st.evals += 1
        v = evaluate(game, s, me)
        if USE['table']:
            table[key] = (v, EXACT)
        return v
    st.expanded += 1
    a0, b0 = alpha, beta
    if player == me:
        v = -float('inf')
        for c in ms:
            v = max(v, search(game, game.play(s, c, player), depth - 1, alpha, beta, 3 - player, me, st, table))
            alpha = max(alpha, v) if USE['bounds'] else alpha
            if alpha >= beta:
                st.cutoffs += 1; break
    else:
        v = float('inf')
        for c in ms:
            v = min(v, search(game, game.play(s, c, player), depth - 1, alpha, beta, 3 - player, me, st, table))
            beta = min(beta, v) if USE['bounds'] else beta
            if alpha >= beta:
                st.cutoffs += 1; break
    flag = UPPER if v <= a0 else LOWER if v >= b0 else EXACT
    if USE['table']:
        table[key] = (v, flag)
    return v

class Stats2(Stats):
    def __init__(self):
        super().__init__(); self.cutoffs = self.reused = 0

def choose_fast(game, s, depth, me, st=None, table=None):
    st = st or Stats2()
    table = {} if table is None else table
    best, best_v, alpha = None, -float('inf'), -float('inf')
    st.nodes += 1; st.expanded += 1
    for c in (ordered(game, s) if USE['order'] else game.moves(s)):
        v = search(game, game.play(s, c, me), depth - 1, alpha, float('inf'), 3 - me, me, st, table)
        if v > best_v:
            best, best_v = c, v
        alpha = max(alpha, v) if USE['bounds'] else alpha
    st.table = len(table)
    return best, best_v, st

def test_positions(game, n=30, seed=5):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s, p = game.start(), 1
        for _ in range(rng.randint(4, 10)):
            ms = game.moves(s)
            s, p = game.play(s, rng.choice(ms), p), 3 - p
            if game.winner(s):
                break
        if not game.winner(s) and game.moves(s):
            out.append((s, p))
    return out

def compare(game, depth, positions):
    tot = dict(o_nodes=0, o_evals=0, o_ms=0.0, n_nodes=0, n_evals=0, n_ms=0.0, same_value=0, move_ok=0)
    for s, p in positions:
        t = time.perf_counter(); m1, v1, a = choose(game, s, depth, p); tot['o_ms'] += time.perf_counter() - t
        t = time.perf_counter(); m2, v2, b = choose_fast(game, s, depth, p); tot['n_ms'] += time.perf_counter() - t
        tot['o_nodes'] += a.nodes; tot['o_evals'] += a.evals
        tot['n_nodes'] += b.nodes; tot['n_evals'] += b.evals
        tot['same_value'] += (v1 == v2)
        # the redesigned move must be worth exactly the best value under plain minimax
        st = Stats()
        mv = minimax(game, game.play(s, m2, p), depth - 1, 3 - p, p, st)
        tot['move_ok'] += (mv == v1)
    k = len(positions)
    tot['o_ms'] *= 1000 / k; tot['n_ms'] *= 1000 / k
    return tot

def ablation(game, depth, positions):
    variants = [('Plain minimax (Task 4)', None),
                ('Position table only', dict(table=True, bounds=False, order=False)),
                ('Bounds, column order', dict(table=False, bounds=True, order=False)),
                ('Bounds + centre-first', dict(table=False, bounds=True, order=True)),
                ('Full redesign', dict(table=True, bounds=True, order=True))]
    rows = []
    for name, use in variants:
        nodes = evals = 0; t0 = time.perf_counter(); vals = []
        for s, p in positions:
            if use is None:
                m, v, st = choose(game, s, depth, p)
            else:
                USE.update(use); m, v, st = choose_fast(game, s, depth, p)
            nodes += st.nodes; evals += st.evals; vals.append(v)
        USE.update(table=True, bounds=True, order=True)
        rows.append(dict(variant=name, nodes=nodes, evals=evals,
                         ms=(time.perf_counter() - t0) * 1000 / len(positions), values=vals))
    base = rows[0]['values']
    for r in rows:
        r['same_values'] = sum(a == b for a, b in zip(r.pop('values'), base))
    return rows
