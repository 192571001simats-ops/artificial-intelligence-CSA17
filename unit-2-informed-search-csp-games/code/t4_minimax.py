# Task 4 - Depth-limited minimax with an evaluation function (Connect-k on a gravity board)
import random, time

class Game:
    def __init__(self, cols, rows, k):
        self.cols, self.rows, self.k = cols, rows, k
        self.windows = []                     # every line of k cells, precomputed once
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(k)]
                    if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in cells):
                        self.windows.append(cells)

    def start(self):
        return tuple(tuple() for _ in range(self.cols))      # one tuple (bottom-up) per column

    def moves(self, s):
        return [c for c in range(self.cols) if len(s[c]) < self.rows]

    def play(self, s, c, p):
        return s[:c] + (s[c] + (p,),) + s[c + 1:]

    def cell(self, s, r, c):
        return s[c][r] if r < len(s[c]) else 0

    def winner(self, s):
        for w in self.windows:
            a = self.cell(s, *w[0])
            if a and all(self.cell(s, r, c) == a for r, c in w[1:]):
                return a
        return 0

def evaluate(game, s, me):
    """Open-line score: lines still winnable for one side only, weighted by how
    full they are (3 of 4 >> 2 of 4), plus a small centre-column preference."""
    score, opp, k = 0, 3 - me, game.k
    W = {k - 1: 50, k - 2: 5}
    for w in game.windows:
        vals = [game.cell(s, r, c) for r, c in w]
        m, o = vals.count(me), vals.count(opp)
        if o == 0 and m:
            score += W.get(m, 1)
        elif m == 0 and o:
            score -= W.get(o, 1)
    centre = game.cols // 2
    score += 3 * (s[centre].count(me) - s[centre].count(opp))
    return score

WIN = 10 ** 6

class Stats:
    def __init__(self):
        self.nodes = self.expanded = self.evals = 0

def minimax(game, s, depth, player, me, st):
    """Plain depth-limited minimax: every legal move at every level is examined."""
    st.nodes += 1
    w = game.winner(s)
    if w:
        return (WIN + depth) if w == me else -(WIN + depth)     # prefer faster wins
    ms = game.moves(s)
    if not ms:
        return 0
    if depth == 0:
        st.evals += 1
        return evaluate(game, s, me)
    st.expanded += 1
    vals = [minimax(game, game.play(s, c, player), depth - 1, 3 - player, me, st) for c in ms]
    return max(vals) if player == me else min(vals)

def choose(game, s, depth, me, st=None):
    st = st or Stats()
    best, best_v = None, -float('inf')
    st.nodes += 1; st.expanded += 1
    for c in game.moves(s):
        v = minimax(game, game.play(s, c, me), depth - 1, 3 - me, me, st)
        if v > best_v:
            best, best_v = c, v
    return best, best_v, st

def predicted_nodes(b, d):
    return sum(b ** i for i in range(d + 1)), b ** d

def play_vs_random(game, depth, games=20, seed=0, chooser=choose):
    rng = random.Random(seed)
    res = {'win': 0, 'draw': 0, 'loss': 0}
    for g in range(games):
        ai = 1 if g % 2 == 0 else 2                # AI moves first in half the games
        s, p = game.start(), 1
        while True:
            w = game.winner(s)
            if w or not game.moves(s):
                res['win' if w == ai else 'loss' if w else 'draw'] += 1
                break
            c = chooser(game, s, depth, ai)[0] if p == ai else rng.choice(game.moves(s))
            s, p = game.play(s, c, p), 3 - p
    return res

BOARDS = {'5x4, connect-3': (5, 4, 3), '6x5, connect-4': (6, 5, 4), '7x6, connect-4': (7, 6, 4)}

def play_vs_greedy(game, depth, games=20, seed=0, opening=2, chooser=choose):
    """Stronger yardstick: opponent is depth-1 minimax; the first `opening` plies
    of each game are random so that the 20 games differ."""
    rng = random.Random(seed)
    res = {'win': 0, 'draw': 0, 'loss': 0}
    for g in range(games):
        ai = 1 if g % 2 == 0 else 2
        s, p, ply = game.start(), 1, 0
        while True:
            w = game.winner(s)
            if w or not game.moves(s):
                res['win' if w == ai else 'loss' if w else 'draw'] += 1
                break
            if ply < opening:
                c = rng.choice(game.moves(s))
            elif p == ai:
                c = chooser(game, s, depth, ai)[0]
            else:
                c = choose(game, s, 1, p)[0]
            s, p, ply = game.play(s, c, p), 3 - p, ply + 1
    return res
