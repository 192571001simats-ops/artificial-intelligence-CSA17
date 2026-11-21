# Task 2 - Online DFS agent and LRTA* in an unmapped facility
import random
from collections import deque

MOVES = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
BACK = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

def facility(cells, seed, open_dead_ends=0.0):
    """Perfect maze (many dead ends); a share of dead ends is then opened up."""
    rng = random.Random(seed)
    H = W = 2 * cells + 1
    g = [['#'] * W for _ in range(H)]
    stack, seen = [(0, 0)], {(0, 0)}
    g[1][1] = '.'
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in MOVES.values()
              if 0 <= r + dr < cells and 0 <= c + dc < cells and (r + dr, c + dc) not in seen]
        if not nb:
            stack.pop(); continue
        nr, nc = rng.choice(nb)
        g[r + nr + 1][c + nc + 1] = '.'
        g[2 * nr + 1][2 * nc + 1] = '.'
        seen.add((nr, nc)); stack.append((nr, nc))
    for r in range(cells):
        for c in range(cells):
            cell = (2 * r + 1, 2 * c + 1)
            walls = [d for d, (dr, dc) in MOVES.items()
                     if 1 <= cell[0] + 2 * dr < H - 1 and 1 <= cell[1] + 2 * dc < W - 1
                     and g[cell[0] + dr][cell[1] + dc] == '#']
            if len(walls) == 3 and rng.random() < open_dead_ends:      # dead end: knock one wall
                dr, dc = MOVES[rng.choice(walls)]
                g[cell[0] + dr][cell[1] + dc] = '.'
    return g

def dead_ends(g):
    return sum(1 for r in range(1, len(g), 2) for c in range(1, len(g), 2)
               if sum(g[r + dr][c + dc] == '.' for dr, dc in MOVES.values()) == 1)

def sense(g, s):
    """The rover can only sense which neighbouring cells are open."""
    return [a for a, (dr, dc) in MOVES.items() if g[s[0] + dr][s[1] + dc] != '#']

def step(s, a):
    return (s[0] + MOVES[a][0], s[1] + MOVES[a][1])

def bfs_dist(g, s):
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for a in sense(g, u):
            v = step(u, a)
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist

def online_dfs(g, start, goal):
    """Online DFS: the rover knows its own coordinates but not the map.
    It enters an unvisited open neighbour if one exists, otherwise it drives
    back to the cell it came from. Every move is a physical move."""
    path, visited, moves = [start], {start}, 0
    untried = {start: sense(g, start)}
    while path:
        s = path[-1]
        if s == goal:
            return dict(found=True, moves=moves, revisits=moves + 1 - len(visited), explored=len(visited))
        nxt = None
        while untried[s]:
            v = step(s, untried[s].pop(0))
            if v not in visited:
                nxt = v; break
        if nxt is not None:
            path.append(nxt); visited.add(nxt); untried[nxt] = sense(g, nxt); moves += 1
        else:
            path.pop()
            if path:
                moves += 1                            # physical backtrack to the parent
    return dict(found=False, moves=moves, revisits=moves + 1 - len(visited), explored=len(visited))

def lrta_star(g, start, goal, H, cap=200000):
    """One LRTA* trial; H is the learned cost-to-go table shared across trials."""
    h = lambda x: H.get(x, abs(x[0] - goal[0]) + abs(x[1] - goal[1]))
    s, moves, visited = start, 0, {start}
    while s != goal and moves < cap:
        options = [(1 + h(step(s, a)), a) for a in sense(g, s)]
        best, a = min(options)
        H[s] = max(h(s), best)                        # learn: raise the estimate
        s = step(s, a); moves += 1; visited.add(s)
    return dict(found=s == goal, moves=moves, revisits=moves + 1 - len(visited))

def sealed_dock(g, target):
    """Make the dock unreachable: wall in every open neighbour of the dock cell."""
    g2 = [row[:] for row in g]
    for dr, dc in MOVES.values():
        r, c = target[0] + dr, target[1] + dc
        if 0 < r < len(g2) - 1 and 0 < c < len(g2) - 1:
            g2[r][c] = '#'
    return g2

BANDS = [(1, 25), (26, 75), (76, 150), (151, 10 ** 9)]

def distance_experiment(seeds=range(5), cells=20, per_band=10):
    rows = []
    for lo, hi in BANDS:
        crs, mv, rv, n = [], 0, 0, 0
        for sd in seeds:
            g = facility(cells, sd, 0.3)
            d = bfs_dist(g, (1, 1))
            cand = sorted(c for c in d if lo <= d[c] <= hi and c[0] % 2 and c[1] % 2)
            rng = random.Random(100 + sd)
            for t in rng.sample(cand, min(per_band, len(cand))):
                r = online_dfs(g, (1, 1), t)
                crs.append(r['moves'] / d[t]); mv += r['moves']; rv += r['revisits']; n += 1
        rows.append(dict(band=f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}', targets=n, moves=mv / n,
                         revisits=rv / n, cr_mean=sum(crs) / n, cr_max=max(crs)))
    return rows

def unreachable_experiment(seeds=range(5)):
    rows = []
    for cells in (10, 20, 40):
        for sd in seeds[:1]:
            g = facility(cells, sd, 0.3)
            far = max(bfs_dist(g, (1, 1)).items(), key=lambda kv: kv[1])[0]
            g2 = sealed_dock(g, far)
            reach = len(bfs_dist(g2, (1, 1)))
            r = online_dfs(g2, (1, 1), far)
            rows.append(dict(cells=cells * cells, reachable=reach, found=r['found'], moves=r['moves'],
                             bound=2 * (reach - 1), revisits=r['revisits']))
    return rows

def size_density_experiment(seeds=range(5)):
    rows = []
    for cells in (10, 20, 40):
        for p in (0.0, 0.5, 1.0):
            crs, de, mv = [], 0, 0
            for sd in seeds:
                g = facility(cells, sd, p)
                d = bfs_dist(g, (1, 1))
                far = max(d, key=d.get)
                r = online_dfs(g, (1, 1), far)
                crs.append(r['moves'] / d[far]); de += dead_ends(g); mv += r['moves']
            rows.append(dict(cells=cells * cells, opened=p, dead_ends=de / len(seeds), moves=mv / len(seeds),
                             cr_mean=sum(crs) / len(crs), cr_max=max(crs)))
    return rows

def lrta_trials(seeds=range(5), cells=20, trials=10):
    per = [[] for _ in range(trials)]
    for sd in seeds:
        g = facility(cells, sd, 0.3)
        d = bfs_dist(g, (1, 1)); far = max(d, key=d.get)
        H = {}
        for k in range(trials):
            per[k].append(lrta_star(g, (1, 1), far, H)['moves'] / d[far])
    return [sum(x) / len(x) for x in per]
