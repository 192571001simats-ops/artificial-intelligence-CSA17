# Task 1 - UCS, Greedy best-first and A* on campus grid maps
import heapq, random, time

# '#' blocked, '.' road (cost 1), 'p' pedestrian zone (cost 5)
COST = {'.': 1, 'p': 5, 'S': 1, 'G': 1}

def make_open():
    W = H = 30
    g = [['.'] * W for _ in range(H)]
    for (r0, c0, h, w) in [(5, 5, 3, 4), (5, 20, 4, 3), (14, 9, 3, 5), (20, 18, 4, 4), (23, 4, 3, 3)]:
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                g[r][c] = '#'
    return g, (1, 1), (28, 28)

def make_slow_zone():
    g, _, _ = make_open()
    for r in range(13, 17):          # pedestrian zone band, open only at the far left
        for c in range(3, 30):
            if g[r][c] == '.':
                g[r][c] = 'p'
    return g, (1, 5), (28, 24)

def make_maze(cells=14, seed=7, braid=0.25):
    rng = random.Random(seed)
    H = W = 2 * cells + 1
    g = [['#'] * W for _ in range(H)]
    stack, seen = [(0, 0)], {(0, 0)}
    g[1][1] = '.'
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
              if 0 <= r + dr < cells and 0 <= c + dc < cells and (r + dr, c + dc) not in seen]
        if not nb:
            stack.pop(); continue
        nr, nc = rng.choice(nb)
        g[2 * r + 1 + (nr - r)][2 * c + 1 + (nc - c)] = '.'
        g[2 * nr + 1][2 * nc + 1] = '.'
        seen.add((nr, nc)); stack.append((nr, nc))
    for r in range(1, H - 1):          # open a share of walls to create loops (cluttered)
        for c in range(1, W - 1):
            if g[r][c] == '#' and (r % 2) != (c % 2) and rng.random() < braid * 0.3:
                g[r][c] = '.'
    return g, (1, 1), (H - 2, W - 2)

MAPS = {'Open campus': make_open, 'Cluttered maze': make_maze, 'Slow-zone trap': make_slow_zone}

def manhattan(a, b):
    # relaxed problem: no obstacles and every cell at the cheapest cost (1)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbours(g, u):
    r, c = u
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        v = (r + dr, c + dc)
        if 0 <= v[0] < len(g) and 0 <= v[1] < len(g[0]) and g[v[0]][v[1]] != '#':
            yield v, COST[g[v[0]][v[1]]]

def best_first(g, s, t, mode):
    """mode: 'UCS' f=g, 'GBFS' f=h, 'A*' f=g+h. Ties prefer the smaller h."""
    def f(gc, v):
        h = manhattan(v, t)
        return (gc if mode == 'UCS' else h if mode == 'GBFS' else gc + h), h
    best, parent, closed = {s: 0}, {s: None}, set()
    pri, h0 = f(0, s)
    heap, tick = [(pri, h0, 0, 0, s)], 1
    generated, expanded, peak = 1, 0, 1
    while heap:
        _, _, _, gc, u = heapq.heappop(heap)
        if u in closed or gc > best[u]:
            continue
        if u == t:
            path = []
            while u is not None:
                path.append(u); u = parent[u]
            return path[::-1], best[t], dict(generated=generated, expanded=expanded, peak=peak)
        closed.add(u); expanded += 1
        for v, w in neighbours(g, u):
            ng = gc + w
            if v not in closed and ng < best.get(v, float('inf')):
                best[v], parent[v] = ng, u
                pri, h = f(ng, v)
                heapq.heappush(heap, (pri, h, tick, ng, v)); tick += 1
                generated += 1
        peak = max(peak, len(heap))
    return None, float('inf'), dict(generated=generated, expanded=expanded, peak=peak)

def timed(g, s, t, mode, reps=20, rounds=5):
    vals = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            best_first(g, s, t, mode)
        vals.append((time.perf_counter() - t0) * 1000 / reps)
    vals.sort()
    return vals[len(vals) // 2]          # median of rounds, in ms

def run_task1():
    rows = []
    for name, maker in MAPS.items():
        g, s, t = maker()
        opt = best_first(g, s, t, 'UCS')[1]
        for mode in ('UCS', 'GBFS', 'A*'):
            path, cost, st = best_first(g, s, t, mode)
            rows.append(dict(map=name, method=mode, generated=st['generated'], expanded=st['expanded'],
                             peak=st['peak'], cost=cost, steps=len(path) - 1, optimal=(cost == opt),
                             ms=round(timed(g, s, t, mode), 3)))
    return rows

def check_admissible():
    """h never exceeds the true cost-to-go (exact distances by reverse Dijkstra)."""
    worst = 0
    for name, maker in MAPS.items():
        g, s, t = maker()
        dist, heap = {t: 0}, [(0, t)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, _ in neighbours(g, u):
                w = COST[g[u[0]][u[1]]]          # cost of entering u from v
                if d + w < dist.get(v, float('inf')):
                    dist[v] = d + w; heapq.heappush(heap, (d + w, v))
        worst = max(worst, max(manhattan(u, t) - d for u, d in dist.items()))
    return worst

def open_road_trips(n=40, seed=3):
    """Typical trips: random start/goal pairs on the open campus map."""
    g, _, _ = make_open()
    free = [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] != '#']
    rng = random.Random(seed)
    tot = {m: dict(expanded=0, generated=0, cost=0, optimal=0) for m in ('UCS', 'GBFS', 'A*')}
    for _ in range(n):
        s, t = rng.sample(free, 2)
        opt = best_first(g, s, t, 'UCS')[1]
        for m in tot:
            _, c, st = best_first(g, s, t, m)
            tot[m]['expanded'] += st['expanded']; tot[m]['generated'] += st['generated']
            tot[m]['cost'] += c; tot[m]['optimal'] += (c == opt)
    return {m: {k: (v / n if k != 'optimal' else v) for k, v in d.items()} for m, d in tot.items()}
