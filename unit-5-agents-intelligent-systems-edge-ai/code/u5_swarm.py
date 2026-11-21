# Task A2 / B1.3-B1.4 - flood search-and-rescue coverage: single drone, centralised assignment, local-rule swarm.
import time
import numpy as np

MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]            # N, E, S, W (fixed order for deterministic ties)
NAMES = ['N', 'E', 'S', 'W']


class SearchArea:
    """Ground truth: grid size, victims, which cells have physically been searched."""
    def __init__(self, W, H, n_victims, rng):
        self.W, self.H = W, H
        cells = rng.choice(W * H, size=n_victims, replace=False)
        self.victims = {(int(c % W), int(c // W)) for c in cells}
        self.visits = np.zeros((W, H), dtype=int)      # physical visit counts (for redundancy)
        self.found = set()

    def visit(self, cell):
        self.visits[cell] += 1
        if cell in self.victims: self.found.add(cell)

    def coverage(self):
        return float((self.visits > 0).mean())


class SwarmDrone:
    """Per-agent local rule. Knows only: own position, own visit map, and what neighbours within
    comm_radius share (their positions and visit maps), when the link works."""
    def __init__(self, idx, pos, W, H, rng, repulsion=0.3, rep_radius=3, frontier_penalty=2.0, tie='random'):
        self.id, self.pos, self.alive = idx, pos, True
        self.known = np.zeros((W, H), dtype=np.int16)  # visits this drone knows about
        self.W, self.H, self.rng, self.lam, self.rr, self.tie = W, H, rng, repulsion, rep_radius, tie
        self.mu = frontier_penalty
        self.energy, self.decision_time = 0.0, 0.0
        self.target, self.idle = None, False          # committed frontier target; idle = nothing left to search
        xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
        self.xs, self.ys = xs, ys

    def decide(self, neighbours, reserved=()):
        """neighbours: list of (pos, known_map) received this step; reserved: cells that neighbours
        which already decided this step will move into (empty in the basic rule). Returns a move index."""
        t0 = time.perf_counter()
        for _, km in neighbours: np.maximum(self.known, km, out=self.known)   # gossip merge
        x, y = self.pos
        options = []
        for k, (dx, dy) in enumerate(MOVES):
            c = (x + dx, y + dy)
            if 0 <= c[0] < self.W and 0 <= c[1] < self.H:
                rep = sum(max(0, self.rr - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) for p, _ in neighbours)
                options.append((self.known[c] + self.lam * rep + (1 if c in reserved else 0), k, c))
        if self.target is not None and self.known[self.target] > 0: self.target = None   # target reached by anyone
        if all(self.known[c] > 0 for _, _, c in options):          # inside explored area: frontier mode
            unv = self.known == 0
            if not unv.any():                                       # nothing left that I know of: hover
                self.idle = True
                self.decision_time += time.perf_counter() - t0
                return None
            if self.target is None:                                 # choose and commit to a frontier cell
                d = np.abs(self.xs - x) + np.abs(self.ys - y)
                for p, _ in neighbours:                             # leave frontiers near neighbours to them
                    d = d + self.mu * np.maximum(0, self.rr - (np.abs(self.xs - p[0]) + np.abs(self.ys - p[1])))
                d = np.where(unv, d, 1e9)
                self.target = tuple(int(v) for v in np.unravel_index(int(np.argmin(d)), d.shape))
            tx, ty = self.target
            options = [(abs(c[0] - tx) + abs(c[1] - ty) + (0.5 if c in reserved else 0), k, c) for _, k, c in options]
        else:
            self.target = None
        best = min(s for s, _, _ in options)
        ties = [k for s, k, _ in options if s == best]
        k = ties[0] if self.tie == 'order' else ties[int(self.rng.integers(len(ties)))]
        self.decision_time += time.perf_counter() - t0
        return k

    def move(self, k, area):
        if k is None:                                               # hover
            self.energy += 0.5; return
        self.pos = (self.pos[0] + MOVES[k][0], self.pos[1] + MOVES[k][1])
        self.energy += 1.0
        self.known[self.pos] += 1
        area.visit(self.pos)


def sweep_order(W, H):
    """Boustrophedon (lawn-mower) ordering of all cells."""
    out = []
    for x in range(W):
        col = [(x, y) for y in range(H)]
        out += col if x % 2 == 0 else col[::-1]
    return out


def step_towards(pos, target):
    x, y = pos
    if x != target[0]: return (x + (1 if target[0] > x else -1), y)
    if y != target[1]: return (x, y + (1 if target[1] > y else -1))
    return pos


def run(mode, N, seed, W=30, H=30, T=150, n_victims=10, fail_frac=0.0, comm_radius=5, p_loss=0.3,
        controller_up=True, curve_every=10, repulsion=0.3, reserve=False):
    """mode: 'single' (one lawn-mower drone), 'central' (station assigns contiguous sweep segments),
    'swarm' (local rule). Returns a metrics dict."""
    rng = np.random.default_rng(seed)
    area = SearchArea(W, H, n_victims, rng)
    n = 1 if mode == 'single' else N
    starts = [(i % 3, i // 3 % 3) for i in range(n)]               # launch cluster near the base corner
    fail_at = {}
    k_fail = int(np.floor(fail_frac * n + 0.5)) if n > 1 else int(rng.random() < fail_frac)   # round half up; one drone: probability
    for i in rng.choice(n, size=k_fail, replace=False): fail_at[int(i)] = int(rng.integers(1, T))
    curve, t_found, t95, latency, energy = [], None, None, [], 0.0
    if mode == 'swarm':
        drones = [SwarmDrone(i, starts[i], W, H, np.random.default_rng(seed * 1000 + i), repulsion=repulsion) for i in range(n)]
        for d in drones: d.known[d.pos] += 1; area.visit(d.pos)
    else:
        order = sweep_order(W, H)
        seg = np.array_split(np.arange(len(order)), n)
        routes = [[order[j] for j in s] for s in seg] if (mode == 'single' or controller_up) else [[] for _ in range(n)]
        if mode == 'central' and controller_up:                    # each assignment message may be lost
            routes = [r if rng.random() > p_loss else [] for r in routes]
        pos = list(starts); idx = [0] * n; alive = [True] * n
        for p in pos: area.visit(p)
    for t in range(1, T + 1):
        if mode == 'swarm':
            for d in drones:
                if d.alive and fail_at.get(d.id) == t: d.alive = False
            live = [d for d in drones if d.alive]
            snapshot = [(d.pos, d.known.copy()) for d in live]
            moves, intents = [], {}
            for d in live:                                          # all decide on the same snapshot
                nb_ids = [j for j, e in enumerate(live) if e is not d and abs(e.pos[0] - d.pos[0]) + abs(e.pos[1] - d.pos[1]) <= comm_radius
                          and rng.random() > p_loss]
                nb = [snapshot[j] for j in nb_ids]
                res = {intents[live[j].id] for j in nb_ids if live[j].id in intents} if reserve else ()
                t0 = time.perf_counter(); k = d.decide(nb, res); latency.append(time.perf_counter() - t0)
                moves.append(k)
                intents[d.id] = d.pos if k is None else (d.pos[0] + MOVES[k][0], d.pos[1] + MOVES[k][1])
            for d, k in zip(live, moves): d.move(k, area)
        else:
            for i in range(n):
                if alive[i] and fail_at.get(i) == t: alive[i] = False
                if not alive[i]: continue
                t0 = time.perf_counter()
                r = routes[i]
                if idx[i] < len(r) and pos[i] == r[idx[i]]: idx[i] += 1
                if idx[i] < len(r):
                    nxt = step_towards(pos[i], r[idx[i]])
                else:
                    nxt = pos[i]                                     # no assignment / finished: hover
                latency.append(time.perf_counter() - t0)
                if nxt != pos[i]: pos[i] = nxt; area.visit(nxt); energy += 1.0
                else: energy += 0.5
        if t_found is None and len(area.found) == n_victims: t_found = t
        cov = area.coverage()
        if t95 is None and cov >= 0.95: t95 = t
        if t % curve_every == 0: curve.append(cov)
    if mode == 'swarm': energy = sum(d.energy for d in drones)
    visited = int((area.visits > 0).sum())
    return dict(mode=mode, N=n, seed=seed, fail=fail_frac, success=t_found is not None, t_found=t_found, t95=t95,
                coverage=area.coverage(), redundancy=float(area.visits.sum() / max(1, visited)),
                victims=len(area.found), energy_total=energy, energy_per_drone=energy / n,
                latency_us=1e6 * float(np.mean(latency)) if latency else 0.0, curve=curve)


def hand_trace(steps=6, comm=True, reserve=False):
    """5 x 5 grid, three drones, deterministic ties (N, E, S, W) - reproduces the Task A2.3 trace."""
    W = H = 5
    rng = np.random.default_rng(0)
    area = SearchArea(W, H, 1, rng)
    drones = [SwarmDrone(i, p, W, H, rng, tie='order') for i, p in enumerate([(0, 0), (2, 0), (4, 0)])]
    for d in drones: d.known[d.pos] += 1; area.visit(d.pos)
    rows = []
    for t in range(1, steps + 1):
        snap = [(d.pos, d.known.copy()) for d in drones]
        ks, intents = [], set()
        for d in drones:
            nb = [(p, km) for (p, km), e in zip(snap, drones) if e is not d] if comm else []
            k = d.decide(nb, intents if (comm and reserve) else ())
            ks.append(k)
            intents.add(d.pos if k is None else (d.pos[0] + MOVES[k][0], d.pos[1] + MOVES[k][1]))
        for d, k in zip(drones, ks): d.move(k, area)
        rows.append((t, [NAMES[k] if k is not None else 'H' for k in ks], [d.pos for d in drones], int((area.visits > 0).sum()), int(area.visits.sum())))
    return rows
