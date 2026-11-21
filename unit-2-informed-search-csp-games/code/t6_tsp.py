# Task 6 - Exhaustive enumeration vs 2-opt local search for the delivery tour
import itertools, math, random, time

def customers(n, seed=1, size=100):
    rng = random.Random(seed)
    depot = (50.0, 50.0)
    return [depot] + [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]

def dist_matrix(pts):
    return [[math.dist(a, b) for b in pts] for a in pts]

def tour_length(D, order):
    """order lists customers 1..n; the tour starts and ends at the depot (0)."""
    total, prev = 0.0, 0
    for c in order:
        total += D[prev][c]; prev = c
    return total + D[prev][0]

def exhaustive(D):
    """Fix the depot, enumerate every ordering of the customers, skip each tour's
    mirror image (first customer > last customer) and keep the shortest."""
    n = len(D) - 1
    best, best_len, evaluated = None, float('inf'), 0
    for perm in itertools.permutations(range(1, n + 1)):
        if n > 2 and perm[0] > perm[-1]:
            continue                                    # reverse of a tour already seen
        evaluated += 1
        L = tour_length(D, perm)
        if L < best_len:
            best, best_len = perm, L
    return best, best_len, evaluated

def predicted_tours(n):
    return math.factorial(n) // 2 if n > 2 else 1

def two_opt(D, order, rng=None):
    """Hill climbing: reverse a segment whenever that shortens the tour; stop when
    no 2-opt move improves (a local minimum)."""
    t = [0] + list(order) + [0]
    improved, moves = True, 0
    while improved:
        improved = False
        for i in range(1, len(t) - 2):
            for j in range(i + 1, len(t) - 1):
                a, b, c, d = t[i - 1], t[i], t[j], t[j + 1]
                delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                if delta < -1e-12:
                    t[i:j + 1] = reversed(t[i:j + 1]); improved = True; moves += 1
    return t[1:-1], tour_length(D, t[1:-1]), moves

def local_search(D, restarts, rng):
    n = len(D) - 1
    best, best_len = None, float('inf')
    for _ in range(restarts):
        start = list(range(1, n + 1)); rng.shuffle(start)
        o, L, _ = two_opt(D, start)
        if L < best_len:
            best, best_len = o, L
    return best, best_len

def verification_instance():
    """Depot at the middle of the bottom edge of a 40 x 30 km rectangle, four
    customers at the corners: the optimum is the perimeter, 140 km."""
    return [(20.0, 0.0), (0.0, 0.0), (0.0, 30.0), (40.0, 30.0), (40.0, 0.0)]

def all_tours(D):
    n = len(D) - 1
    return [(p, tour_length(D, p)) for p in itertools.permutations(range(1, n + 1)) if p[0] < p[-1]]

def local_search_study(sizes=(8, 9, 10), runs=20, restarts=(1, 10), seed=1):
    rows = []
    for n in sizes:
        D = dist_matrix(customers(n, seed))
        _, opt, _ = exhaustive(D)
        for R in restarts:
            ex, ts, hits = [], [], 0
            for r in range(runs):
                rng = random.Random(1000 * n + r)
                t0 = time.perf_counter()
                _, L = local_search(D, R, rng)
                ts.append((time.perf_counter() - t0) * 1000)
                e = 100 * (L - opt) / opt
                ex.append(e); hits += e < 1e-9
            rows.append(dict(n=n, restarts=R, optimum=round(opt, 2), mean_excess=sum(ex) / runs,
                             worst_excess=max(ex), optimal_runs=hits, mean_ms=sum(ts) / runs))
    return rows

def local_search_large(sizes=(20, 50, 100), restarts=10, seed=1):
    rows = []
    for n in sizes:
        D = dist_matrix(customers(n, seed)); rng = random.Random(n)
        t0 = time.perf_counter(); _, L = local_search(D, restarts, rng)
        rows.append(dict(n=n, length=round(L, 2), ms=(time.perf_counter() - t0) * 1000))
    return rows
