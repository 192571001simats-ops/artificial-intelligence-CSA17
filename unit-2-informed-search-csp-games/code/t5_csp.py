# Task 5 - Constraint propagation for exam timetabling, derived from the support principle
import random, time
from collections import deque

class Timetable:
    """Variables = exams, domain = allowed slots, constraint = 'different slot'
    for every pair of exams that share at least one student."""
    def __init__(self, domains, clashes):
        self.domains = {x: set(d) for x, d in domains.items()}
        self.nb = {x: set() for x in domains}
        for a, b in clashes:
            self.nb[a].add(b); self.nb[b].add(a)

    @staticmethod
    def compatible(x_slot, y_slot):
        return x_slot != y_slot

class Count:
    def __init__(self):
        self.checks = self.nodes = self.backtracks = self.wipeouts = self.removed = 0

def revise(D, xi, xj, cnt, log=None, step=None):
    """Remove every value of xi that has no compatible value left in xj."""
    removed = False
    for v in sorted(D[xi]):
        supported = False
        for w in D[xj]:
            cnt.checks += 1
            if Timetable.compatible(v, w):
                supported = True
                break
        if not supported:
            D[xi].discard(v); removed = True; cnt.removed += 1
            if log is not None:
                log.append((step[0], xi, v))
    return removed

def propagate(csp, D, cnt, arcs=None, log=None):
    """Arc queue: after xi loses a value, every arc (xk, xi) must be re-examined,
    because xk's values may have depended on the value that just disappeared."""
    q = deque(arcs if arcs is not None else [(x, y) for x in csp.nb for y in sorted(csp.nb[x])])
    queued = set(q)
    step = [0]
    while q:
        xi, xj = q.popleft(); queued.discard((xi, xj)); step[0] += 1
        if revise(D, xi, xj, cnt, log, step):
            if not D[xi]:
                cnt.wipeouts += 1
                return False                    # empty domain: no solution below this point
            for xk in sorted(csp.nb[xi] - {xj}):
                if (xk, xi) not in queued:
                    q.append((xk, xi)); queued.add((xk, xi))
    return True

def order(csp):
    return sorted(csp.nb, key=lambda x: (-len(csp.nb[x]), x))   # same static order for both

def backtrack_plain(csp, cap=200_000):
    cnt, A, xs = Count(), {}, order(csp)
    def bt(i):
        if i == len(xs):
            return True
        if cnt.nodes >= cap:
            return False
        x = xs[i]
        for v in sorted(csp.domains[x]):
            ok = True
            for y in sorted(csp.nb[x]):
                if y in A:
                    cnt.checks += 1
                    if not Timetable.compatible(v, A[y]):
                        ok = False; break
            if ok:
                cnt.nodes += 1; A[x] = v
                if bt(i + 1):
                    return True
                del A[x]; cnt.backtracks += 1
        return False
    found = bt(0)
    return (A if found else None), cnt, cnt.nodes >= cap

def backtrack_mac(csp, cap=200_000):
    """Backtracking that re-runs propagation after every assignment."""
    cnt, xs = Count(), order(csp)
    D0 = {x: set(d) for x, d in csp.domains.items()}
    if not propagate(csp, D0, cnt):
        return None, cnt, False
    def bt(i, D):
        if i == len(xs):
            return {x: next(iter(D[x])) for x in xs}
        if cnt.nodes >= cap:
            return None
        x = xs[i]
        for v in sorted(D[x]):
            cnt.nodes += 1
            D2 = {y: set(d) for y, d in D.items()}
            D2[x] = {v}
            if propagate(csp, D2, cnt, arcs=[(y, x) for y in sorted(csp.nb[x])]):
                sol = bt(i + 1, D2)
                if sol:
                    return sol
            cnt.backtracks += 1
        return None
    sol = bt(0, D0)
    return sol, cnt, cnt.nodes >= cap

def valid(csp, A):
    return A is not None and all(A[x] in csp.domains[x] and all(A[x] != A[y] for y in csp.nb[x]) for x in csp.nb)

# ---------- designed instances ----------
SLOTS = ['Mon-AM', 'Mon-PM', 'Tue-AM', 'Tue-PM']

def instance_forced():
    """Instance 1: two exams are pre-booked (hall availability). Propagation alone
    pins every exam to a single slot - no search needed."""
    exams = ['AI', 'DBMS', 'OS', 'CN', 'Maths', 'Physics', 'Bio', 'Chem']
    dom = {e: set(range(4)) for e in exams}
    dom['AI'] = {0}; dom['DBMS'] = {1}
    dom['OS'] = {0, 1, 2}; dom['Maths'] = {1, 2, 3}; dom['Physics'] = {0, 2, 3}
    clashes = [('AI', 'OS'), ('DBMS', 'OS'), ('OS', 'CN'), ('AI', 'CN'), ('DBMS', 'CN'),
               ('CN', 'Maths'), ('OS', 'Maths'), ('Maths', 'Physics'), ('AI', 'Physics'),
               ('CN', 'Physics'), ('Physics', 'Bio'), ('Maths', 'Bio'), ('OS', 'Bio'), ('CN', 'Bio'),
               ('Bio', 'Chem'), ('Physics', 'Chem'), ('AI', 'Chem'), ('DBMS', 'Chem')]
    return Timetable(dom, clashes)

def instance_infeasible():
    """Same timetable, but the CN hall is unavailable on Tue-PM (slot 3):
    propagation detects the impossibility without assigning anything."""
    csp = instance_forced()
    csp.domains['CN'] = {0, 1, 2}
    return csp

def instance_challenge():
    """Instance 2: five exams that pairwise share students, four slots. Every value
    keeps a support, so propagation removes nothing; only search proves failure."""
    exams = ['E1', 'E2', 'E3', 'E4', 'E5']
    return Timetable({e: set(range(4)) for e in exams}, [(a, b) for i, a in enumerate(exams) for b in exams[i + 1:]])

def random_timetable(n, slots=4, avg_degree=6.0, seed=0, prebooked=0.1):
    """Solvable random instance: a hidden timetable is planted first, clashes are
    only drawn between exams in different hidden slots."""
    rng = random.Random(seed)
    hidden = {i: rng.randrange(slots) for i in range(n)}
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if hidden[a] != hidden[b]]
    m = min(len(pairs), int(avg_degree * n / 2))
    clashes = rng.sample(pairs, m)
    dom = {i: set(range(slots)) for i in range(n)}
    for i in rng.sample(range(n), max(1, int(prebooked * n))):
        dom[i] = {hidden[i]}
    return Timetable(dom, clashes)
