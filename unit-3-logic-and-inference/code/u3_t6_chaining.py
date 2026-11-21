# Task 6 - Forward chaining vs backward chaining on an organisation hierarchy
import random, time
from u3_fol import *

x, y, z, m, r = '?x', '?y', '?z', '?m', '?r'

RULES = [
    (('Superior', x, y), [('ReportsTo', y, x)]),                                     # R1
    (('Superior', x, z), [('ReportsTo', y, x), ('Superior', y, z)]),                 # R2 (right-recursive)
    (('Manages', x, y), [('ReportsTo', y, x)]),                                      # R3
    (('Director', x), [('Role', x, 'director')]),                                    # R4
    (('Peer', x, y), [('ReportsTo', x, m), ('ReportsTo', y, m)]),                    # R5
    (('CanApprove', x, r), [('Request', r, y), ('Amount', r, 'small'), ('Manages', x, y)]),     # R6
    (('CanApprove', x, r), [('Request', r, y), ('Superior', x, y), ('Director', x)]),          # R7
    (('Escalate', r, x), [('Request', r, y), ('Amount', r, 'large'), ('Manages', x, y)]),      # R8
    (('UnderDirector', x, z), [('Superior', z, x), ('Director', z)]),                         # R9  (not used by the query)
    (('NeedsReview', r), [('Request', r, y), ('Amount', r, 'large')]),               # R10
]
CLIMB = (('Superior', x, z), [('ReportsTo', z, y), ('Superior', x, y)])               # R2'' climbs from the employee
LEFT_RECURSIVE = (('Superior', x, z), [('Superior', x, y), ('ReportsTo', z, y)])     # R2' left-recursive

def org_facts(n_facts, branching=5, seed=7):
    """Tree of employees, about n_facts facts in total (ReportsTo, Role, Request, Amount)."""
    rng = random.Random(seed)
    F, people = [], ['e0']
    F.append(('Role', 'e0', 'director'))
    i = 1
    while len(F) < n_facts:
        boss = people[(i - 1) // branching]
        p = f'e{i}'; people.append(p)
        F.append(('ReportsTo', p, boss))
        if i <= branching: F.append(('Role', p, 'director'))
        if rng.random() < 0.15:
            rq = f'r{i}'
            F += [('Request', rq, p), ('Amount', rq, rng.choice(['small', 'large']))]
        i += 1
    return F[:n_facts]

def forward_chain(rules, facts):
    """Naive forward chaining to a fixed point: every rule is re-matched against all
    known facts on every iteration (with a predicate index for the joins)."""
    kb = KB((), facts)
    st = dict(iterations=0, firings=0, redundant=0, inferred=0)
    while True:
        st['iterations'] += 1
        new = []
        for head, body in rules:
            for s in match_body(kb, body, {}):
                st['firings'] += 1
                f = subst(s, head)
                if f in kb.facts or f in new:
                    st['redundant'] += 1
                else:
                    new.append(f)
        if not new:
            return kb, st
        for f in new:
            kb.tell(f); st['inferred'] += 1

def match_body(kb, body, s):
    if not body:
        yield s; return
    g = subst(s, body[0])
    for f in kb.candidates(g):
        s2 = unify(g, f, s, False)
        if s2 is not None:
            yield from match_body(kb, body[1:], s2)

def relevant_facts(kb, query_atom, rules):
    """Derived facts that appear in at least one proof of the query (backward slice)."""
    need, out, stack = set(), set(), [query_atom]
    while stack:
        g = stack.pop()
        if g in need: continue
        need.add(g)
        for head, body in rules:
            s = unify(head, g, {}, False)
            if s is None: continue
            for s2 in match_body(kb, [subst(s, b) for b in body], s):
                for b in body:
                    a = subst(s2, b)
                    if a not in need: stack.append(a)
    return need

def fc_experiment(n):
    F = org_facts(n)
    reqs = [f for f in F if f[0] == 'Request']
    q = ('CanApprove', 'e0', reqs[-1][1])                    # can the CEO approve the deepest request?
    t0 = time.perf_counter(); kb, st = forward_chain(RULES, F); sec = time.perf_counter() - t0
    derived = kb.facts - set(F)
    rel = relevant_facts(kb, q, RULES) & derived
    return dict(facts=n, inferred=st['inferred'], firings=st['firings'], redundant=st['redundant'],
                iterations=st['iterations'], irrelevant_pct=100 * (1 - len(rel) / len(derived)),
                answer=q in kb.facts, seconds=sec)

class BCStats(Stats):
    def __init__(self):
        super().__init__(); self.seen = {}; self.repeated = 0

def bc_counted(kb, goal, depth_limit=None, memo=False):
    """Backward chaining with substitutions. Records goals expanded, unifications,
    maximum depth and repeated subgoals. Optional depth bound or memoisation."""
    st = BCStats()
    table = {}
    def key(g):                                      # variant key: variable names do not matter
        m = {}
        def r(t):
            if is_var(t): m.setdefault(t, f'_{len(m)}'); return m[t]
            return (t[0],) + tuple(r(a) for a in t[1:]) if isinstance(t, tuple) else t
        return show_t(r(g))
    def solve(goals, s, depth):
        if not goals:
            yield s; return
        g = subst(s, goals[0])
        st.goals += 1; st.max_depth = max(st.max_depth, depth)
        k = key(g)
        st.seen[k] = st.seen.get(k, 0) + 1
        if st.seen[k] > 1: st.repeated += 1
        if depth_limit is not None and depth > depth_limit:
            return
        if memo and k in table:                      # answer from the table (complete or in progress)
            for ans in table.get(k, []):
                s2 = unify(g, ans, s, False, st)
                if s2 is not None: yield from solve(goals[1:], s2, depth)
            return
        answers = []
        def raw():
            for f in kb.candidates(g):
                s2 = unify(g, f, {}, False, st)
                if s2 is not None: yield subst(s2, g)
            for rule in kb.by_head.get(g[0], []):
                head, body = standardize(rule)
                s2 = unify(g, head, {}, False, st)
                if s2 is not None:
                    st.inferences += 1
                    for s3 in solve(body, s2, depth + 1):
                        yield subst(s3, g)
        if memo:
            table[k] = []                            # mark in progress: recursive calls see current answers
            changed = True
            while changed:                           # iterate to a fixed point (simple tabling)
                changed = False
                for a in list(raw()):
                    if a not in table[k]:
                        table[k].append(a); changed = True
            for a in table[k]:
                s2 = unify(g, a, s, False, st)
                if s2 is not None: yield from solve(goals[1:], s2, depth)
            return
        for a in raw():
            s2 = unify(g, a, s, False, st)
            if s2 is not None: yield from solve(goals[1:], s2, depth)
    return solve, st

def bc_experiment(n, climb=False):
    F = org_facts(n)
    reqs = [f for f in F if f[0] == 'Request']
    q = ('CanApprove', 'e0', reqs[-1][1])
    rules = [CLIMB if r == RULES[1] else r for r in RULES] if climb else RULES
    kb = KB(rules, F)
    solve, st = bc_counted(kb, q)
    t0 = time.perf_counter(); ok = next(solve([q], {}, 0), None) is not None; sec = time.perf_counter() - t0
    return dict(facts=n, answer=ok, goals=st.goals, unify=st.unify, rule_uses=st.inferences,
                max_depth=st.max_depth, repeated=st.repeated, seconds=sec)

def left_recursion_demo(n=100, limit=200):
    """Naive BC on the left-recursive rule: show unbounded descent, then remedies."""
    F = org_facts(n)
    target = F[-1][1] if F[-1][0] == 'ReportsTo' else 'e20'
    rules = [r for r in RULES if r[0][0] != 'Superior'] + [(('Superior', x, y), [('ReportsTo', y, x)]), LEFT_RECURSIVE]
    kb = KB(rules, F)
    q = ('Superior', 'e0', 'e99999')                          # not true: forces full search
    out = []
    solve, st = bc_counted(kb, q, depth_limit=limit)
    t0 = time.perf_counter()
    try:
        res = next(solve([q], {}, 0), None)
    except RecursionError:
        res = 'RecursionError'
    out.append(dict(method=f'naive (stopped at depth {limit})', answer=res is not None, goals=st.goals,
                    unify=st.unify, max_depth=st.max_depth, repeated=st.repeated, seconds=time.perf_counter() - t0))
    for name, kw in (('depth bound 8', dict(depth_limit=8)), ('memoisation (tabling)', dict(memo=True))):
        for qq, label in ((('Superior', 'e0', 'e30'), 'true query'), (q, 'false query')):
            solve, st = bc_counted(kb, qq, **kw)
            t0 = time.perf_counter(); res = next(solve([qq], {}, 0), None)
            out.append(dict(method=f'{name}, {label}', answer=res is not None, goals=st.goals, unify=st.unify,
                            max_depth=st.max_depth, repeated=st.repeated, seconds=time.perf_counter() - t0))
    return out
