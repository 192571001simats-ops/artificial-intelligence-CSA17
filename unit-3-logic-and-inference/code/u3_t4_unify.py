# Task 4 - UNIFY, lifted Modus Ponens and propositionalisation
import itertools, time
from u3_fol import *

x, y, z, w = '?x', '?y', '?z', '?w'

PAIRS = [
    ('Knows(John, x) vs Knows(y, Mother(y))', ('Knows', 'John', x), ('Knows', y, ('Mother', y))),
    ('P(x, f(y), z) vs P(g(z), f(A), w)', ('P', x, ('f', y), z), ('P', ('g', z), ('f', 'A'), w)),
    ('Loves(x, f(x), y) vs Loves(g(z), w, z)', ('Loves', x, ('f', x), y), ('Loves', ('g', z), w, z)),
    ('Q(x, x) vs Q(y, f(y))  [occurs check]', ('Q', x, x), ('Q', y, ('f', y))),
    ('Knows(John, x) vs Knows(x, Elizabeth)  [same x]', ('Knows', 'John', x), ('Knows', x, 'Elizabeth')),
]

def unify_report(a, b, occurs_check=True):
    s = unify(a, b, {}, occurs_check)
    if s is None: return None
    try:
        return {v: show_t(subst(s, t)) for v, t in sorted(s.items())}
    except RecursionError:
        return 'cyclic binding: ' + ', '.join(f'{v}/{show_t(t)}' for v, t in s.items())

def standardize_apart(t, suffix='1'):
    m = {v: v + suffix for v in variables(t)}
    return subst(m, t)

# ---------- lifted vs propositional on the same KB ----------
RULES = [
    (('Qualified', x), [('Employee', x), ('Certified', x)]),
    (('CanOperate', x, y), [('Qualified', x), ('Assigned', x, y), ('Machine', y)]),
    (('WorksAt', x, z), [('CanOperate', x, y), ('InPlant', y, z)]),
]

def facts(k):
    C = [f'c{i}' for i in range(k)]
    F = []
    for i, c in enumerate(C):
        if i % 2 == 0: F.append(('Employee', c))
        if i % 3 == 0: F.append(('Certified', c))
        if i % 4 == 1: F.append(('Machine', c))
        F.append(('Assigned', c, C[(i + 1) % k]))
        F.append(('InPlant', c, C[(i + 2) % k]))
    return C, F

def propositionalise(rules, constants):
    """Instantiate every universal rule with every combination of ground terms."""
    ground = []
    for head, body in rules:
        vs = sorted(variables(head) | set().union(*(variables(b) for b in body)))
        for combo in itertools.product(constants, repeat=len(vs)):
            m = dict(zip(vs, combo))
            ground.append((show_t(subst(m, head)), [show_t(subst(m, b)) for b in body]))
    return ground

def pl_fc(rules, facts, q):
    count = [len(b) for _, b in rules]
    by = {}
    for i, (_, b) in enumerate(rules):
        for p in b: by.setdefault(p, []).append(i)
    inferred, agenda, steps = set(), list(facts), 0
    while agenda:
        p = agenda.pop()
        if p == q: return True, steps
        if p in inferred: continue
        inferred.add(p)
        for i in by.get(p, []):
            count[i] -= 1; steps += 1
            if count[i] == 0: agenda.append(rules[i][0])
    return False, steps

def lifted_vs_prop(k, query=None):
    C, F = facts(k)
    query = query or ('WorksAt', 'c0', f'c{3 % k}')
    t0 = time.perf_counter()
    g = propositionalise(RULES, C)
    ok_p, steps_p = pl_fc(g, [show_t(f) for f in F], show_t(query))
    tp = time.perf_counter() - t0
    st = Stats(); t0 = time.perf_counter()
    ok_l = next(bc_ask(KB(RULES, F), query, st), None) is not None
    tl = time.perf_counter() - t0
    return dict(k=k, ground_clauses=len(g), prop_steps=steps_p, prop_ms=tp * 1000,
                lifted_unify=st.unify, lifted_rule_uses=st.inferences, lifted_ms=tl * 1000, same=ok_p == ok_l == True)

def arity_growth(k=5, max_arity=6):
    out = []
    for a in range(1, max_arity + 1):
        vs = [f'?v{i}' for i in range(a)]
        rule = (('Q',) + tuple(vs), [('P',) + tuple(vs)])
        out.append(dict(arity=a, k=k, ground_clauses=len(propositionalise([rule], [f'c{i}' for i in range(k)]))))
    return out

def ground_terms(k, depth, functions):
    """All ground terms up to nesting depth d built from k constants and the given
    function symbols {name: arity}."""
    terms = {f'c{i}' for i in range(k)}
    sizes = [len(terms)]
    for _ in range(depth):
        new = set(terms)
        for f, ar in functions.items():
            for args in itertools.product(sorted(terms, key=str), repeat=ar):
                new.add((f,) + args)
        terms = new
        sizes.append(len(terms))
    return sizes

def closure(rules, facts):
    """Forward chaining to a fixed point on ground rules (all consequences)."""
    count = [len(b) for _, b in rules]
    by = {}
    for i, (_, b) in enumerate(rules):
        for p in b: by.setdefault(p, []).append(i)
    inferred, agenda = set(), list(facts)
    while agenda:
        p = agenda.pop()
        if p in inferred: continue
        inferred.add(p)
        for i in by.get(p, []):
            count[i] -= 1
            if count[i] == 0: agenda.append(rules[i][0])
    return inferred

def many_queries(k):
    """Answer every WorksAt(a, b) query (k*k of them) with each approach."""
    C, F = facts(k)
    t0 = time.perf_counter()
    known = closure(propositionalise(RULES, C), [show_t(f) for f in F])
    ans_p = sum(f'WorksAt({a}, {b})' in known for a in C for b in C)
    tp = time.perf_counter() - t0
    kb = KB(RULES, F); st = Stats(); t0 = time.perf_counter()
    ans_l = sum(next(bc_ask(kb, ('WorksAt', a, b), st), None) is not None for a in C for b in C)
    tl = time.perf_counter() - t0
    return dict(k=k, queries=k * k, prop_ms=tp * 1000, lifted_ms=tl * 1000, lifted_unify=st.unify, same=ans_p == ans_l, true=ans_p)
