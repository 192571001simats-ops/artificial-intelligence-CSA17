# First-order core shared by Tasks 3-7: terms, UNIFY, substitution composition,
# backward chaining (with negation as failure and built-ins) and forward chaining.
import itertools, sys
sys.setrecursionlimit(20000)

# Variables are strings starting with '?'; constants are other strings or numbers;
# compound terms and atoms are tuples (functor, arg1, ..., argn).
def is_var(t): return isinstance(t, str) and t.startswith('?')

class Stats:
    def __init__(self):
        self.unify = 0; self.goals = 0; self.max_depth = 0; self.inferences = 0

def walk(t, s):
    while is_var(t) and t in s:
        t = s[t]
    return t

def subst(s, t):
    """Apply substitution s to term t (fully)."""
    t = walk(t, s)
    if isinstance(t, tuple):
        return (t[0],) + tuple(subst(s, a) for a in t[1:])
    return t

def occurs(v, t, s):
    t = walk(t, s)
    if t == v: return True
    return isinstance(t, tuple) and any(occurs(v, a, s) for a in t[1:])

def unify(x, y, s=None, occurs_check=True, st=None):
    """Most general unifier extending s, or None. Bindings are kept in triangular
    form and resolved by walk(); subst() gives the composed result."""
    if st is not None: st.unify += 1
    s = {} if s is None else s
    stack = [(x, y)]
    s = dict(s)
    while stack:
        a, b = stack.pop()
        a, b = walk(a, s), walk(b, s)
        if a == b:
            continue
        if is_var(a):
            if occurs_check and occurs(a, b, s): return None
            s[a] = b
        elif is_var(b):
            if occurs_check and occurs(b, a, s): return None
            s[b] = a
        elif isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
            stack.extend(zip(a[1:], b[1:]))
        else:
            return None
    return s

def compose(s1, s2):
    """Composition s1∘s2: apply s1 first, then s2 (as in SUBST(COMPOSE(θ1,θ2), p) =
    SUBST(θ2, SUBST(θ1, p)))."""
    out = {v: subst(s2, t) for v, t in s1.items()}
    for v, t in s2.items():
        if v not in out: out[v] = t
    return {v: t for v, t in out.items() if v != t}

def resolved(s):
    return {v: subst(s, t) for v, t in s.items()}

_counter = itertools.count()
def variables(t, out=None):
    out = set() if out is None else out
    if is_var(t): out.add(t)
    elif isinstance(t, tuple):
        for a in t[1:]: variables(a, out)
    return out

def standardize(rule):
    """Rename rule variables apart (fresh names for every use)."""
    head, body = rule
    vs = variables(head)
    for b in body: variables(b, vs)
    k = next(_counter)
    m = {v: f'{v}_{k}' for v in vs}
    return subst(m, head), [subst(m, b) for b in body]

BUILTINS = {'>=': lambda a, b: a >= b, '<': lambda a, b: a < b, '!=': lambda a, b: a != b}

class KB:
    """Definite clauses (head, [body]) plus facts indexed by predicate and by
    (predicate, argument position, value) so that any bound argument can be used."""
    def __init__(self, rules=(), facts=()):
        self.rules, self.by_head = [], {}
        self.facts, self.index, self.by_pred = set(), {}, {}
        for r in rules: self.add_rule(r)
        for f in facts: self.tell(f)
    def add_rule(self, r):
        self.rules.append(r); self.by_head.setdefault(r[0][0], []).append(r)
    def tell(self, f):
        if f in self.facts: return False
        self.facts.add(f)
        self.by_pred.setdefault(f[0], []).append(f)
        for i, a in enumerate(f[1:], 1):
            if not isinstance(a, tuple):
                self.index.setdefault((f[0], i, a), []).append(f)
        return True
    def candidates(self, goal):
        best = self.by_pred.get(goal[0], [])
        for i, a in enumerate(goal[1:], 1):
            if not is_var(a) and not isinstance(a, tuple):
                lst = self.index.get((goal[0], i, a), [])
                if len(lst) < len(best): best = lst
        return best

def bc_ask(kb, goal, st=None, depth_limit=10_000):
    """Backward chaining (depth-first, left-to-right) yielding substitutions."""
    st = st or Stats()
    def solve(goals, s, depth):
        if not goals:
            yield s; return
        g, rest = goals[0], goals[1:]
        if g[0] == 'not':                      # negation as failure (closed world)
            for _ in solve([g[1]], s, depth + 1):
                return
            yield from solve(rest, s, depth); return
        if g[0] in BUILTINS:
            a, b = subst(s, g[1]), subst(s, g[2])
            if BUILTINS[g[0]](a, b):
                yield from solve(rest, s, depth)
            return
        st.goals += 1; st.max_depth = max(st.max_depth, depth)
        if depth > depth_limit:
            raise RecursionError('depth limit')
        g2 = subst(s, g)
        for f in kb.candidates(g2):
            s2 = unify(g2, f, s, False, st)
            if s2 is not None:
                yield from solve(rest, s2, depth)
        for r in kb.by_head.get(g[0], []):
            head, body = standardize(r)
            s2 = unify(g2, head, s, False, st)
            if s2 is not None:
                st.inferences += 1
                yield from solve(body + rest, s2, depth + 1)
    yield from solve([goal], {}, 0)

def show_t(t):
    if isinstance(t, tuple):
        return f"{t[0]}({', '.join(show_t(a) for a in t[1:])})"
    return str(t)
