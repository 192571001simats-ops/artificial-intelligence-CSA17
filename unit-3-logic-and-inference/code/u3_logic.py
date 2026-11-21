# Shared propositional logic core: formulas, evaluation, compiled model checking, CNF
import itertools, time

# formula = 'Sym' | ('not', f) | ('and', f, ...) | ('or', f, ...) | ('imp', a, b) | ('iff', a, b)
def Not(f): return ('not', f)
def And(*fs): return fs[0] if len(fs) == 1 else ('and',) + tuple(fs)
def Or(*fs): return fs[0] if len(fs) == 1 else ('or',) + tuple(fs)
def Imp(a, b): return ('imp', a, b)
def Iff(a, b): return ('iff', a, b)

def symbols(f, out=None):
    out = set() if out is None else out
    if isinstance(f, str):
        out.add(f)
    else:
        for g in f[1:]:
            symbols(g, out)
    return out

def show(f):
    if isinstance(f, str): return f
    op = f[0]
    if op == 'not': return '¬' + show(f[1])
    sep = {'and': ' ∧ ', 'or': ' ∨ ', 'imp': ' ⇒ ', 'iff': ' ⇔ '}[op]
    return '(' + sep.join(show(g) for g in f[1:]) + ')'

def pycode(f, idx):
    if isinstance(f, str): return f'v[{idx[f]}]'
    op = f[0]
    if op == 'not': return f'(not {pycode(f[1], idx)})'
    if op == 'and': return '(' + ' and '.join(pycode(g, idx) for g in f[1:]) + ')'
    if op == 'or': return '(' + ' or '.join(pycode(g, idx) for g in f[1:]) + ')'
    if op == 'imp': return f'((not {pycode(f[1], idx)}) or {pycode(f[2], idx)})'
    if op == 'iff': return f'({pycode(f[1], idx)} == {pycode(f[2], idx)})'

def tt_entails(kb, alpha, limit_symbols=26):
    """Truth-table model checking: enumerate every model of the symbols in KB and
    alpha. Returns 'entailed', 'negation entailed' or 'neither' plus counters."""
    syms = sorted(set().union(*(symbols(s) for s in kb)) | symbols(alpha))
    m = len(syms)
    st = dict(symbols=m, sentences=len(kb), models=0, kb_evals=0, kb_true=0, seconds=0.0)
    if m > limit_symbols:
        st['models'] = 2 ** m; st['infeasible'] = True
        return None, st
    idx = {s: i for i, s in enumerate(syms)}
    kbf = eval('lambda v: ' + (' and '.join(pycode(s, idx) for s in kb) or 'True'))
    af = eval('lambda v: ' + pycode(alpha, idx))
    t0 = time.perf_counter()
    pos = neg = True
    for v in itertools.product((False, True), repeat=m):
        st['models'] += 1; st['kb_evals'] += 1
        if kbf(v):
            st['kb_true'] += 1
            a = af(v)
            pos = pos and a
            neg = neg and not a
    st['seconds'] = time.perf_counter() - t0
    if st['kb_true'] == 0:
        return 'KB unsatisfiable', st
    return ('entailed' if pos else 'negation entailed' if neg else 'neither'), st

# ---------- CNF conversion ----------
def elim(f):
    """Remove ⇔ and ⇒."""
    if isinstance(f, str): return f
    op = f[0]
    if op == 'iff':
        a, b = elim(f[1]), elim(f[2])
        return ('and', ('or', ('not', a), b), ('or', ('not', b), a))
    if op == 'imp':
        return ('or', ('not', elim(f[1])), elim(f[2]))
    return (op,) + tuple(elim(g) for g in f[1:])

def push_not(f):
    """Move ¬ inwards (De Morgan, double negation)."""
    if isinstance(f, str): return f
    op = f[0]
    if op == 'not':
        g = f[1]
        if isinstance(g, str): return f
        if g[0] == 'not': return push_not(g[1])
        if g[0] == 'and': return ('or',) + tuple(push_not(('not', h)) for h in g[1:])
        if g[0] == 'or': return ('and',) + tuple(push_not(('not', h)) for h in g[1:])
    return (op,) + tuple(push_not(g) for g in f[1:])

def to_clauses(f):
    """Distribute ∨ over ∧; return a list of clauses (frozensets of literals 'P' / '~P')."""
    if isinstance(f, str): return [frozenset([f])]
    op = f[0]
    if op == 'not': return [frozenset(['~' + f[1]])]
    if op == 'and':
        out = []
        for g in f[1:]: out += to_clauses(g)
        return out
    if op == 'or':
        acc = [frozenset()]
        for g in f[1:]:
            acc = [a | c for a in acc for c in to_clauses(g)]
        return acc

def cnf(f):
    return [c for c in to_clauses(push_not(elim(f))) if not is_tautology(c)]

def neg(l): return l[1:] if l.startswith('~') else '~' + l
def is_tautology(c): return any(neg(l) in c for l in c)
def show_clause(c): return '{' + ', '.join(sorted(l.replace('~', '¬') for l in c)) + '}' if c else '{} (empty)'
