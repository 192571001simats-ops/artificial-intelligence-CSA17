# Task 5 - First-order resolution with Skolemisation, using the Task 4 unifier
import itertools
from collections import deque
from u3_fol import *

e, d, x = '?e', '?d', '?x'

# CNF after Skolemisation (E1 is the Skolem constant for "some employee" in statement 3)
CLAUSES = {
    'C1': [('-', ('Employee', e)), ('-', ('Handled', e, d)), ('-', ('Restricted', d)), ('+', ('Authorised', e))],
    'C2': [('-', ('Document', d)), ('-', ('PatientData', d)), ('+', ('Restricted', d))],
    'C3a': [('+', ('Employee', 'E1'))],
    'C3b': [('-', ('Document', d)), ('-', ('Audited', d)), ('+', ('Handled', 'E1', d))],
    'C4a': [('+', ('Document', 'R1'))],
    'C4b': [('+', ('PatientData', 'R1'))],
    'C4c': [('+', ('Audited', 'R1'))],
}
GOAL = [('-', ('Authorised', x))]                      # negation of  ∃x Authorised(x)
MANAGER = [('-', ('Employee', e)), ('+', ('Employee', ('mgr', e)))]   # every employee has a manager who is an employee

def lit_vars(c):
    vs = set()
    for _, a in c: variables(a, vs)
    return vs

_k = itertools.count()
def rename(c):
    m = {v: f'{v}_{next(_k)}' for v in lit_vars(c)}
    return [(s, subst(m, a)) for s, a in c]

def canon(c):
    """Variant-insensitive key: rename variables in order of appearance."""
    m, out = {}, []
    def r(t):
        if is_var(t):
            m.setdefault(t, f'?v{len(m)}'); return m[t]
        if isinstance(t, tuple): return (t[0],) + tuple(r(a) for a in t[1:])
        return t
    for s, a in sorted(c, key=lambda l: (l[0], show_t(l[1]))):
        out.append((s, r(a)))
    return tuple(sorted(set(out), key=lambda l: (l[0], show_t(l[1]))))

def resolvents(c1, c2, st):
    c1, c2 = rename(c1), rename(c2)
    out = []
    for i, (s1, a1) in enumerate(c1):
        for j, (s2, a2) in enumerate(c2):
            if s1 != s2 and a1[0] == a2[0]:
                th = unify(a1, a2, {}, True, st)
                if th is not None:
                    st.inferences += 1
                    r = [(s, subst(th, a)) for k, (s, a) in enumerate(c1) if k != i] + \
                        [(s, subst(th, a)) for k, (s, a) in enumerate(c2) if k != j]
                    r = list(dict.fromkeys(r))                      # factoring of identical literals
                    if not any((('+' if s == '-' else '-'), a) in r for s, a in r):   # drop tautologies
                        out.append(r)
    return out

def fol_resolution(clauses, goal, strategy='sos', cap=2000):
    """Given-clause resolution. 'sos': every resolution uses at least one clause
    descended from the negated goal. 'none': all pairs of clauses are resolved."""
    st = Stats()
    base = [list(c) for c in clauses]
    usable = list(base) if strategy == 'sos' else []
    todo = deque([goal] if strategy == 'sos' else base + [goal])
    seen = {canon(c) for c in base + [goal]}
    parent = {canon(c): None for c in base + [goal]}
    generated = 0
    while todo:
        g = todo.popleft()
        for p in list(usable):
            for r in resolvents(g, p, st):
                generated += 1
                key = canon(r)
                if key not in parent: parent[key] = (canon(g), canon(p))
                if not r:
                    return True, dict(generated=generated, steps=st.inferences, unify=st.unify,
                                      proof=proof_length(parent, key)), parent
                if key not in seen and generated < cap:
                    seen.add(key); todo.append(r)
        usable.append(g)
        if generated >= cap:
            break
    return (None if generated >= cap else False), dict(generated=generated, steps=st.inferences,
                                                       unify=st.unify, proof=0), parent

def proof_length(parent, key):
    seen, stack = set(), [key]
    while stack:
        k = stack.pop()
        if k in seen: continue
        seen.add(k)
        if parent.get(k): stack.extend(parent[k])
    return len(seen)

def show_clause(c):
    if not c: return '□ (empty clause)'
    return ' ∨ '.join(('¬' if s == '-' else '') + show_t(a) for s, a in c)

def proof_lines(parent, key=(), depth=0, out=None):
    out = [] if out is None else out
    out.append('    ' * depth + show_clause(list(key)))
    if parent.get(key):
        for p in parent[key]:
            proof_lines(parent, p, depth + 1, out)
    return out

# ----- propositional version of the same small domain (constants E1, R1) for comparison -----
def propositional_version():
    from u3_logic import Imp, And, Not
    kb = []
    for E in ('E1', 'R1'):
        for D in ('E1', 'R1'):
            kb.append(Imp(And(f'Employee_{E}', f'Handled_{E}_{D}', f'Restricted_{D}'), f'Authorised_{E}'))
            kb.append(Imp(And(f'Document_{D}', f'Audited_{D}'), f'Handled_E1_{D}'))
        kb.append(Imp(And(f'Document_{E}', f'PatientData_{E}'), f'Restricted_{E}'))
    kb += ['Employee_E1', 'Document_R1', 'PatientData_R1', 'Audited_R1']
    from u3_logic import Or
    return kb, Or('Authorised_E1', 'Authorised_R1')

NOT_ENTAILED = {
    'Authorised(mgr(E1))': [('-', ('Authorised', ('mgr', 'E1')))],
    '∃x Employee(x) ∧ Audited(x)': [('-', ('Employee', x)), ('-', ('Audited', x))],
}
