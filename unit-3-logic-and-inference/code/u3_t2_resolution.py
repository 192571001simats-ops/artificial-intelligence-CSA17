# Task 2 - Propositional resolution refutation vs truth-table model checking
import time, itertools
from collections import deque
from u3_logic import *

def resolve(c1, c2, st):
    out = []
    for l in c1:
        if neg(l) in c2:
            st['attempted'] += 1
            r = (c1 - {l}) | (c2 - {neg(l)})
            if is_tautology(r):
                st['tautologies'] += 1
                continue
            out.append(frozenset(r))
    return out

def pl_resolution(kb, alpha, cap=200_000, order='fifo'):
    """Refutation: KB ∧ ¬α is turned into clauses; resolve pairs until the empty
    clause appears (entailed) or nothing new can be produced (not entailed).
    Given-clause loop with forward and backward subsumption."""
    st = dict(input_clauses=0, generated=0, attempted=0, tautologies=0, subsumed=0, kept=0, seconds=0.0)
    t0 = time.perf_counter()
    start = []
    for s in kb + [Not(alpha)]:
        for c in to_clauses(push_not(elim(s))):
            if is_tautology(c): st['tautologies'] += 1
            elif c not in start: start.append(c)
    st['input_clauses'] = len(start)
    todo = deque(sorted(start, key=len) if order == 'short' else start)
    seen, processed, parent = set(start), [], {c: None for c in start}
    while todo:
        if st['generated'] > cap:
            st['kept'] = len(processed) + len(todo)
            st['seconds'] = time.perf_counter() - t0
            return None, st, parent
        g = todo.popleft()
        if any(p <= g for p in processed):              # forward subsumption
            st['subsumed'] += 1; continue
        before = len(processed)
        processed = [p for p in processed if not g <= p]  # backward subsumption
        st['subsumed'] += before - len(processed)
        for p in processed:
            for r in resolve(g, p, st):
                st['generated'] += 1
                if r not in parent: parent[r] = (g, p)
                if not r:
                    st['kept'] = len(processed) + len(todo)
                    st['seconds'] = time.perf_counter() - t0
                    return True, st, parent
                if r not in seen:
                    seen.add(r); todo.append(r)
        processed.append(g)
    st['kept'] = len(processed)
    st['seconds'] = time.perf_counter() - t0
    return False, st, parent

def proof_tree(parent, c=frozenset(), depth=0, out=None):
    out = [] if out is None else out
    out.append('    ' * depth + show_clause(c))
    if parent.get(c):
        for p in parent[c]:
            proof_tree(parent, p, depth + 1, out)
    return out

# ---------- Smart-building access-control policy ----------
POLICY = [
    Iff('BadgeValid', And('BadgeActive', Not('BadgeExpired'))),           # R1  biconditional
    Or('Employee', 'Visitor'),                                            # R2  disjunction
    Imp('Visitor', Not('Employee')),                                      # R3
    Iff('WorkHours', Not('Night')),                                       # R4  biconditional
    Iff('Emergency', Or('FireAlarm', 'ManualOverride')),                  # R5  biconditional + disjunction
    Imp(And('Employee', 'BadgeValid', 'WorkHours'), 'LabAccess'),         # R6
    Imp(And('Visitor', 'Escorted', 'BadgeValid', 'WorkHours'), 'LobbyAccess'),  # R7
    Imp('LabAccess', 'LobbyAccess'),                                      # R8
    Imp(And('Admin', 'BadgeValid'), 'ServerRoom'),                        # R9
    Imp('ServerRoom', 'LabAccess'),                                       # R10
    Imp('Emergency', And('ExitsUnlocked', Not('ServerRoom'))),            # R11 emergency override
    Imp(And('Night', Not('Admin')), Not('LabAccess')),                    # R12
    Imp('Admin', 'Employee'),                                             # R13
]
SCENARIO = ['BadgeActive', Not('BadgeExpired'), 'Admin', Not('FireAlarm'), Not('ManualOverride')]

def chain(k):
    """Implication chain A0, A0⇒A1, ..., A(k-1)⇒Ak; query Ak."""
    return ['A0'] + [Imp(f'A{i}', f'A{i + 1}') for i in range(k)], f'A{k}'

def pigeonhole(n):
    """n+1 pigeons in n holes: unsatisfiable, so it entails any query."""
    kb = [Or(*[f'p{i}h{j}' for j in range(n)]) for i in range(n + 1)]
    for j in range(n):
        for a, b in itertools.combinations(range(n + 1), 2):
            kb.append(Or(Not(f'p{a}h{j}'), Not(f'p{b}h{j}')))
    return kb, 'Q'

def wide_or(k):
    """Few symbols, wide disjunctions: model checking enumerates 2^k models quickly,
    resolution must combine long clauses."""
    syms = [f'X{i}' for i in range(k)]
    kb = [Or(*syms)] + [Imp(s, 'Goal') for s in syms]
    return kb, 'Goal'
