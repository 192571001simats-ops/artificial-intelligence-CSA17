# Task 1 - Wumpus-world KB, model checking and inference-rule derivation
import itertools
from u3_logic import *

def nb(x, y, n):
    return [(a, b) for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)) if 1 <= a <= n and 1 <= b <= n]

def wumpus_kb(n):
    """General rules of an n x n Wumpus world (no percepts yet)."""
    kb = [Not('P11'), Not('W11')]                                 # start square is safe
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            kb.append(Iff(f'B{x}{y}', Or(*[f'P{a}{b}' for a, b in nb(x, y, n)])))
            kb.append(Iff(f'S{x}{y}', Or(*[f'W{a}{b}' for a, b in nb(x, y, n)])))
    squares = [(x, y) for x in range(1, n + 1) for y in range(1, n + 1)]
    kb.append(Or(*[f'W{x}{y}' for x, y in squares]))              # at least one wumpus
    for (a, b), (c, d) in itertools.combinations(squares, 2):     # at most one wumpus
        kb.append(Or(Not(f'W{a}{b}'), Not(f'W{c}{d}')))
    return kb

def counts(n):
    q = n * n
    return dict(n=n, symbols=4 * q, sentences=2 + 2 * q + 1 + q * (q - 1) // 2, models=2 ** (4 * q))

def percepts(n):
    """Designed percept sequences (consistent with a hidden world)."""
    if n == 2:   # visited [1,1] only: no breeze, stench (pit at [2,2], wumpus at [1,2])
        return [Not('B11'), 'S11']
    # visited [1,1], [2,1], [1,2] (pit at [3,1], wumpus at [1,3])
    return [Not('B11'), Not('S11'), 'B21', Not('S21'), Not('B12'), 'S12']

QUERIES = {
    2: [('[2,1] has no pit', Not('P21')), ('[1,2] has a pit', 'P12'), ('wumpus in [2,1]', 'W21')],
    3: [('[2,2] is safe', And(Not('P22'), Not('W22'))), ('[3,1] is pit-free', Not('P31')), ('[3,2] is pit-free', Not('P32'))],
}
QUERIES[4] = QUERIES[3]

def relevant(kb, alpha):
    """Keep only the sentences connected to the query through shared symbols,
    starting from the query and never crossing the global wumpus-count sentences
    or the rules of squares whose percept is still unknown."""
    known = {s for s in kb if isinstance(s, str) or (s[0] == 'not' and isinstance(s[1], str))}
    known_syms = set().union(*(symbols(s) for s in known))
    usable = [s for s in kb if s in known or
              (s[0] == 'iff' and s[1] in known_syms)]            # rule of a visited square
    need, chosen, changed = set(symbols(alpha)), [], True
    while changed:
        changed = False
        for s in usable:
            if s not in chosen and symbols(s) & need:
                chosen.append(s); need |= symbols(s); changed = True
    return chosen

def run_model_checking(n, restrict):
    kb = wumpus_kb(n) + percepts(n)
    rows = []
    for name, q in QUERIES[n]:
        k = relevant(kb, q) if restrict else kb
        res, st = tt_entails(k, q)
        rows.append(dict(n=n, restricted=restrict, query=name, answer=res if res else 'infeasible',
                         symbols=st['symbols'], sentences=st['sentences'], models=st['models'],
                         kb_evals=st['kb_evals'] if res else 0, seconds=round(st['seconds'], 4)))
    return rows

# ---------- Part B: inference rules ----------
def derive(kb, goals, max_steps=500):
    """Forward application of sound rules until every goal is derived.
    Rules: Biconditional elimination, Contraposition (as an equivalence),
    Modus Ponens, De Morgan, And-Elimination, Unit Resolution."""
    known, why = [], {}
    def add(f, rule, parents):
        if f not in why:
            known.append(f); why[f] = (rule, parents); return True
        return False
    for s in kb: add(s, 'KB', ())
    steps, i, progress = 0, 0, False
    while any(g not in why for g in goals) and steps < max_steps:
        if i == len(known):                  # end of a pass: start again if anything new appeared
            if not progress: break
            i, progress = 0, False
        f = known[i]; i += 1
        new = []
        if not isinstance(f, str):
            op = f[0]
            if op == 'iff':
                new += [(Imp(f[1], f[2]), 'Biconditional elimination', (f,)), (Imp(f[2], f[1]), 'Biconditional elimination', (f,))]
            if op == 'imp':
                new.append((Imp(Not(f[2]) if not (isinstance(f[2], tuple) and f[2][0] == 'not') else f[2][1],
                                Not(f[1]) if not (isinstance(f[1], tuple) and f[1][0] == 'not') else f[1][1]), 'Contraposition', (f,)))
                if f[1] in why: new.append((f[2], 'Modus Ponens', (f, f[1])))
            if op == 'not' and not isinstance(f[1], str) and f[1][0] == 'or':
                new.append((And(*[Not(g) for g in f[1][1:]]), 'De Morgan', (f,)))
            if op == 'and':
                new += [(g, 'And-Elimination', (f,)) for g in f[1:]]
            if op == 'or':
                for g in f[1:]:
                    comp = g[1] if isinstance(g, tuple) and g[0] == 'not' else Not(g)
                    if comp in why:
                        rest = [h for h in f[1:] if h != g]
                        new.append((Or(*rest), 'Unit Resolution', (f, comp)))
        # Modus Ponens where f is the premise of an implication already known
        for h in list(why):
            if isinstance(h, tuple) and h[0] == 'imp' and h[1] == f:
                new.append((h[2], 'Modus Ponens', (h, f)))
        for g, rule, parents in new:
            if add(g, rule, parents):
                steps += 1; progress = True
    def proof(g, seen=None):
        seen = set() if seen is None else seen
        if g in seen: return []
        seen.add(g); rule, ps = why[g]
        out = []
        for p in ps: out += proof(p, seen)
        return out + ([(g, rule, ps)] if rule != 'KB' else [])
    return steps, {g: proof(g) for g in goals if g in why}

def soundness_table(premises, conclusion):
    """Truth table over the symbols; the rule is sound iff the conclusion is true
    in every row where all premises are true."""
    syms = sorted(set().union(*(symbols(p) for p in premises)) | symbols(conclusion))
    idx = {s: i for i, s in enumerate(syms)}
    fp = [eval('lambda v: ' + pycode(p, idx)) for p in premises]
    fc = eval('lambda v: ' + pycode(conclusion, idx))
    rows, sound = [], True
    for v in itertools.product((False, True), repeat=len(syms)):
        ps = [f(v) for f in fp]; c = fc(v)
        rows.append((v, ps, c))
        if all(ps) and not c: sound = False
    return syms, rows, sound

RULES = {
    'Modus Ponens': (['P', Imp('P', 'Q')], 'Q'),
    'And-Elimination': ([And('P', 'Q')], 'P'),
    'Unit Resolution': ([Or('L', 'M'), Not('M')], 'L'),
    'Biconditional elimination': ([Iff('P', 'Q')], Imp('P', 'Q')),
    'Contraposition': ([Imp('P', 'Q')], Imp(Not('Q'), Not('P'))),
    'De Morgan': ([Not(Or('P', 'Q'))], And(Not('P'), Not('Q'))),
}

def compare_methods(n, goal):
    kb = wumpus_kb(n) + percepts(n)
    full = tt_entails(kb, goal)[1]
    res, rst = tt_entails(relevant(kb, goal), goal)
    steps, pr = derive(kb, [goal])
    return dict(n=n, query=show(goal), full_models=full['models'], restricted_models=rst['models'],
                rule_applications=steps, proof_length=len(pr[goal]))
