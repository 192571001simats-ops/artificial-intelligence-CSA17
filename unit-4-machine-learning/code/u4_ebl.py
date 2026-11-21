# Task 5.3 - explanation-based learning of a hazard rule from one explained case
from u4_fol import unify, subst, standardize, variables, show_t, Stats, BUILTINS, is_var

s_, p_, a_, t_, v_ = '?s', '?p', '?a', '?t', '?v'
THEORY = [
    ('R1', ('Hazard', s_), [('Overloaded', s_), ('CoolingImpaired', s_)]),
    ('R2', ('Hazard', s_), [('VoltageUnstable', s_)]),
    ('R3', ('Overloaded', s_), [('LoadPct', s_, p_), ('>=', p_, 105)]),
    ('R4', ('CoolingImpaired', s_), [('Fan', s_, 'off'), ('AmbientHot', s_)]),
    ('R5', ('CoolingImpaired', s_), [('OilTemp', s_, t_), ('>=', t_, 90)]),
    ('R6', ('AmbientHot', s_), [('Ambient', s_, a_), ('>=', a_, 40)]),
    ('R7', ('VoltageUnstable', s_), [('Voltage', s_, v_), ('<', v_, 0.92)]),
    ('R8', ('VoltageUnstable', s_), [('Voltage', s_, v_), ('>=', v_, 1.08)]),
]
OPERATIONAL = {'LoadPct', 'Fan', 'Ambient', 'OilTemp', 'Voltage'} | set(BUILTINS)

def prove(goal, facts, rules, st, s=None, depth=0):
    """Backward chaining that yields (substitution, explanation tree)."""
    s = {} if s is None else s
    g = subst(s, goal)
    if g[0] in BUILTINS:
        if BUILTINS[g[0]](g[1], g[2]):
            yield s, (g, 'built-in', [])
        return
    st.goals += 1; st.max_depth = max(st.max_depth, depth)
    for f in facts:
        if f[0] == g[0] and len(f) == len(g):
            s2 = unify(g, f, s, False, st)
            if s2 is not None:
                yield s2, (f, 'fact', [])
    for name, head, body in rules:
        h, b = standardize((head, body))
        s2 = unify(g, h, s, False, st)
        if s2 is None: continue
        st.inferences += 1
        def chain(goals, s3):
            if not goals:
                yield s3, []; return
            for s4, tree in prove(goals[0], facts, rules, st, s3, depth + 1):
                for s5, rest in chain(goals[1:], s4):
                    yield s5, [tree] + rest
        for s3, kids in chain(b, s2):
            yield s3, (subst(s3, h), name, kids)

def show_tree(tree, depth=0, out=None):
    out = [] if out is None else out
    atom, how, kids = tree
    out.append('    ' * depth + f'{show_t(atom)}   [{how}]')
    for k in kids: show_tree(k, depth + 1, out)
    return out

def generalise(goal_pred, tree, rules):
    """EBG: replay the explanation with variables instead of the example's constants.
    Each rule used is re-applied to a general goal; the operational leaves under the
    general substitution form the body of the learned rule."""
    rule_by = {n: (h, b) for n, h, b in rules}
    leaves = []
    def replay(general_goal, node, s):
        atom, how, kids = node
        if how in ('fact', 'built-in'):
            leaves.append(general_goal); return s
        h, b = standardize(rule_by[how])
        s = unify(general_goal, h, s, False)
        for sub_goal, kid in zip(b, kids):
            s = replay(sub_goal, kid, s)
        return s
    head = (goal_pred, '?x')
    s = replay(head, tree, {})
    body = [subst(s, l) for l in leaves]
    # rename variables neatly
    names, k = {}, iter('xyzuvw')
    def tidy(t):
        if is_var(t):
            if t not in names: names[t] = '?' + next(k)
            return names[t]
        return (t[0],) + tuple(tidy(a) for a in t[1:]) if isinstance(t, tuple) else t
    head, body = tidy(subst(s, head)), [tidy(l) for l in body]
    assert all(l[0] in OPERATIONAL for l in body)
    return ('LEARNED', head, body)

def case(name, load, fan, ambient, oil, volt):
    return [('LoadPct', name, load), ('Fan', name, fan), ('Ambient', name, ambient), ('OilTemp', name, oil), ('Voltage', name, volt)]

TRAIN_EXAMPLE = case('S17', 112, 'off', 43, 80, 0.98)

TEST_CASES = [   # (name, load, fan, ambient, oil, volt) - designed to cover every path
    ('T1', 110, 'off', 42, 82, 1.00), ('T2', 120, 'off', 45, 88, 0.99), ('T3', 105, 'off', 40, 79, 1.01),
    ('T4', 130, 'off', 47, 95, 0.97), ('T5', 115, 'off', 41, 70, 1.02),                                   # learned path
    ('T6', 112, 'on', 43, 92, 1.00), ('T7', 108, 'on', 30, 95, 0.99),                                     # overload + hot oil (R5)
    ('T8', 80, 'on', 30, 60, 0.88), ('T9', 95, 'on', 25, 70, 1.10), ('T10', 60, 'off', 44, 65, 0.90),      # voltage (R7/R8)
    ('T11', 104, 'off', 45, 85, 1.00), ('T12', 112, 'off', 39, 85, 1.00), ('T13', 112, 'on', 43, 85, 1.00),  # boundary misses
    ('T14', 90, 'off', 46, 80, 1.00), ('T15', 70, 'on', 20, 55, 1.00), ('T16', 100, 'on', 35, 89, 0.95),
    ('T17', 140, 'on', 48, 89, 1.07), ('T18', 106, 'off', 40, 91, 0.93), ('T19', 50, 'on', 15, 45, 1.05),
    ('T20', 118, 'off', 10, 75, 1.00),
]

def evaluate_cases(learned):
    rows = []
    for name, *vals in TEST_CASES:
        facts = case(name, *vals)
        st_t, st_l = Stats(), Stats()
        theory = next(prove(('Hazard', name), facts, THEORY, st_t), None) is not None
        lr = next(prove(('Hazard', name), facts, [learned], st_l), None) is not None
        rows.append(dict(case=name, theory=theory, learned=lr, theory_goals=st_t.goals, theory_unify=st_t.unify,
                         learned_goals=st_l.goals, learned_unify=st_l.unify))
    return rows
