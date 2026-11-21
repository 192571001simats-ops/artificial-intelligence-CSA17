# Task 3 - Exam-eligibility advisor: knowledge engineering in FOL vs propositional encoding
import random, time, sys
from u3_fol import *

V = lambda n: '?' + n
s_, c_, p_, q_, g_, k_, l_, h_, i_ = (V(x) for x in 'scpqgklhi')

# ---- final (debugged) first-order axioms ----
AXIOMS = [
 (('Eligible', s_, c_), [('Registered', s_, c_), ('AttendanceOK', s_, c_), ('PrereqsCleared', s_, c_),
                         ('FeesCleared', s_), ('GoodStanding', s_), ('ExamReady', c_)]),            # A1
 (('AttendanceOK', s_, c_), [('Attendance', s_, c_, p_), ('>=', p_, 75)]),                           # A2
 (('AttendanceOK', s_, c_), [('Attendance', s_, c_, p_), ('>=', p_, 65), ('ApprovedLeave', s_, c_)]),# A3 condonation
 (('PrereqsCleared', s_, c_), [('Registered', s_, c_), ('not', ('MissingPrereq', s_, c_))]),         # A4
 (('MissingPrereq', s_, c_), [('Registered', s_, c_), ('Prereq', q_, c_), ('not', ('Passed', s_, q_))]),  # A5
 (('Passed', s_, q_), [('Grade', s_, q_, g_), ('PassGrade', g_)]),                                   # A6
 (('FeesCleared', s_), [('FeePaid', s_)]),                                                           # A7
 (('FeesCleared', s_), [('Scholarship', s_)]),                                                       # A8
 (('GoodStanding', s_), [('Student', s_), ('not', ('Debarred', s_))]),                               # A9
 (('Debarred', s_), [('DisciplinaryCase', s_, k_), ('Upheld', k_)]),                                 # A10
 (('ApprovedLeave', s_, c_), [('LeaveRequest', l_, s_, c_), ('ApprovedBy', l_, h_), ('HOD', h_)]),   # A11
 (('Registered', s_, c_), [('Enrolled', s_, c_), ('Offered', c_)]),                                  # A12
 (('ExamReady', c_), [('Offered', c_), ('Invigilator', i_, c_)]),                                    # A13 ∃i in the body
]

# ---- earlier versions kept for the defect log ----
def axioms_v1():
    ax = list(AXIOMS)
    ax[4] = (('MissingPrereq', s_, c_), [('Prereq', q_, c_), ('not', ('Passed', s_, q_))])   # D4: NAF with s unbound
    return ax

def axioms_v0():
    ax = [r for r in AXIOMS]
    ax[3] = (('PrereqsCleared', s_, c_), [('Prereq', q_, c_), ('Passed', s_, q_)])        # D1: one prerequisite enough
    ax.insert(4, (('PrereqsCleared', s_, c_), [('NoPrereq', c_)]))
    ax.pop(5)                                                                               # (MissingPrereq not yet defined)
    ax[8] = (('GoodStanding', s_), [('not', ('Debarred', s_)), ('Student', s_)])           # D2: NAF before s is bound
    ax = [r for r in ax if r[1] != [('Scholarship', s_)]]                                  # D3: scholarship forgotten
    return ax

def demo_facts():
    F = []
    for c in ('AI', 'DBMS', 'OS', 'CN'):
        F += [('Offered', c), ('Course', c)]
    F += [('Invigilator', 'DrRao', 'AI'), ('Invigilator', 'DrSen', 'DBMS'), ('Invigilator', 'DrIyer', 'OS')]   # CN has none
    F += [('Prereq', 'DS', 'AI'), ('Prereq', 'Maths', 'AI'), ('NoPrereq', 'DBMS'), ('NoPrereq', 'OS'), ('NoPrereq', 'CN')]
    F += [('PassGrade', g) for g in ('S', 'A', 'B', 'C', 'D', 'E')] + [('HOD', 'DrKumar')]
    people = {'Asha': dict(att={'AI': 82, 'DBMS': 90}, grades={'DS': 'A', 'Maths': 'B'}, fee='paid'),
              'Bala': dict(att={'AI': 70, 'DBMS': 68}, grades={'DS': 'B', 'Maths': 'C'}, fee='paid', leave=['AI']),
              'Chitra': dict(att={'AI': 88}, grades={'DS': 'A', 'Maths': 'F'}, fee='paid'),
              'Dev': dict(att={'AI': 95, 'OS': 91}, grades={'DS': 'S', 'Maths': 'A'}, fee='scholarship'),
              'Esha': dict(att={'DBMS': 85, 'OS': 60}, grades={}, fee='none'),
              'Farid': dict(att={'DBMS': 92, 'CN': 90}, grades={}, fee='paid', case='K7'),
              'Gopi': dict(att={'OS': 80}, grades={}, fee='paid', case='K9', upheld=False)}
    for name, d in people.items():
        F.append(('Student', name))
        for c, p in d['att'].items():
            F += [('Enrolled', name, c), ('Attendance', name, c, p)]
        for q, g in d['grades'].items():
            F.append(('Grade', name, q, g))
        if d['fee'] == 'paid': F.append(('FeePaid', name))
        if d['fee'] == 'scholarship': F.append(('Scholarship', name))
        for c in d.get('leave', []):
            F += [('LeaveRequest', f'L_{name}_{c}', name, c), ('ApprovedBy', f'L_{name}_{c}', 'DrKumar')]
        if 'case' in d:
            F.append(('DisciplinaryCase', name, d['case']))
            if d.get('upheld', True): F.append(('Upheld', d['case']))
    return F

QUERIES = [
 # (description, goal, expected) - expected True/False, or the set of bindings for the variable
 ('Asha may write AI', ('Eligible', 'Asha', 'AI'), True),
 ('Asha may write DBMS', ('Eligible', 'Asha', 'DBMS'), True),
 ('Bala may write AI (68-75% with approved leave)', ('Eligible', 'Bala', 'AI'), True),
 ('Dev may write AI (scholarship covers fees)', ('Eligible', 'Dev', 'AI'), True),
 ('Gopi may write OS (case not upheld)', ('Eligible', 'Gopi', 'OS'), True),
 ('Bala may write DBMS (68%, no leave)', ('Eligible', 'Bala', 'DBMS'), False),
 ('Chitra may write AI (failed Maths prerequisite)', ('Eligible', 'Chitra', 'AI'), False),
 ('Esha may write DBMS (fees unpaid)', ('Eligible', 'Esha', 'DBMS'), False),
 ('Farid may write DBMS (debarred)', ('Eligible', 'Farid', 'DBMS'), False),
 ('Asha may write CN (not enrolled, no invigilator)', ('Eligible', 'Asha', 'CN'), False),
 ('Who may write AI?', ('Eligible', s_, 'AI'), {'Asha', 'Bala', 'Dev'}),
 ('Which exams may Dev write?', ('Eligible', 'Dev', c_), {'AI', 'OS'}),
 ('Who is in good standing?', ('GoodStanding', s_), {'Asha', 'Bala', 'Chitra', 'Dev', 'Esha', 'Gopi'}),
 ('Which courses are exam-ready?', ('ExamReady', c_), {'AI', 'DBMS', 'OS'}),
 ('Who has a prerequisite missing for AI?', ('MissingPrereq', s_, 'AI'), {'Chitra'}),
]

def ask(kb, goal, expected):
    st = Stats()
    if isinstance(expected, set):
        var = next(iter(variables(goal)))
        got = {subst(s, var) for s in bc_ask(kb, goal, st)}
    else:
        got = next(bc_ask(kb, goal, st), None) is not None
    return got, st

def run_queries(axioms):
    kb = KB(axioms, demo_facts())
    rows = []
    for d, g, e in QUERIES:
        got, st = ask(kb, g, e)
        rows.append(dict(query=d, goal=show_t(g), expected=e, actual=got, ok=got == e, unify=st.unify, goals=st.goals))
    return rows

# ---- scaling: synthetic data, both encodings ----
def synthetic(students, courses, seed=1):
    rng = random.Random(seed)
    C = [f'C{j}' for j in range(courses)]
    F = [('PassGrade', g) for g in 'SABCDE'] + [('HOD', 'H1')]
    for j, c in enumerate(C):
        F += [('Offered', c), ('Invigilator', f'I{j}', c)]
        for q in rng.sample(C[:j], min(j, rng.randint(0, 2))):
            F.append(('Prereq', q, c))
    for i in range(students):
        s = f'S{i}'
        F.append(('Student', s))
        if rng.random() < 0.85: F.append(('FeePaid', s))
        elif rng.random() < 0.5: F.append(('Scholarship', s))
        if rng.random() < 0.05: F += [('DisciplinaryCase', s, f'K{i}'), ('Upheld', f'K{i}')]
        for c in C:
            F += [('Enrolled', s, c), ('Attendance', s, c, rng.randint(55, 100))]
            F.append(('Grade', s, c, rng.choice('SABCDEF')))
            if rng.random() < 0.1:
                F += [('LeaveRequest', f'L{i}{c}', s, c), ('ApprovedBy', f'L{i}{c}', 'H1')]
    return C, F

def propositionalise(students, courses, facts):
    """One set of ground rules per student and course (the propositional encoding).
    Arithmetic and negation are compiled into ground facts when the KB is built."""
    fs = set(facts)
    pre = {}
    for f in facts:
        if f[0] == 'Prereq': pre.setdefault(f[2], []).append(f[1])
    att = {(f[1], f[2]): f[3] for f in facts if f[0] == 'Attendance'}
    passed = {(f[1], f[2]) for f in facts if f[0] == 'Grade' and f[3] != 'F'}
    rules, base = [], set()
    for c in courses:
        rules.append((f'ExamReady_{c}', [f'Offered_{c}', f'HasInvigilator_{c}']))
        base |= {f'Offered_{c}', f'HasInvigilator_{c}'}
    for i in range(students):
        s = f'S{i}'
        rules += [(f'Fees_{s}', [f'FeePaid_{s}']), (f'Fees_{s}', [f'Scholar_{s}']), (f'Good_{s}', [f'NotDebarred_{s}'])]
        if ('FeePaid', s) in fs: base.add(f'FeePaid_{s}')
        if ('Scholarship', s) in fs: base.add(f'Scholar_{s}')
        if ('DisciplinaryCase', s, f'K{i}') not in fs: base.add(f'NotDebarred_{s}')
        for c in courses:
            rules.append((f'Eligible_{s}_{c}', [f'Registered_{s}_{c}', f'AttOK_{s}_{c}', f'Prereq_{s}_{c}',
                                                f'Fees_{s}', f'Good_{s}', f'ExamReady_{c}']))
            rules.append((f'Registered_{s}_{c}', [f'Enrolled_{s}_{c}', f'Offered_{c}']))
            rules.append((f'AttOK_{s}_{c}', [f'Att75_{s}_{c}']))
            rules.append((f'AttOK_{s}_{c}', [f'Att65_{s}_{c}', f'Leave_{s}_{c}']))
            rules.append((f'Prereq_{s}_{c}', [f'Passed_{s}_{q}' for q in pre.get(c, [])]))
            rules.append((f'Passed_{s}_{c}', [f'GradeOK_{s}_{c}']))
            base.add(f'Enrolled_{s}_{c}')
            if att[(s, c)] >= 75: base.add(f'Att75_{s}_{c}')
            if att[(s, c)] >= 65: base.add(f'Att65_{s}_{c}')
            if (s, c) in passed: base.add(f'GradeOK_{s}_{c}')
            if any(f[0] == 'LeaveRequest' and f[2] == s and f[3] == c for f in ()): pass
    for f in facts:
        if f[0] == 'LeaveRequest': base.add(f'Leave_{f[2]}_{f[3]}')
    return rules, base

def pl_fc_entails(rules, facts, q, st):
    """Forward chaining with an agenda (PL-FC-ENTAILS): linear in KB size per query."""
    count = [len(b) for _, b in rules]
    by_premise = {}
    for i, (_, b) in enumerate(rules):
        for p in b: by_premise.setdefault(p, []).append(i)
    inferred, agenda = set(), list(facts)
    for i, (h, b) in enumerate(rules):
        if not b: agenda.append(h)
    while agenda:
        p = agenda.pop()
        if p == q: return True
        if p in inferred: continue
        inferred.add(p)
        for i in by_premise.get(p, []):
            count[i] -= 1; st.inferences += 1
            if count[i] == 0: agenda.append(rules[i][0])
    return False

def scaling(sizes, n_queries=20, seed=3):
    rows = []
    for s, c in sizes:
        C, F = synthetic(s, c)
        rng = random.Random(seed)
        qs = [(f'S{rng.randrange(s)}', rng.choice(C)) for _ in range(n_queries)]
        t0 = time.perf_counter(); rules, base = propositionalise(s, C, F); build_p = time.perf_counter() - t0
        st_p = Stats(); t0 = time.perf_counter()
        ans_p = [pl_fc_entails(rules, base, f'Eligible_{a}_{b}', st_p) for a, b in qs]
        q_p = (time.perf_counter() - t0) / n_queries
        t0 = time.perf_counter(); kb = KB(AXIOMS, F); build_f = time.perf_counter() - t0
        st_f = Stats(); t0 = time.perf_counter()
        ans_f = [next(bc_ask(kb, ('Eligible', a, b), st_f), None) is not None for a, b in qs]
        q_f = (time.perf_counter() - t0) / n_queries
        rows.append(dict(students=s, courses=c, facts=len(F), prop_rules=len(rules), fol_axioms=len(AXIOMS),
                         prop_build_ms=build_p * 1000, prop_query_ms=q_p * 1000,
                         fol_build_ms=build_f * 1000, fol_query_ms=q_f * 1000,
                         prop_steps=st_p.inferences // n_queries, fol_unify=st_f.unify // n_queries,
                         agree=ans_p == ans_f))
    return rows
