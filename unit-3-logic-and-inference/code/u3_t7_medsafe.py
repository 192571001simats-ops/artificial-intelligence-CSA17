# Task 7 - Medication-safety advisor: definite-clause FOL KB, indexed backward chaining
# with proof-tree explanations. The clinical rules are illustrative test knowledge for
# the assignment, not a clinical reference.
import random, time, tracemalloc
from u3_fol import *

p, d, d2, a, c = '?p', '?d', '?d2', '?a', '?c'

# Drug catalogue: drug -> classes
CATALOGUE = {
    'amoxicillin': ['penicillin'], 'cefalexin': ['cephalosporin'], 'cotrimoxazole': ['sulfonamide', 'trimethoprim'],
    'aspirin': ['nsaid', 'salicylate', 'antiplatelet'], 'ibuprofen': ['nsaid'], 'naproxen': ['nsaid'],
    'warfarin': ['anticoagulant'], 'apixaban': ['anticoagulant'], 'sertraline': ['ssri'], 'fluoxetine': ['ssri'],
    'phenelzine': ['maoi'], 'tramadol': ['opioid', 'serotonergic_opioid'], 'morphine': ['opioid'],
    'diazepam': ['benzodiazepine'], 'sildenafil': ['pde5_inhibitor'], 'nitroglycerin': ['nitrate'],
    'lisinopril': ['ace_inhibitor'], 'spironolactone': ['k_sparing_diuretic'], 'methotrexate': ['methotrexate'],
    'simvastatin': ['statin'], 'clarithromycin': ['macrolide'], 'digoxin': ['digoxin'], 'amiodarone': ['amiodarone'],
    'lithium': ['lithium'], 'propranolol': ['nonselective_beta_blocker'], 'metformin': ['biguanide'],
    'doxycycline': ['tetracycline'], 'ciprofloxacin': ['fluoroquinolone'], 'chlorphenamine': ['first_gen_antihistamine'],
    'paracetamol': ['analgesic'], 'pseudoephedrine': ['decongestant'],
}

def R(label, head, body, text):
    return dict(label=label, rule=(head, body), text=text)

def U(reason): return ('Unsafe', p, d, reason)
def RX(): return [('Prescribed', p, d)]

def base_rules():
    rs = [
        R('H1', ('AgeGroup', p, 'child'), [('Age', p, a), ('<', a, 12)], 'a patient younger than 12 is a child'),
        R('H2', ('AgeGroup', p, 'elderly'), [('Age', p, a), ('>=', a, 65)], 'a patient aged 65 or over is elderly'),
        R('H3', ('Takes', p, d), [('CurrentMed', p, d)], 'a current medicine is being taken'),
    ]
    allergy = [('penicillin', 'penicillin'), ('penicillin', 'cephalosporin'), ('sulfonamide', 'sulfonamide'),
               ('nsaid', 'nsaid'), ('salicylate', 'salicylate')]
    for i, (alg, cls) in enumerate(allergy, 1):
        rs.append(R(f'A{i}', U(f'allergy:{alg}->{cls}'), RX() + [('InClass', d, cls), ('AllergicTo', p, alg)],
                    f'{cls} drug prescribed to a patient allergic to {alg}'))
    inter = [('nsaid', 'anticoagulant', 'bleeding risk'), ('antiplatelet', 'anticoagulant', 'bleeding risk'),
             ('anticoagulant', 'nsaid', 'bleeding risk'), ('ssri', 'maoi', 'serotonin syndrome'),
             ('maoi', 'ssri', 'serotonin syndrome'), ('serotonergic_opioid', 'ssri', 'serotonin syndrome'),
             ('pde5_inhibitor', 'nitrate', 'severe hypotension'), ('nitrate', 'pde5_inhibitor', 'severe hypotension'),
             ('k_sparing_diuretic', 'ace_inhibitor', 'hyperkalaemia'), ('trimethoprim', 'methotrexate', 'bone-marrow toxicity'),
             ('macrolide', 'statin', 'myopathy'), ('amiodarone', 'digoxin', 'digoxin toxicity'),
             ('nsaid', 'lithium', 'lithium toxicity'), ('opioid', 'benzodiazepine', 'respiratory depression'),
             ('benzodiazepine', 'opioid', 'respiratory depression')]
    for i, (c1, c2, why) in enumerate(inter, 1):
        rs.append(R(f'I{i}', U(f'interaction:{c1}+{c2}'),
                    RX() + [('InClass', d, c1), ('Takes', p, d2), ('InClass', d2, c2)],
                    f'{c1} with current {c2}: {why}'))
    cond = [('nsaid', 'chronic_kidney_disease'), ('nsaid', 'peptic_ulcer'), ('ace_inhibitor', 'pregnancy'),
            ('statin', 'pregnancy'), ('anticoagulant', 'active_bleeding'), ('nonselective_beta_blocker', 'asthma'),
            ('biguanide', 'severe_renal_impairment'), ('decongestant', 'uncontrolled_hypertension'),
            ('fluoroquinolone', 'myasthenia_gravis')]
    for i, (cls, k) in enumerate(cond, 1):
        rs.append(R(f'C{i}', U(f'condition:{cls}|{k}'), RX() + [('InClass', d, cls), ('HasCondition', p, k)],
                    f'{cls} is contraindicated in {k}'))
    age = [('salicylate', 'child'), ('tetracycline', 'child'), ('fluoroquinolone', 'child'),
           ('benzodiazepine', 'elderly'), ('first_gen_antihistamine', 'elderly')]
    for i, (cls, g) in enumerate(age, 1):
        rs.append(R(f'G{i}', U(f'age:{cls}|{g}'), RX() + [('InClass', d, cls), ('AgeGroup', p, g)],
                    f'{cls} should be avoided in a {g} patient'))
    dup = ['nsaid', 'ssri', 'anticoagulant']
    for i, cls in enumerate(dup, 1):
        rs.append(R(f'D{i}', U(f'duplicate:{cls}'),
                    RX() + [('InClass', d, cls), ('Takes', p, d2), ('!=', d, d2), ('InClass', d2, cls)],
                    f'two {cls} drugs at the same time'))
    return rs                                                         # 3 + 5 + 15 + 9 + 5 + 3 = 40

def synthetic_rules(n, seed=11):
    """Extra rules of the same shapes over synthetic drug classes, to grow the KB."""
    rng = random.Random(seed)
    rs = []
    for i in range(n):
        c1, c2 = f'cls{rng.randrange(400)}', f'cls{rng.randrange(400)}'
        kind = rng.choice(['I', 'C', 'A'])
        if kind == 'I':
            rs.append(R(f'S{i}', U(f'interaction:{c1}+{c2}'), RX() + [('InClass', d, c1), ('Takes', p, d2), ('InClass', d2, c2)], f'{c1} with {c2}'))
        elif kind == 'C':
            k = f'cond{rng.randrange(150)}'
            rs.append(R(f'S{i}', U(f'condition:{c1}|{k}'), RX() + [('InClass', d, c1), ('HasCondition', p, k)], f'{c1} in {k}'))
        else:
            rs.append(R(f'S{i}', U(f'allergy:{c2}->{c1}'), RX() + [('InClass', d, c1), ('AllergicTo', p, c2)], f'{c1} allergy {c2}'))
    return rs

def catalogue_facts(seed=5):
    rng = random.Random(seed)
    F = [('InClass', drug, cl) for drug, cls in CATALOGUE.items() for cl in cls]
    for j in range(800):                                         # synthetic drugs
        for cl in rng.sample(range(400), rng.randint(1, 2)):
            F.append(('InClass', f'drug{j}', f'cls{cl}'))
    return F

def patient_facts(n, seed=9):
    rng = random.Random(seed)
    real = list(CATALOGUE); conds = ['chronic_kidney_disease', 'peptic_ulcer', 'pregnancy', 'asthma', 'cond3', 'cond7']
    F = []
    for i in range(n):
        pid = f'P{i}'
        F.append(('Age', pid, rng.randint(1, 95)))
        for drug in rng.sample(real, rng.randint(0, 3)) + [f'drug{rng.randrange(800)}' for _ in range(rng.randint(0, 2))]:
            F.append(('CurrentMed', pid, drug))
        if rng.random() < 0.2: F.append(('AllergicTo', pid, rng.choice(['penicillin', 'sulfonamide', 'nsaid', f'cls{rng.randrange(400)}'])))
        for k in rng.sample(conds, rng.randint(0, 2)): F.append(('HasCondition', pid, k))
    return F

class Advisor:
    """Rules are indexed by the drug class that the prescribed drug must belong to
    (the first InClass(?d, class) literal), so a query only touches the rules that
    can possibly apply to the new drug."""
    def __init__(self, rules, facts, indexed=True):
        self.rules, self.indexed = rules, indexed
        self.kb = KB([r['rule'] for r in rules if r['rule'][0][0] != 'Unsafe'], facts)
        self.helpers = {r['rule'][0][0]: [] for r in rules}
        for r in rules:
            if r['rule'][0][0] != 'Unsafe': self.helpers[r['rule'][0][0]].append(r)
        self.by_class, self.unsafe = {}, [r for r in rules if r['rule'][0][0] == 'Unsafe']
        for r in self.unsafe:
            cls = next(b[2] for b in r['rule'][1] if b[0] == 'InClass' and b[1] == d)
            self.by_class.setdefault(cls, []).append(r)

    def prove(self, goals, s, st, depth=0):
        """Backward chaining that also returns the proof (facts and rules used)."""
        if not goals:
            yield s, []; return
        g = subst(s, goals[0]); rest = goals[1:]
        if g[0] in BUILTINS:
            if BUILTINS[g[0]](g[1], g[2]):
                for s2, pr in self.prove(rest, s, st, depth): yield s2, [f'{show_t(g)} (built-in)'] + pr
            return
        st.goals += 1; st.max_depth = max(st.max_depth, depth)
        for f in self.kb.candidates(g):
            s2 = unify(g, f, s, False, st)
            if s2 is not None:
                for s3, pr in self.prove(rest, s2, st, depth): yield s3, [f'{show_t(f)} (fact)'] + pr
        for r in self.helpers.get(g[0], []):
            head, body = standardize(r['rule'])
            s2 = unify(g, head, s, False, st)
            if s2 is not None:
                st.inferences += 1
                for s3, pr in self.prove(body, s2, st, depth + 1):
                    for s4, pr2 in self.prove(rest, s3, st, depth):
                        yield s4, [f"{show_t(subst(s3, head))} by {r['label']}: {r['text']}"] + pr + pr2

    def check(self, patient, drug):
        st = Stats()
        self.kb.tell(('Prescribed', patient, drug))
        if self.indexed:
            classes = [f[2] for f in self.kb.candidates(('InClass', drug, '?c'))]
            cand = [r for cl in classes for r in self.by_class.get(cl, [])]
        else:
            cand = self.unsafe
        found = []
        for r in cand:
            head, body = standardize(r['rule'])
            s0 = unify(('Unsafe', patient, drug, '?why'), head, {}, False, st)
            if s0 is None: continue
            st.inferences += 1
            for s, pr in self.prove(body, s0, st, 1):
                found.append((r, [f"Unsafe({patient}, {drug}) by {r['label']}: {r['text']}"] + pr))
                break
        self.kb.facts.discard(('Prescribed', patient, drug))
        for lst in (self.kb.by_pred.get('Prescribed', []), self.kb.index.get(('Prescribed', 1, patient), []),
                    self.kb.index.get(('Prescribed', 2, drug), [])):
            if ('Prescribed', patient, drug) in lst: lst.remove(('Prescribed', patient, drug))
        return found, st

# ---------- 30 designed test cases ----------
def test_patients():
    T = []
    def case(pid, age, meds, drug, expect, allergies=(), conds=(), note=''):
        T.append(dict(pid=pid, age=age, meds=meds, drug=drug, expect=expect, allergies=allergies, conds=conds, note=note))
    case('T01', 40, ['warfarin'], 'ibuprofen', 'interaction', note='warfarin + NSAID')
    case('T02', 70, ['warfarin'], 'aspirin', 'interaction', note='warfarin + aspirin')
    case('T03', 35, ['sertraline'], 'phenelzine', 'interaction', note='SSRI + MAOI')
    case('T04', 50, ['fluoxetine'], 'tramadol', 'interaction', note='SSRI + tramadol')
    case('T05', 62, ['nitroglycerin'], 'sildenafil', 'interaction', note='nitrate + PDE5 inhibitor')
    case('T06', 68, ['lisinopril'], 'spironolactone', 'interaction', note='ACE inhibitor + K-sparing diuretic')
    case('T07', 55, ['methotrexate'], 'cotrimoxazole', 'interaction', note='methotrexate + trimethoprim')
    case('T08', 60, ['simvastatin'], 'clarithromycin', 'interaction', note='statin + macrolide')
    case('T09', 75, ['digoxin'], 'amiodarone', 'interaction', note='digoxin + amiodarone')
    case('T10', 45, ['lithium'], 'naproxen', 'interaction', note='lithium + NSAID')
    case('T11', 58, ['morphine'], 'diazepam', 'interaction', note='opioid + benzodiazepine')
    case('T12', 30, [], 'amoxicillin', 'allergy', allergies=['penicillin'], note='penicillin allergy')
    case('T13', 30, [], 'cefalexin', 'allergy', allergies=['penicillin'], note='penicillin -> cephalosporin cross-reactivity')
    case('T14', 42, [], 'cotrimoxazole', 'allergy', allergies=['sulfonamide'], note='sulfa allergy')
    case('T15', 25, [], 'ibuprofen', 'allergy', allergies=['nsaid'], note='NSAID allergy')
    case('T16', 66, [], 'ibuprofen', 'condition', conds=['chronic_kidney_disease'], note='NSAID in CKD')
    case('T17', 29, [], 'lisinopril', 'condition', conds=['pregnancy'], note='ACE inhibitor in pregnancy')
    case('T18', 33, [], 'propranolol', 'condition', conds=['asthma'], note='non-selective beta-blocker in asthma')
    case('T19', 52, [], 'naproxen', 'condition', conds=['peptic_ulcer'], note='NSAID with peptic ulcer')
    case('T20', 7, [], 'aspirin', 'age', note='aspirin in a child')
    case('T21', 6, [], 'doxycycline', 'age', note='tetracycline in a child')
    case('T22', 80, [], 'diazepam', 'age', note='benzodiazepine in the elderly')
    case('T23', 78, [], 'chlorphenamine', 'age', note='first-generation antihistamine in the elderly')
    case('T24', 40, ['ibuprofen'], 'naproxen', 'duplicate', note='two NSAIDs')
    case('T25', 81, ['warfarin', 'sertraline'], 'ibuprofen', 'interaction', note='several hazards at once')
    case('T26', 30, [], 'paracetamol', 'safe', note='no hazard')
    case('T27', 45, ['lisinopril'], 'paracetamol', 'safe', note='unrelated current medicine')
    case('T28', 30, [], 'amoxicillin', 'safe', allergies=['sulfonamide'], note='allergy to an unrelated class')
    case('T29', 40, ['sertraline'], 'paracetamol', 'safe', note='SSRI, non-serotonergic analgesic')
    case('T30', 64, [], 'diazepam', 'safe', note='age 64: just below the elderly threshold')
    return T

def facts_for(t):
    F = [('Age', t['pid'], t['age'])] + [('CurrentMed', t['pid'], m) for m in t['meds']]
    F += [('AllergicTo', t['pid'], x) for x in t['allergies']] + [('HasCondition', t['pid'], k) for k in t['conds']]
    return F

def run_tests(adv):
    rows = []
    for t in test_patients():
        for f in facts_for(t): adv.kb.tell(f)
        found, st = adv.check(t['pid'], t['drug'])
        kinds = sorted({r['rule'][0][3].split(':')[0] for r, _ in found})
        verdict = 'safe' if not found else kinds
        ok = (t['expect'] == 'safe' and not found) or (t['expect'] != 'safe' and t['expect'] in kinds)
        rows.append(dict(case=t['pid'], note=t['note'], drug=t['drug'], expected=t['expect'],
                         verdict='SAFE' if not found else 'UNSAFE: ' + ', '.join(kinds), ok=ok,
                         goals=st.goals, unify=st.unify,
                         explanation=found[0][1] if found else [], rules_fired=len(found)))
    return rows

def build(n_rules, n_patients):
    rules = base_rules() + synthetic_rules(n_rules - 40)
    facts = catalogue_facts() + patient_facts(n_patients)
    return rules, facts

def scale_experiment(sizes=(40, 400, 2000), n_patients=50_000, n_queries=300, seed=2):
    rows = []
    rng = random.Random(seed)
    synth = [f'drug{j}' for j in range(800)]
    queries = [(f'P{rng.randrange(n_patients)}', rng.choice(list(CATALOGUE)) if i % 2 == 0 else rng.choice(synth))
               for i in range(n_queries)]                        # half real drugs, half synthetic
    for n in sizes:
        rules, facts = build(n, n_patients)
        tracemalloc.start()
        adv = Advisor(rules, facts, indexed=True)
        mem = tracemalloc.get_traced_memory()[0] / 2 ** 20
        tracemalloc.stop()
        base = Advisor(rules, facts, indexed=False)
        for f in [f for t in test_patients() for f in facts_for(t)]:
            adv.kb.tell(f); base.kb.tell(f)
        acc = sum(r['ok'] for r in run_tests(adv))
        res = {}
        for name, a in (('indexed', adv), ('baseline', base)):
            g = u = 0; expl = []; unsafe = 0
            t0 = time.perf_counter()
            for pid, drug in queries:
                found, st = a.check(pid, drug)
                g += st.goals; u += st.unify
                if found: unsafe += 1; expl.append(len(found[0][1]))
            res[name] = dict(ms=(time.perf_counter() - t0) * 1000 / n_queries, goals=g / n_queries,
                             unify=u / n_queries, unsafe=unsafe, expl=sum(expl) / max(1, len(expl)))
        rows.append(dict(rules=n, patients=n_patients, facts=len(facts), memory_mb=round(mem, 1), tests_passed=acc,
                         idx_ms=res['indexed']['ms'], idx_goals=res['indexed']['goals'], idx_unify=res['indexed']['unify'],
                         base_ms=res['baseline']['ms'], base_goals=res['baseline']['goals'], base_unify=res['baseline']['unify'],
                         same_unsafe=res['indexed']['unsafe'] == res['baseline']['unsafe'], unsafe=res['indexed']['unsafe'],
                         expl_len=res['indexed']['expl']))
    return rows
