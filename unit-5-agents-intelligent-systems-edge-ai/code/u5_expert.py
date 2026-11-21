# Task A3 / B2 - generic forward-chaining expert-system shell + pump-fault knowledge base.
# The engine knows nothing about pumps: every domain fact lives in the rule list (data), so rules can be
# added or edited at run time through the knowledge-acquisition interface without touching engine code.
import random


class Rule:
    def __init__(self, rid, conds, concl, priority=0, note=''):
        self.id, self.conds, self.concl, self.priority, self.note = rid, list(conds), concl, priority, note

    @property
    def specificity(self):
        return len(self.conds)

    def __repr__(self):
        c = ' AND '.join(f'{a}={v}' for a, v in self.conds)
        return f'{self.id}: IF {c} THEN {self.concl[0]}={self.concl[1]}' + (f'  [priority {self.priority}]' if self.priority else '')


class KnowledgeBase:
    """Rules + knowledge-acquisition interface (add / edit / remove at run time, with validation)."""
    def __init__(self, rules, goal='fault', askable=()):
        self.rules, self.goal, self.askable = {r.id: r for r in rules}, goal, set(askable)

    def add(self, rule):
        if rule.id in self.rules: raise ValueError(f'{rule.id} already exists - use edit()')
        if not rule.conds: raise ValueError('a rule needs at least one condition')
        self.rules[rule.id] = rule

    def edit(self, rid, conds=None, concl=None, priority=None):
        r = self.rules[rid]
        if conds is not None: r.conds = list(conds)
        if concl is not None: r.concl = concl
        if priority is not None: r.priority = priority

    def remove(self, rid):
        del self.rules[rid]


class WorkingMemory:
    def __init__(self, facts):
        self.facts, self.time, self.source, self.clock = {}, {}, {}, 0
        for a, v in facts.items():
            if v is not None: self.assert_fact(a, v, 'given')

    def assert_fact(self, a, v, src):
        self.clock += 1
        self.facts[a], self.time[a], self.source[a] = v, self.clock, src


class InferenceEngine:
    """Recognise-act cycle: MATCH -> CONFLICT SET -> RESOLVE -> FIRE, until no rule can fire."""
    STRATEGIES = ('order', 'specificity', 'priority', 'recency')

    def __init__(self, kb, strategy='specificity'):
        assert strategy in self.STRATEGIES
        self.kb, self.strategy = kb, strategy

    def match(self, wm, fired):
        cs = []
        for r in self.kb.rules.values():
            if r.id in fired: continue                                   # refraction
            if r.concl[0] in wm.facts: continue                          # single-valued attribute already set
            if all(wm.facts.get(a) == v for a, v in r.conds): cs.append(r)
        return cs

    def resolve(self, cs, wm):
        order = list(self.kb.rules)                                      # textual order as final tie-break
        key = {'order': lambda r: (order.index(r.id),),
               'specificity': lambda r: (-r.specificity, order.index(r.id)),
               'priority': lambda r: (-r.priority, order.index(r.id)),
               'recency': lambda r: (-max(wm.time[a] for a, _ in r.conds), order.index(r.id))}[self.strategy]
        return sorted(cs, key=key)[0]

    def run(self, facts, max_cycles=100):
        wm, fired, trace, conflicts = WorkingMemory(facts), [], [], []
        for cycle in range(max_cycles):
            cs = self.match(wm, set(fired))
            if not cs: break
            if len({r.concl for r in cs if r.concl[0] == self.kb.goal}) > 1:
                conflicts.append([r.id for r in cs])
            r = self.resolve(cs, wm)
            wm.assert_fact(r.concl[0], r.concl[1], r.id)
            fired.append(r.id)
            trace.append((cycle + 1, [x.id for x in cs], r.id, r.concl))
        return dict(conclusion=wm.facts.get(self.kb.goal, 'unknown'), wm=wm, fired=fired, trace=trace, conflicts=conflicts)

    # ---------------- explanation facility
    def how(self, result, attr=None, depth=0):
        """HOW was this fact established? Recursively shows the rule chain down to given facts."""
        attr = attr or self.kb.goal
        wm = result['wm']
        if attr not in wm.facts: return '  ' * depth + f'{attr} is not established'
        src = wm.source[attr]
        if src == 'given': return '  ' * depth + f'{attr}={wm.facts[attr]} was observed (given)'
        r = self.kb.rules[src]
        lines = ['  ' * depth + f'{attr}={wm.facts[attr]} because {r.id} fired: {r}']
        for a, _ in r.conds: lines.append(self.how(result, a, depth + 1))
        return '\n'.join(lines)

    def why(self, facts, asked):
        """WHY is the shell asking for `asked`? Name the rules that need it and what they would conclude."""
        out = []
        for r in self.kb.rules.values():
            if any(a == asked for a, _ in r.conds):
                known = [(a, v) for a, v in r.conds if a != asked and facts.get(a) == v]
                missing = [(a, v) for a, v in r.conds if a != asked and facts.get(a) is None]
                if known and not [1 for a, v in r.conds if a != asked and facts.get(a) not in (None, v)]:
                    out.append(f'{r.id} would conclude {r.concl[0]}={r.concl[1]} if {asked}={dict(r.conds)[asked]}; '
                               f'already known: {", ".join(f"{a}={v}" for a, v in known)}'
                               + (f'; still needs {", ".join(a for a, _ in missing)}' if missing else ''))
        return out or [f'no rule currently needs {asked}']

    def consult(self, facts, answer):
        """Interactive consultation: ask for missing askable facts that a partly-matched rule needs.
        answer(attr) plays the operator. Returns result + transcript with WHY explanations."""
        facts, transcript = dict(facts), []
        while True:
            res = self.run(facts)
            if res['conclusion'] != 'unknown': return res, transcript
            q = None
            for r in self.kb.rules.values():
                if r.concl[0] != self.kb.goal and r.concl[0] in res['wm'].facts: continue
                known = [a for a, v in r.conds if a in res['wm'].facts]
                unknown = [a for a, v in r.conds if a not in res['wm'].facts and a in self.kb.askable]
                ok = all(res['wm'].facts.get(a) == v for a in known for _a, v in r.conds if _a == a)
                if ok and known and unknown: q = unknown[0]; break       # only pursue partly matched rules
            if q is None: return res, transcript
            transcript.append(('ASK', q, self.why(res['wm'].facts, q)))
            facts[q] = answer(q)
            transcript.append(('ANSWER', q, facts[q]))


# ====================================================================== pump-fault domain (data, not code)
ATTRS = {
    'flow': ['normal', 'low', 'none'],
    'discharge_pressure': ['normal', 'low', 'high', 'fluctuating'],
    'suction_pressure': ['normal', 'low'],
    'noise': ['none', 'grinding', 'rattling', 'whine'],
    'vibration': ['normal', 'high'],
    'motor_current': ['normal', 'high', 'low'],
    'temperature': ['normal', 'high'],
    'leak': ['none', 'shaft', 'casing'],
}
FAULTS = ['cavitation', 'bearing_failure', 'impeller_blockage', 'seal_failure', 'motor_electrical', 'dry_running']
PROFILES = {   # typical presentation used to generate the expert-labelled cases (value, probability typical)
    'cavitation':        dict(flow=('low', .8), suction_pressure=('low', .9), noise=('rattling', .8), discharge_pressure=('fluctuating', .7), vibration=('high', .5)),
    'bearing_failure':   dict(noise=('grinding', .85), vibration=('high', .9), temperature=('high', .8), motor_current=('high', .4)),
    'impeller_blockage': dict(flow=('low', .9), discharge_pressure=('low', .8), motor_current=('low', .8), vibration=('high', .4)),
    'seal_failure':      dict(leak=('shaft', .9), flow=('low', .3)),
    'motor_electrical':  dict(motor_current=('high', .8), temperature=('high', .7), noise=('whine', .6), flow=('low', .4)),
    'dry_running':       dict(flow=('none', .9), discharge_pressure=('low', .8), temperature=('high', .8), motor_current=('low', .7), noise=('whine', .3)),
}
DEFAULTS = dict(flow='normal', discharge_pressure='normal', suction_pressure='normal', noise='none',
                vibration='normal', motor_current='normal', temperature='normal', leak='none')


def make_cases(seed=11, per_fault=5, p_missing=0.08):
    """30 labelled cases. A symptom shows its typical value with the profile probability, otherwise a
    different value; untouched attributes are normal; each reading is unrecorded with p_missing."""
    rng = random.Random(seed)
    cases = []
    for f in FAULTS:
        for i in range(per_fault):
            c = dict(DEFAULTS)
            for a, (v, p) in PROFILES[f].items():
                c[a] = v if rng.random() < p else rng.choice([x for x in ATTRS[a] if x != v])
            for a in list(c):
                if rng.random() < p_missing: c[a] = None
            cases.append(dict(id=f'C{len(cases) + 1:02d}', label=f, facts=c))
    rng.shuffle(cases)
    dev, hold = [], []
    for f in FAULTS:                                  # stratified 50/50 split: rules written from dev only
        fc = [c for c in cases if c['label'] == f]
        dev += fc[:len(fc) // 2 + (1 if FAULTS.index(f) % 2 == 0 else 0)]
        hold += fc[len(fc) // 2 + (1 if FAULTS.index(f) % 2 == 0 else 0):]
    return sorted(dev, key=lambda c: c['id']), sorted(hold, key=lambda c: c['id'])


def pump_rules():
    """Rule base written from the 15 development cases only. Priorities rank safety-critical faults
    (dry running and electrical faults can destroy the pump or start a fire) above the others."""
    R = Rule
    return [
        R('R1', [('flow', 'low'), ('discharge_pressure', 'fluctuating')], ('hydraulic', 'unstable')),
        R('R2', [('suction_pressure', 'low'), ('flow', 'low')], ('hydraulic', 'starved_suction')),
        R('R3', [('hydraulic', 'unstable')], ('fault', 'cavitation'), 2),
        R('R4', [('hydraulic', 'starved_suction'), ('noise', 'rattling')], ('fault', 'cavitation'), 2),
        R('R5', [('noise', 'grinding'), ('vibration', 'high')], ('mechanical', 'bearing_distress')),
        R('R6', [('mechanical', 'bearing_distress')], ('fault', 'bearing_failure'), 3),
        R('R7', [('noise', 'grinding'), ('temperature', 'high')], ('fault', 'bearing_failure'), 3),
        R('R8', [('flow', 'low'), ('discharge_pressure', 'low'), ('motor_current', 'low')], ('fault', 'impeller_blockage'), 2),
        R('R9', [('flow', 'low'), ('motor_current', 'low'), ('suction_pressure', 'normal')], ('fault', 'impeller_blockage'), 2),
        R('R10', [('leak', 'shaft')], ('fault', 'seal_failure'), 1),
        R('R11', [('motor_current', 'high'), ('noise', 'whine')], ('electrical', 'overload')),
        R('R12', [('electrical', 'overload')], ('fault', 'motor_electrical'), 4),
        R('R13', [('noise', 'whine'), ('vibration', 'normal'), ('flow', 'normal')], ('fault', 'motor_electrical'), 4),
        R('R14', [('motor_current', 'high'), ('vibration', 'normal')], ('fault', 'motor_electrical'), 4),
        R('R15', [('flow', 'none'), ('temperature', 'high')], ('fault', 'dry_running'), 5),
        R('R16', [('flow', 'none'), ('motor_current', 'low')], ('fault', 'dry_running'), 5),
    ]


def pump_kb():
    return KnowledgeBase(pump_rules(), goal='fault', askable=list(ATTRS))


def evaluate(engine, cases):
    rows = []
    for c in cases:
        res = engine.run(c['facts'])
        rows.append(dict(id=c['id'], label=c['label'], pred=res['conclusion'], fired=len(res['fired']),
                         chain=res['fired'], conflicts=len(res['conflicts'])))
    labels = FAULTS
    acc = sum(r['label'] == r['pred'] for r in rows) / len(rows)
    per = {}
    for f in labels:
        tp = sum(r['label'] == f and r['pred'] == f for r in rows)
        fp = sum(r['label'] != f and r['pred'] == f for r in rows)
        fn = sum(r['label'] == f and r['pred'] != f for r in rows)
        per[f] = (tp / (tp + fp) if tp + fp else float('nan'), tp / (tp + fn) if tp + fn else float('nan'), tp + fn)
    return rows, acc, per
