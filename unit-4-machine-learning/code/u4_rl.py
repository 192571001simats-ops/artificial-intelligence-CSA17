# Task 6 - energy-aware lecture-hall air-conditioning controller: simulator, Q-learning, baseline
import math
import numpy as np

ACTIONS = ['Off', 'Low', 'High']
POWER_KW = [0.0, 1.5, 3.5]            # electrical power of each action
COOLING = [0.0, 0.9, 1.9]             # degrees C removed per 15-minute slot
SLOT_H = 0.25
START_H, END_H = 6.0, 20.0            # an episode is one day, 06:00-20:00 (56 slots)
N_SLOTS = int((END_H - START_H) / SLOT_H)
COMFORT = (22.0, 26.0)

def price_at(hour, pattern='normal'):
    """Time-of-use tariff (Rs per kWh)."""
    if pattern == 'spike' and 8.0 <= hour < 9.0:
        return 25.0                                     # adversarial: pre-arrival price spike
    if 14.0 <= hour < 18.0: return 10.0
    if 6.0 <= hour < 14.0: return 7.0
    return 4.0

def price_level(p):
    return 0 if p <= 4 else 1 if p <= 7 else 2

class HallSim:
    """Room temperature: T' = T + k_env (T_out - T) + heat_people * occupied - cooling[a] + noise."""
    def __init__(self, seed, price_pattern='normal', k_env=0.12, heat_people=0.9):
        self.rng = np.random.RandomState(seed)
        self.pattern, self.k_env, self.heat = price_pattern, k_env, heat_people

    def reset(self):
        r = self.rng
        self.t = 0
        self.peak = 33 + r.uniform(-3, 3)                 # hottest outdoor temperature today
        self.lectures = []
        for start, end in ((9.0, 12.0), (13.0, 17.0)):
            if r.rand() < 0.9:                            # 10% chance a block is cancelled
                shift = r.choice([-0.25, 0.0, 0.0, 0.25])
                self.lectures.append((start + shift, end))
        self.T = self.outdoor(START_H) + 1.0
        return self.observe()

    def hour(self, t=None):
        return START_H + SLOT_H * (self.t if t is None else t)

    def outdoor(self, h):
        return self.peak - 6 + 6 * math.sin((h - 9) / 24 * 2 * math.pi)

    def occupied(self, h):
        return any(s <= h < e for s, e in self.lectures)

    def observe(self):
        h = self.hour()
        nxt = min([s - h for s, e in self.lectures if s > h] or [99])
        return dict(slot=self.t, T=self.T, occupied=self.occupied(h), price=price_at(h, self.pattern),
                    T_out=self.outdoor(h), hour=h, next_in=nxt)

    def step(self, a, comfort_weight=10.0):
        h = self.hour()
        occ = self.occupied(h)
        kwh = POWER_KW[a] * SLOT_H
        cost = kwh * price_at(h, self.pattern)
        self.T = self.T + self.k_env * (self.outdoor(h) - self.T) + self.heat * occ - COOLING[a] + self.rng.normal(0, 0.1)
        self.t += 1
        h2 = self.hour()
        occ2 = self.occupied(h2)
        viol = max(0.0, self.T - COMFORT[1]) + max(0.0, COMFORT[0] - self.T) if occ2 else 0.0
        reward = -cost - comfort_weight * viol
        done = self.t >= N_SLOTS
        return self.observe(), reward, done, dict(kwh=kwh, cost=cost, violation=viol > 0, viol_deg=viol)

def encode(obs, features=('slot', 'T', 'occupied', 'price')):
    """Discrete state: slot (0..55), temperature bucket (1 C, 18..35), occupancy, price level."""
    key = []
    if 'slot' in features: key.append(obs['slot'])
    if 'hour' in features: key.append(int(obs['hour']))                      # 1-hour bucket
    if 'until' in features:                                                     # time to next lecture
        n = obs.get('next_in', 99)
        key.append(0 if obs['occupied'] else 1 if n <= 0.5 else 2 if n <= 1.0 else 3)
    if 'part' in features: key.append(0 if obs['hour'] < 12 else 1 if obs['hour'] < 14 else 2 if obs['hour'] < 18 else 3)
    if 'soon' in features: key.append(int(not obs['occupied'] and obs.get('next_in', 99) <= 0.5))
    if 'T' in features: key.append(int(np.clip(round(obs['T'] * 2), 36, 70)))         # 0.5 C buckets
    if 'occupied' in features: key.append(int(obs['occupied']))
    if 'price' in features: key.append(price_level(obs['price']))
    return tuple(key)

def q_learning(seed, episodes=3000, alpha=None, gamma=0.97, eps_start=1.0, eps_end=0.02,
               features=('slot', 'T', 'occupied', 'price'), price_pattern='normal'):
    rng = np.random.RandomState(seed + 1000)
    env = HallSim(seed, price_pattern)
    Q, N = {}, {}
    curve = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.7 * episodes))
        obs = env.reset(); s = encode(obs, features); total = 0.0
        done = False
        while not done:
            q = Q.setdefault(s, np.zeros(3))
            a = rng.randint(3) if rng.rand() < eps else int(np.argmax(q))
            obs2, r, done, info = env.step(a)
            s2 = encode(obs2, features)
            q2 = Q.setdefault(s2, np.zeros(3))
            target = r if done else r + gamma * np.max(q2)
            n = N.setdefault(s, np.zeros(3)); n[a] += 1
            lr = alpha if alpha is not None else max(0.02, 1.0 / n[a] ** 0.6)     # visit-count step size
            q[a] += lr * (target - q[a])
            s, total = s2, total + r
        curve.append(total)
    return Q, curve

def greedy_policy(Q, features):
    def act(obs):
        q = Q.get(encode(obs, features))
        return 0 if q is None else int(np.argmax(q))
    return act

def thermostat(obs):
    """Fixed-rule baseline: during the published timetable window 08:45-17:00, cool when warm."""
    if 8.75 <= obs['hour'] < 17.0:
        if obs['T'] > 25.0: return 2
        if obs['T'] > 24.0: return 1
    return 0

def evaluate(policy, days=30, seed=500, price_pattern='normal', log_day=None):
    env = HallSim(seed, price_pattern)
    tot = dict(reward=0.0, kwh=0.0, cost=0.0, violations=0, viol_deg=0.0, arrival_viol=0)
    trace = []
    for d in range(days):
        obs = env.reset(); done = False
        while not done:
            a = policy(obs)
            prev = obs
            obs, r, done, info = env.step(a)
            tot['reward'] += r; tot['kwh'] += info['kwh']; tot['cost'] += info['cost']
            tot['violations'] += info['violation']; tot['viol_deg'] += info['viol_deg']
            if info['violation'] and not prev['occupied'] and obs['occupied']: tot['arrival_viol'] += 1
            if log_day is not None and d == log_day:
                trace.append((round(prev['hour'], 2), round(prev['T'], 1), int(prev['occupied']), prev['price'], ACTIONS[a]))
    out = {k: v / days for k, v in tot.items()}
    return (out, trace) if log_day is not None else out

def episodes_to_converge(curve, window=100, tol=0.05):
    sm = np.convolve(curve, np.ones(window) / window, mode='valid')
    final = np.mean(curve[-window:])
    for i, v in enumerate(sm):
        if abs(v - final) <= tol * abs(final):
            return i + window
    return len(curve)

def toy_desk_check():
    """3 states (Cool, Warm, Hot) x 2 actions (Off, On); deterministic toy transitions."""
    T = {('Cool', 'Off'): ('Warm', 0), ('Cool', 'On'): ('Cool', -2), ('Warm', 'Off'): ('Hot', -1),
         ('Warm', 'On'): ('Cool', -2), ('Hot', 'Off'): ('Hot', -10), ('Hot', 'On'): ('Warm', -2)}
    Q = {(s, a): 0.0 for s in ('Cool', 'Warm', 'Hot') for a in ('Off', 'On')}
    alpha, gamma = 0.5, 0.9
    steps = [('Cool', 'Off'), ('Warm', 'Off'), ('Hot', 'Off'), ('Hot', 'On'), ('Warm', 'On'), ('Cool', 'Off')]
    log = []
    for s, a in steps:
        s2, r = T[(s, a)]
        best_next = max(Q[(s2, 'Off')], Q[(s2, 'On')])
        old = Q[(s, a)]
        Q[(s, a)] = old + alpha * (r + gamma * best_next - old)
        log.append((s, a, r, s2, old, best_next, Q[(s, a)]))
    return log, Q
