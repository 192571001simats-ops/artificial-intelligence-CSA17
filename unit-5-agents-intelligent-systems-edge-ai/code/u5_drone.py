# Task A1 / B1 - layered (hybrid) parcel-delivery drone agent and its grid simulator.
# The agent (LayeredDroneAgent) only sees percepts; the world (DeliveryWorld) owns the truth.
import heapq

MOVES = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
MOVE_COST, HOVER_COST, HEADWIND_EXTRA = 1.0, 0.5, 0.5


# ------------------------------------------------------------------ simulator (the truth)
class DeliveryWorld:
    """W x H grid. no_fly is published (known to the agent); obstacles appear at given times and are
    only detectable by the drone's proximity sensor (range 1). Wind drifts the drone by one cell
    on scripted steps or with probability p_drift."""

    def __init__(self, W, H, base, goal, pads, no_fly=(), obstacles=None, wind=(0, 0),
                 drift_steps=(), p_drift=0.0, battery=30.0, seed=0, max_steps=200):
        import random
        self.W, self.H, self.base, self.goal = W, H, base, goal
        self.pads = [base] + [p for p in pads if p != base]
        self.no_fly = set(no_fly)
        self.obstacles = dict(obstacles or {})         # cell -> time it appears
        self.wind, self.drift_steps, self.p_drift = wind, set(drift_steps), p_drift
        self.rng = random.Random(seed)
        self.pos, self.battery, self.t = base, battery, 0
        self.delivered = self.crashed = self.landed = False
        self.max_steps, self.log = max_steps, []

    def blocked(self, c):
        x, y = c
        if not (0 <= x < self.W and 0 <= y < self.H): return True
        if c in self.no_fly: return True
        return c in self.obstacles and self.obstacles[c] <= self.t

    def percept(self):
        x, y = self.pos
        near = {d: self.blocked((x + dx, y + dy)) for d, (dx, dy) in MOVES.items()}
        return dict(t=self.t, pos=self.pos, battery=round(self.battery, 2), blocked=near,
                    wind=self.wind, at_goal=self.pos == self.goal, at_pad=self.pos in self.pads)

    def execute(self, action):
        """Apply one action; returns a short event string. Physics, not decisions."""
        event = ''
        if action in MOVES:
            dx, dy = MOVES[action]
            cost = MOVE_COST + (HEADWIND_EXTRA if (dx, dy) == (-self.wind[0], -self.wind[1]) and self.wind != (0, 0) else 0)
            self.battery -= cost
            nxt = (self.pos[0] + dx, self.pos[1] + dy)
            if self.blocked(nxt):
                self.crashed, event = True, 'COLLISION'
            else:
                self.pos = nxt
                drift = self.t in self.drift_steps or (self.p_drift > 0 and self.rng.random() < self.p_drift)
                if drift and self.wind != (0, 0):
                    d = (self.pos[0] + self.wind[0], self.pos[1] + self.wind[1])
                    if not self.blocked(d):
                        self.pos, event = d, 'drifted'
        elif action == 'HOVER':
            self.battery -= HOVER_COST
        elif action == 'DROP':
            if self.pos == self.goal: self.delivered, event = True, 'parcel delivered'
        elif action == 'LAND':
            self.landed = True
            event = 'landed at pad' if self.pos in self.pads else 'EMERGENCY landing'
        if self.battery <= 0 and not self.landed:
            self.crashed, event = True, 'BATTERY EXHAUSTED'
        self.t += 1
        return event

    def done(self):
        return self.crashed or self.landed or self.t >= self.max_steps


# ------------------------------------------------------------------ agent (decisions only)
def astar(start, target, W, H, blocked):
    """A* on the 4-connected grid with Manhattan heuristic; returns list of moves or None."""
    if start == target: return []
    h = lambda c: abs(c[0] - target[0]) + abs(c[1] - target[1])
    frontier, best, parent = [(h(start), 0, start)], {start: 0}, {start: None}
    order = ['N', 'E', 'S', 'W']                       # fixed expansion order = deterministic ties
    while frontier:
        f, g, c = heapq.heappop(frontier)
        if c == target:
            path = []
            while parent[c]: c, m = parent[c]; path.append(m)
            return path[::-1]
        if g > best[c]: continue
        for m in order:
            dx, dy = MOVES[m]; n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < W and 0 <= n[1] < H) or n in blocked: continue
            if g + 1 < best.get(n, 1e9):
                best[n], parent[n] = g + 1, (c, m)
                heapq.heappush(frontier, (g + 1 + h(n), g + 1, n))
    return None


class LayeredDroneAgent:
    """Two layers + arbiter. Safety layer (reactive) proposes an action only when a hazard exists;
    otherwise the deliberative layer (mission phase + A* plan) acts. Safety > mission > economy."""

    def __init__(self, W, H, goal, pads, no_fly, battery, reserve=3.0, critical_margin=1.0, use_safety=True, wind_aware=False):
        self.W, self.H, self.goal, self.pads = W, H, goal, list(pads)
        self.known_blocked = set(no_fly)                 # internal map: no-fly + discovered obstacles
        self.battery, self.reserve, self.critical = battery, reserve, critical_margin
        self.phase, self.plan, self.target = 'TO_GOAL', [], goal
        self.expected, self.pos, self.delivered = None, None, False
        self.use_safety = use_safety                    # False = ablation: deliberative layer only
        self.wind_aware, self.wind = wind_aware, (0, 0)  # True = price headwind moves at 1.5 in energy estimates

    # --- helpers
    def cost_to(self, a, b):
        """Energy estimate along the A* path. First implementation: every move costs MOVE_COST.
        wind_aware=True adds the headwind surcharge, as the design pseudocode specifies."""
        p = astar(a, b, self.W, self.H, self.known_blocked)
        if p is None: return None
        if not self.wind_aware or self.wind == (0, 0): return len(p) * MOVE_COST
        head = (-self.wind[0], -self.wind[1])
        return sum(MOVE_COST + (HEADWIND_EXTRA if MOVES[m] == head else 0) for m in p)

    def nearest_pad(self, a):
        best = None
        for p in self.pads:
            c = self.cost_to(a, p)
            if c is not None and (best is None or c < best[0]): best = (c, p)
        return best

    # --- state update
    def update(self, pc):
        self.pos, self.battery, self.wind = pc['pos'], pc['battery'], pc['wind']
        x, y = self.pos
        for d, b in pc['blocked'].items():
            n = (x + MOVES[d][0], y + MOVES[d][1])
            if b and 0 <= n[0] < self.W and 0 <= n[1] < self.H: self.known_blocked.add(n)
        notes = []
        if self.expected is not None and self.pos != self.expected:
            notes.append('drift detected'); self.plan = []
        return notes

    # --- reactive safety layer
    def safety_layer(self, pc):
        np_ = self.nearest_pad(self.pos)
        reach = np_[0] if np_ else 1e9
        if not pc['at_pad'] and self.battery < reach + self.critical:
            return 'LAND', 'S1 battery cannot reach any pad: emergency land'
        if self.plan and pc['blocked'][self.plan[0]]:
            self.plan = []
            free = [m for m in ['N', 'E', 'S', 'W'] if not pc['blocked'][m]]
            if not free: return 'HOVER', 'S2 boxed in: hover'
            t = self.target
            m = min(free, key=lambda m: abs(self.pos[0] + MOVES[m][0] - t[0]) + abs(self.pos[1] + MOVES[m][1] - t[1]))
            return m, 'S2 obstacle ahead: sidestep'
        return None, ''

    # --- deliberative layer
    def deliberative_layer(self, pc):
        why = []
        if self.phase == 'TO_GOAL':
            if pc['at_goal']:
                self.delivered, self.phase = True, 'RETURN'
                return 'DROP', 'D parcel at goal: drop'
            to_goal = self.cost_to(self.pos, self.goal)
            after = self.nearest_pad(self.goal)
            if to_goal is None:
                self.phase = 'RETURN'; why.append('goal unreachable -> RETURN')
            elif after is None or self.battery < to_goal + after[0] + self.reserve:
                self.phase = 'RETURN'; why.append(f'battery {self.battery} < {to_goal}+{after[0] if after else "?"}+{self.reserve} -> RETURN')
        if self.phase == 'RETURN':
            if pc['at_pad']:
                self.phase = 'DONE'
                return 'LAND', 'D at pad: land'
            np_ = self.nearest_pad(self.pos)
            if np_ is None: return 'HOVER', 'D no pad reachable: hover'
            target = np_[1]
        else:
            target = self.goal
        if target != self.target or not self.plan:
            self.target = target
            self.plan = astar(self.pos, target, self.W, self.H, self.known_blocked) or []
            why.append(f'replan to {target}: {"".join(self.plan) or "-"}')
        if not self.plan: return 'HOVER', 'D no path: hover'
        return self.plan[0], 'D follow plan' + (' (' + '; '.join(why) + ')' if why else '')

    # --- one sense-decide-act decision
    def decide(self, pc):
        notes = self.update(pc)
        act, why = self.safety_layer(pc) if self.use_safety else (None, '')
        layer = 'SAFETY'
        if act is None:
            act, why = self.deliberative_layer(pc); layer = 'DELIBERATIVE'
        if act in MOVES:
            if self.plan and self.plan[0] == act: self.plan.pop(0)
            self.expected = (self.pos[0] + MOVES[act][0], self.pos[1] + MOVES[act][1])
        else:
            self.expected = self.pos
        return act, layer, '; '.join(notes + [why])


def run_mission(world, agent, verbose=False):
    rows = []
    while not world.done():
        pc = world.percept()
        act, layer, why = agent.decide(pc)
        ev = world.execute(act)
        rows.append(dict(t=pc['t'], pos=pc['pos'], battery=pc['battery'], phase=agent.phase,
                         layer=layer, action=act, reason=why, event=ev))
    outcome = ('CRASH: ' + rows[-1]['event']) if world.crashed else \
              ('delivered and landed at pad' if world.delivered and world.landed and world.pos in world.pads else
               'landed at pad, parcel NOT delivered' if world.landed and world.pos in world.pads else
               'emergency landing' + (' after delivery' if world.delivered else '') if world.landed else 'timeout')
    return rows, outcome, world


def print_log(rows, outcome, world):
    for r in rows:
        print(f"t={r['t']:>2} pos={str(r['pos']):<8} batt={r['battery']:>5} {r['layer']:<12} {r['action']:<5} {r['phase']:<8} {r['reason']}" + (f"  [{r['event']}]" if r['event'] else ''))
    print(f'OUTCOME: {outcome} | final position {world.pos} | battery left {round(world.battery, 2)} | steps {world.t}')
