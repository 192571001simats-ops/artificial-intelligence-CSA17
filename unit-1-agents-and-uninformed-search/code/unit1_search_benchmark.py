import random, time, statistics, heapq, json, platform
from collections import deque
from array import array

SIZES = (1000, 10000, 100000, 1000000)
SEED = 17
RUNS = 5
PREDICTION = ("The direct lift will give BFS one hop but excess cost. "
              "DFS should store few active frames. UCS should recover the minimum cost. "
              "A side sensor should reduce worst-case docking moves.")
print("Before running:", PREDICTION, flush=True)

class Hospital:
    def __init__(self, n):
        self.n = n
        self.ward = n - 2
        self.weights = random.Random(SEED).randbytes(n)

    def edges(self, u):
        # Last vertex is an isolated, closed ward. Costs are positive time units.
        for v in (2*u + 1, 2*u + 2):
            if v <= self.ward:
                yield v, 1 + self.weights[v] % 8
        if u == 0:
            yield self.ward, 1000

def finish(parent, s, g):
    if parent[g] == -1:
        return []
    path = [g]
    while path[-1] != s:
        path.append(parent[path[-1]])
    return path[::-1]

def search(h, s, g, name):
    parent = array('i', [-1]) * h.n
    parent[s] = s
    generated, expanded, peak = 1, 0, 1
    if name == 'DFS':
        def visit(u, level):
            nonlocal generated, expanded, peak
            peak = max(peak, level)
            if u == g:
                return True
            expanded += 1
            for v, w in h.edges(u):
                if parent[v] == -1:
                    generated += 1
                    parent[v] = u
                    if visit(v, level + 1):
                        return True
            return False
        visit(s, 1)
    elif name == 'BFS':
        q = deque([s])
        while q:
            u = q.popleft()
            if u == g:
                break
            expanded += 1
            for v, w in h.edges(u):
                if parent[v] == -1:
                    parent[v] = u
                    q.append(v)
                    generated += 1
            peak = max(peak, len(q))
    else:
        assert name == 'UCS'
        dist = array('q', [-1]) * h.n
        dist[s] = 0
        q = [(0,s)]
        while q:
            cost,u = heapq.heappop(q)
            if cost != dist[u]:
                continue
            if u == g:
                break
            expanded += 1
            for v,w in h.edges(u):
                new = cost + w
                if dist[v] == -1 or new < dist[v]:
                    dist[v],parent[v] = new,u
                    heapq.heappush(q,(new,v))
                    generated += 1
            peak = max(peak,len(q))
    return finish(parent,s,g), (generated,expanded,peak)

def cost_of(h,path):
    return sum(dict(h.edges(u))[v] for u,v in zip(path,path[1:]))

def minimum(h):
    # Independent oracle: compare the unique tree path with the lift.
    u, cost = h.ward, 0
    while u:
        cost += 1 + h.weights[u] % 8
        u = (u-1)//2
    return min(cost,1000)

def experiment():
    rows, prep = [], {}
    for n in SIZES:
        builds = []
        for _ in range(RUNS):
            t = time.perf_counter()
            h = Hospital(n)
            builds.append(time.perf_counter()-t)
        prep[n] = statistics.mean(builds)*1000
        for name in ('BFS','DFS','UCS'):
            times = []
            for _ in range(RUNS):
                t = time.perf_counter()
                path, stats = search(h,0,h.ward,name)
                times.append((time.perf_counter()-t)*1000)
                assert path[0] == 0 and path[-1] == h.ward
                cost = cost_of(h,path)
                if name == 'UCS': assert cost == minimum(h)
                if name == 'BFS': assert len(path) == 2
            for goal,expected in ((n-1,[]),(0,[0])):
                assert search(h,0,goal,name)[0] == expected
            row = dict(n=n,method=name,gen=stats[0],exp=stats[1],peak=stats[2],
                       depth=len(path)-1,cost=cost,opt=minimum(h),path=path,
                       ms=statistics.mean(times),runs=times)
            rows.append(row)
            print(n,name,stats,len(path)-1,cost,round(row['ms'],3),
                  'unreachable=[]; start=goal=[0]',flush=True)
    return rows,prep

rows, prep = experiment()
