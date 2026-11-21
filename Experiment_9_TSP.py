from itertools import permutations

dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

n = len(dist)
min_cost = float('inf')
best_path = []

for p in permutations(range(1, n)):
    path = [0] + list(p) + [0]
    cost = sum(dist[path[i]][path[i + 1]] for i in range(n))
    if cost < min_cost:
        min_cost = cost
        best_path = path

print('Shortest Path:', ' -> '.join(str(i + 1) for i in best_path))
print('Minimum Cost:', min_cost)
