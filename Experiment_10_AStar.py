import heapq

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 2, 'E': 5},
    'C': {'F': 1},
    'D': {'G': 7},
    'E': {'G': 1},
    'F': {'G': 3},
    'G': {}
}

h = {'A': 7, 'B': 6, 'C': 4, 'D': 7, 'E': 1, 'F': 3, 'G': 0}

def astar(start, goal):
    queue = [(h[start], 0, start, [start])]
    best = {start: 0}

    while queue:
        _, cost, node, path = heapq.heappop(queue)
        if cost > best[node]:
            continue
        if node == goal:
            return path, cost
        for neighbor, distance in graph[node].items():
            new_cost = cost + distance
            if new_cost < best.get(neighbor, float('inf')):
                best[neighbor] = new_cost
                heapq.heappush(queue, (new_cost + h[neighbor], new_cost, neighbor, path + [neighbor]))

    return None, None

path, cost = astar('A', 'G')
if path:
    print('Shortest Path:', ' -> '.join(path))
    print('Total Cost:', cost)
else:
    print('No path found')
