from collections import deque

capacity_a = 4
capacity_b = 3
target = 2

def solve(a_max, b_max, target):
    queue = deque([((0, 0), [(0, 0)])])
    visited = {(0, 0)}

    while queue:
        (a, b), path = queue.popleft()
        if a == target or b == target:
            return path

        pour_ab = min(a, b_max - b)
        pour_ba = min(b, a_max - a)
        next_states = [
            (a_max, b), (a, b_max), (0, b), (a, 0),
            (a - pour_ab, b + pour_ab),
            (a + pour_ba, b - pour_ba)
        ]

        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append((state, path + [state]))
    return None

if __name__ == "__main__":
    path = solve(capacity_a, capacity_b, target)
    if path is None:
        print("No solution.")
    else:
        print("(Jug A, Jug B)")
        for state in path:
            print(state)
