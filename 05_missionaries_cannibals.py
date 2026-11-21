from collections import deque

def safe(m, c):
    if not (0 <= m <= 3 and 0 <= c <= 3):
        return False
    left_safe = m == 0 or m >= c
    right_safe = 3 - m == 0 or 3 - m >= 3 - c
    return left_safe and right_safe

def solve():
    start = (3, 3, 1)
    goal = (0, 0, 0)
    queue = deque([(start, [start])])
    visited = {start}
    passengers = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    while queue:
        (m, c, boat), path = queue.popleft()
        if (m, c, boat) == goal:
            return path

        for dm, dc in passengers:
            direction = -1 if boat == 1 else 1
            nm, nc = m + direction * dm, c + direction * dc
            state = (nm, nc, 1 - boat)
            if safe(nm, nc) and state not in visited:
                visited.add(state)
                queue.append((state, path + [state]))
    return None

if __name__ == "__main__":
    path = solve()
    if path is None:
        print("No solution.")
    else:
        print("Crossings:", len(path) - 1)
        for step, (m, c, boat) in enumerate(path):
            side = "Left" if boat == 1 else "Right"
            print(f"Step {step}: Left=({m}M, {c}C), "
                  f"Right=({3-m}M, {3-c}C), Boat={side}")
