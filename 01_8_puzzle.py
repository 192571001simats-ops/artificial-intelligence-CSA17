from collections import deque

start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def solve(start):
    numbers = [n for n in start if n != 0]
    inversions = sum(numbers[i] > numbers[j]
                     for i in range(8) for j in range(i + 1, 8))
    if inversions % 2:
        return None

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path

        blank = state.index(0)
        row, col = divmod(blank, 3)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < 3 and 0 <= c < 3:
                board = list(state)
                other = r * 3 + c
                board[blank], board[other] = board[other], board[blank]
                next_state = tuple(board)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
    return None

if __name__ == "__main__":
    if len(start) != 9 or set(start) != set(range(9)):
        print("Use each number from 0 to 8 exactly once.")
    else:
        path = solve(start)
        if path is None:
            print("No solution.")
        else:
            print("Moves:", len(path) - 1)
            for step, board in enumerate(path):
                print("Step", step)
                for i in range(0, 9, 3):
                    print(*board[i:i + 3])
