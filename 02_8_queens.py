def solve(queens):
    row = len(queens)
    if row == 8:
        return queens

    for col in range(8):
        safe = all(col != c and abs(col - c) != row - r
                   for r, c in enumerate(queens))
        if safe:
            answer = solve(queens + [col])
            if answer is not None:
                return answer
    return None

if __name__ == "__main__":
    answer = solve([])
    print("Queen columns:", [col + 1 for col in answer])
    for col in answer:
        print(" ".join("Q" if c == col else "." for c in range(8)))
