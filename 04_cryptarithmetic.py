from itertools import permutations

def solve():
    for digits in permutations(range(10), 8):
        s, e, n, d, m, o, r, y = digits
        if s == 0 or m != 1:
            continue

        send = 1000 * s + 100 * e + 10 * n + d
        more = 1000 * m + 100 * o + 10 * r + e
        money = 10000 * m + 1000 * o + 100 * n + 10 * e + y

        if send + more == money:
            return digits, send, more, money
    return None

if __name__ == "__main__":
    answer = solve()
    if answer is None:
        print("No solution.")
    else:
        digits, send, more, money = answer
        print(dict(zip("SENDMORY", digits)))
        print("SEND =", send)
        print("MORE =", more)
        print("MONEY =", money)
        print(send, "+", more, "=", money)
