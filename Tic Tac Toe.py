board = [str(i) for i in range(1, 10)]

def display():
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i + 3]))
    print()

def winner(player):
    combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

    for a, b, c in combinations:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def full():
    return all(cell in ("X", "O") for cell in board)

def minimax(maximizing):
    if winner("O"):
        return 1
    if winner("X"):
        return -1
    if full():
        return 0

    best = -2 if maximizing else 2

    for i in range(9):
        if board[i] not in ("X", "O"):
            old = board[i]
            board[i] = "O" if maximizing else "X"
            score = minimax(not maximizing)
            board[i] = old

            if maximizing:
                best = max(best, score)
            else:
                best = min(best, score)

    return best

def computer_move():
    best = -2
    position = 0

    for i in range(9):
        if board[i] not in ("X", "O"):
            old = board[i]
            board[i] = "O"
            score = minimax(False)
            board[i] = old

            if score > best:
                best = score
                position = i

    return position

player = "X"
print("You are X. Computer is O.")
display()

while True:
    if player == "X":
        value = input("Your move (1-9): ").strip()

        if value not in list("123456789"):
            print("Enter a number from 1 to 9.")
            continue

        position = int(value) - 1

        if board[position] in ("X", "O"):
            print("Position already occupied.")
            continue
    else:
        position = computer_move()
        print("Computer chose:", position + 1)

    board[position] = player
    display()

    if winner(player):
        print("You win!" if player == "X" else "Computer wins!")
        break

    if full():
        print("Game is a draw!")
        break

    player = "O" if player == "X" else "X"
