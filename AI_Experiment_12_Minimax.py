def minimax(depth, index, maximizing, scores):
    if depth == 0:
        return scores[index]

    if maximizing:
        return max(
            minimax(depth - 1, index * 2, False, scores),
            minimax(depth - 1, index * 2 + 1, False, scores)
        )
    return min(
        minimax(depth - 1, index * 2, True, scores),
        minimax(depth - 1, index * 2 + 1, True, scores)
    )

scores = [3, 5, 2, 9, 12, 5, 23, 23]
print('Optimal value:', minimax(3, 0, True, scores))
