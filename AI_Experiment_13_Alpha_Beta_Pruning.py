def alpha_beta(depth, index, maximizing, scores, alpha, beta):
    if depth == 0:
        return scores[index]

    if maximizing:
        best = float('-inf')
        for i in range(2):
            value = alpha_beta(depth - 1, index * 2 + i, False, scores, alpha, beta)
            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    best = float('inf')
    for i in range(2):
        value = alpha_beta(depth - 1, index * 2 + i, True, scores, alpha, beta)
        best = min(best, value)
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best

scores = [3, 5, 2, 9, 12, 5, 23, 23]
print('Optimal value:', alpha_beta(3, 0, True, scores, float('-inf'), float('inf')))
