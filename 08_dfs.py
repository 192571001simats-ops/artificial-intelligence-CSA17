graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [], "E": [], "F": []
}

def dfs(graph, start):
    visited = set()
    order = []

    def visit(node):
        visited.add(node)
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visit(neighbour)

    visit(start)
    return order

if __name__ == "__main__":
    print("DFS:", *dfs(graph, "A"))
