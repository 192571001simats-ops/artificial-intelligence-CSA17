"""
Experiment 11: Map Coloring Problem using Constraint Satisfaction (CSP)

Aim:
    To color a map so that no two adjacent regions share the same color,
    using backtracking search for Constraint Satisfaction Problems.

CSP formulation:
    Variables   : regions of the map (Australia)
    Domains     : {Red, Green, Blue}
    Constraints : adjacent regions must have different colors
"""

# Variables and their neighbours (the constraint graph)
neighbors = {
    "WA":  ["NT", "SA"],
    "NT":  ["WA", "SA", "Q"],
    "SA":  ["WA", "NT", "Q", "NSW", "V"],
    "Q":   ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V":   ["SA", "NSW"],
    "T":   [],
}

colors = ["Red", "Green", "Blue"]


def is_consistent(region, color, assignment):
    """Check that no already-coloured neighbour has the same colour."""
    for neighbor in neighbors[region]:
        if assignment.get(neighbor) == color:
            return False
    return True


def select_unassigned_variable(assignment):
    """MRV heuristic: pick the region with the fewest legal colours left."""
    unassigned = [r for r in neighbors if r not in assignment]
    return min(
        unassigned,
        key=lambda r: sum(is_consistent(r, c, assignment) for c in colors),
    )


def backtrack(assignment):
    """Recursive backtracking search."""
    if len(assignment) == len(neighbors):
        return assignment

    region = select_unassigned_variable(assignment)
    for color in colors:
        if is_consistent(region, color, assignment):
            assignment[region] = color
            print(f"Assign {region:<3} -> {color}")
            result = backtrack(assignment)
            if result is not None:
                return result
            print(f"Backtrack: remove {region}")
            del assignment[region]
    return None


def main():
    print("Map Coloring using CSP (Backtracking + MRV)\n")
    solution = backtrack({})

    if solution:
        print("\nFinal Solution:")
        for region in neighbors:
            print(f"  {region:<3} : {solution[region]}")

        # Verify every constraint
        valid = all(
            solution[r] != solution[n] for r in neighbors for n in neighbors[r]
        )
        print("\nAll constraints satisfied:", valid)
    else:
        print("No solution exists with the given colours.")


if __name__ == "__main__":
    main()
