import pygment


def solve_greedy(pyg):
    """
    Greedy solution of the problem
    :param pyg: object describing the input problem
    :return: a tuple (solution, cost) where solution is a
    list of the produced items and cost is the cost of the
    solution
    """
    output = [-1 for _ in range(pyg.nDays)]
    queue = []
    for i in reversed(range(pyg.nDays)):
        for j in range(pyg.nProducts):
            if pyg.order(j, i):
                queue.append(j)
        if len(queue) != 0:
            output[i] = queue[0]
            queue.pop(0)
    return output, pyg.solution_total_cost(output)


def solve_advance(pyg):
    """
    Advanced solution of the problem
    :param pyg: object describing the input problem
    :return: a tuple (solution, cost) where solution is a
    list of the produced items and cost is the cost of the
    solution
    """
    # TODO implement here your solution
    return solve_greedy(pygment)
