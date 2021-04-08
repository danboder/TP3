import pygment
import copy
import random
import itertools
import math


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


def neighborhood(solution,pyg):
    neighbors = []
    for i in range(1, pyg.nDays):
        for j in range(i):
            new_s = copy.deepcopy(solution)
            e = new_s.pop(i)
            list.insert(new_s,j,e)
            if pyg.verify_solution(new_s): neighbors.append(new_s)
    return neighbors

def hill_climbing(pyg):
    sol,best_score = solve_greedy(pyg)
    best_sol = sol
    better = True
    while better:
        better = False
        neighbors = neighborhood(sol,pyg)
        for n in neighbors:
            s = pyg.solution_total_cost(n)
            if s < best_score:
                best_score = s
                best_sol = n
                better = True
    return best_sol, best_score

def destroy(s,pyg,nb_destroyed_var):
    # TODO
    # nature de la destruction : ND

    l = len(s) # nb of days
    items_removed = []
    index_of_items = []
    for _ in range(nb_destroyed_var):
        
        # ND aléatoire
        i = random.randint(0,l-1)

        while i in index_of_items:
            i = random.randint(0,l-1)
        index_of_items.append(i)
        items_removed.append(s[i])
    return items_removed,index_of_items

    # - dépendance (variables qui ont les mêmes carac) = toutes les variables dans une période de temps
    # - critique (variables qui induisent une hausse de cout) = transition engendre haut cout par exemple
    return

def reconstruct(s,pyg,items_removed,index_of_items):
    # print("items",items_removed)
    # print("indexes",index_of_items)
    s = copy.deepcopy(s)
    best_s = copy.deepcopy(s)
    best_score = pyg.solution_total_cost(s)
    for new_items in itertools.permutations(items_removed):
        for i,item in enumerate(new_items):
            s[index_of_items[i]] = item
        if pyg.verify_solution(s):
            score = pyg.solution_total_cost(s)
            if score < best_score:
                best_score = score
                best_s = copy.deepcopy(s)
                print("UPDATE")
    return best_s

def acceptSolution(sp,s,pyg,T):
    # TODO
    fsp = pyg.solution_total_cost(sp)
    fs = pyg.solution_total_cost(s)
    delta = fsp - fs
    return delta < 0 or random.random() < math.exp(-delta/T)

def LNS(pyg):
    s,score_s = solve_greedy(pyg)
    star = s
    score_star = score_s

    #########################
    # HYPERPARAMETRES
    iterations = 10
    nb_destroyed_var = 5
    temperature = 3
    alphaT = 0.9
    #########################


    for k in range(iterations):
        print(k)
        items_removed,index_of_items = destroy(s,pyg,nb_destroyed_var)
        sp = reconstruct(s,pyg,items_removed,index_of_items)
        if acceptSolution(sp,s,pyg,temperature):
            s = sp
        if pyg.solution_total_cost(sp) < score_star:
            star = sp
        temperature *= alphaT
    return star,pyg.solution_total_cost(star)



def solve_advance(pyg):
    """
    Advanced solution of the problem
    :param pyg: object describing the input problem
    :return: a tuple (solution, cost) where solution is a
    list of the produced items and cost is the cost of the
    solution
    """
    # TODO implement here your solution
    # return solve_greedy(pygment)

    # return hill_climbing(pyg)
    return LNS(pyg)
