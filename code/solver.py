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
    center = random.randint(0,l-1) # for ND dépendance
    for _ in range(nb_destroyed_var):
        
        # ND aléatoire
        # i = random.randint(0,l-1)

        # ND dépendance (variables qui ont les mêmes carac) = toutes les variables dans une période de temps
        to_add = 6
        i = random.randint(max(0,center - nb_destroyed_var - to_add), min(l-1, center + nb_destroyed_var + to_add)) 
        # on prend au hasard mais proche du centre (à +/- le nb d'élément à détruire)
        # permet que la destruction se fasse dans une periode de temps

        while i in index_of_items:
            i = random.randint(0,l-1)
        index_of_items.append(i)
        items_removed.append(s[i])
    return items_removed,index_of_items

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
    iterations = 5000
    nb_destroyed_var = 3
    temperature = 4
    alphaT = 0.95
    reheat = 3
    iterations_to_reheat = 50
    #########################
    no_change_iter = 0

    for k in range(iterations):
        if k%10 == 0:
            print(f"Iter {k}, best solution's cost : {pyg.solution_total_cost(star)}")
        items_removed,index_of_items = destroy(s,pyg,nb_destroyed_var)
        sp = reconstruct(s,pyg,items_removed,index_of_items)
        if acceptSolution(sp,s,pyg,temperature):
            s = sp
            no_change_iter = 0
        if pyg.solution_total_cost(sp) < score_star:
            star = sp
        temperature *= alphaT
        if s == star: no_change_iter += 1
        if no_change_iter > iterations_to_reheat:
            temperature += reheat
            no_change_iter
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
