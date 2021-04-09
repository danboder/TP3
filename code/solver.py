import pygment
import copy
import random
import itertools
import math
import time


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

#######################################
# First functions 
#######################################

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


#######################################
# DESTRCTION FUNCTIONS
#######################################

def destroy_random(s,pyg,nb_destroyed_var):
    """
    Destroy a solution by selecting random days and removing the production on these day
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param nb_destroyed_var: number of production days we want to remove
    :return: a tuple with the destroyed solution, the production types that were removed and
    the indexes where the were removed in the list
    """
    l = len(s) # nb of days
    items_removed = []
    index_of_items = []
    for _ in range(nb_destroyed_var):
        i = random.randint(0,l-1)
        while i in index_of_items: # no 2 same days
            i = random.randint(0,l-1)
        index_of_items.append(i)
        items_removed.append(s[i])
        s[i] = -1 # no production on that day
    return s,items_removed,index_of_items

def destroy_zone(s,pyg,nb_destroyed_var):
    """
    Destroy a solution by selecting random days in a time zone and removing the production on these day
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param nb_destroyed_var: number of production days we want to remove
    :return: a tuple with the destroyed solution, the production types that were removed and 
    the indexes where the were removed in the list
    """
    sp = copy.deepcopy(s)
    l = len(sp) # nb of days
    items_removed = []
    index_of_items = []
    center = random.randint(0,l-1) # for ND dépendance
    range_from_center = nb_destroyed_var + 4
    for _ in range(nb_destroyed_var):

        i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 
        # taken by random but close to chosen center (+/- range)
        # allows destruction to be in a certain time zone
        while i in index_of_items: # no 2 same days
            i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 

        index_of_items.append(i)
        items_removed.append(sp[i])
        sp[i] = -1
    return sp,items_removed,index_of_items

def destroy_zone_large(s,pyg,nb_destroyed_var):
    """
    Destroy a solution by selecting random days in a bigger time zone and removing the production on these day
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param nb_destroyed_var: number of production days we want to remove
    :return: a tuple with the destroyed solution, the production types that were removed and 
    the indexes where the were removed in the list
    """
    sp = copy.deepcopy(s)
    l = len(sp) # nb of days
    items_removed = []
    index_of_items = []
    center = random.randint(0,l-1) # for ND dépendance
    range_from_center = nb_destroyed_var + 15
    for _ in range(nb_destroyed_var):

        i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 
        # taken by random but close to chosen center (+/- range)
        # allows destruction to be in a certain time zone
        while i in index_of_items: # no 2 same days
            i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 

        index_of_items.append(i)
        items_removed.append(sp[i])
        sp[i] = -1
    return sp,items_removed,index_of_items

#######################################
# CONSTRUCTION FUNCTIONS
#######################################

def reconstruct_cp_zone(sp,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible by testing all possibilities for all the items removed placed at 
    all days where there is place in a time frame
    :param s: destroyed solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = None
    best_score = 10e8
    # for new_items in itertools.permutations(items_removed):

    margin = 2
    range_min, range_max = min(index_of_items) - margin, max(index_of_items) + margin
    no_production_days = [i for i,t in enumerate(sp) if t == -1 and i >= range_min and i <= range_max]
    for comb in itertools.combinations(no_production_days,len(index_of_items)):
        for new_index in itertools.permutations(comb):
            s = copy.deepcopy(sp)
            for i,index in enumerate(new_index):
                s[index] = items_removed[i]
            if pyg.verify_solution(s):
                score = pyg.solution_total_cost(s)
                if score < best_score:
                    best_score = score
                    best_s = s
    return best_s

def reconstruct_cp_zone_large(sp,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible by testing all possibilities for all the items removed placed at 
    all days where there is place in a bigger time frame
    :param s: destroyed solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = None
    best_score = 10e8
    # for new_items in itertools.permutations(items_removed):

    margin = 20
    range_min, range_max = min(index_of_items) - margin, max(index_of_items) + margin
    no_production_days = [i for i,t in enumerate(sp) if t == -1 and i >= range_min and i <= range_max]
    for comb in itertools.combinations(no_production_days,len(index_of_items)):
        for new_index in itertools.permutations(comb):
            s = copy.deepcopy(sp)
            for i,index in enumerate(new_index):
                s[index] = items_removed[i]
            if pyg.verify_solution(s):
                score = pyg.solution_total_cost(s)
                if score < best_score:
                    best_score = score
                    best_s = s
    return best_s

def reconstruct_random_1(sp,pyg,items_removed,index_of_items):
    """
    returns one valid permutations taken by random for the removed items
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: solution computed
    """
    permutations = []
    range_min, range_max = min(index_of_items), max(index_of_items)
    no_production_days = [i for i,t in enumerate(sp) if t == -1 and i >= range_min and i <= range_max]
    for comb in itertools.combinations(no_production_days,len(index_of_items)):
        permutations += list(itertools.permutations(comb))

    valid = False
    while not valid:
        s = copy.deepcopy(sp)
        new_index = random.choice(permutations)
        for i,index in enumerate(new_index):
            s[index] = items_removed[i]
        valid = pyg.verify_solution(s)
    return s

def reconstruct_random_many(sp,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible from 20 different permutations
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = None
    best_score = 1e10

    permutations = []
    range_min, range_max = min(index_of_items), max(index_of_items)
    no_production_days = [i for i,t in enumerate(sp) if t == -1 and i >= range_min and i <= range_max]
    for comb in itertools.combinations(no_production_days,len(index_of_items)):
        permutations += list(itertools.permutations(comb))
    
    for new_items in random.sample(permutations,min(20,len(permutations))):
        s = copy.deepcopy(sp)
        for i,item in enumerate(new_items):
            s[index_of_items[i]] = item
        if pyg.verify_solution(s):
            score = pyg.solution_total_cost(s)
            if score < best_score:
                best_score = score
                best_s = s
    return best_s

#######################################
# ACCEPTANCE FUNCTION
#######################################

def acceptSolution(sp,s,pyg,T):
    """
    Acceptance of the new solution based on their score with temperature (simulated annealing)
    :param sp: new solution proposed
    :param s: current solution of the problem
    :param pyg: object describing the input problem
    :param T: Temperature of the model
    :return: boolean. True means that we accept the new solution sp
    """
    fsp = pyg.solution_total_cost(sp)
    fs = pyg.solution_total_cost(s)
    delta = fsp - fs
    return delta < 0 or random.random() < math.exp(-delta/T)

#######################################
# ALNS
#######################################

def ALNS(pyg):
    # for restarts
    best_s = None
    best_f = 1e10

    #########################
    # HYPERPARAMETRES
    iterations = 10000
    nb_destroyed_var = 3
    temperature = 4
    alphaT = 0.8
    reheat = 5
    iterations_to_reheat = 150
    #########################

    s,fs = solve_greedy(pyg)
    star = s
    fstar = fs
    no_change_iter = 0
    T = temperature

    # ALNS
    omega_m = [destroy_random,destroy_zone,destroy_zone_large]
    rho_m = [1,1,1]
    omega_p = [reconstruct_cp_zone,reconstruct_cp_zone_large,reconstruct_random_1,reconstruct_random_many]
    rho_p = [1,1,1,1]

    for k in range(iterations):
        if k%20 == 0:
            print(f"Iter {k}, best solution's cost : {pyg.solution_total_cost(star)}")
        if no_change_iter > iterations_to_reheat:
            print("REHEAT")
            s = star
            no_change_iter = 0
            temperature += reheat
        
        sp,items_removed,index_of_items = destroy_zone(s,pyg,nb_destroyed_var)
        # sp = reconstruct(s,pyg,items_removed,index_of_items)
        sp = reconstruct_cp_zone(sp,pyg,items_removed,index_of_items)
        if acceptSolution(sp,s,pyg,temperature): 
            s = sp
        else: 
            no_change_iter += 1
        fsp = pyg.solution_total_cost(sp)
        if fsp < fstar:
            no_change_iter = 0
            star = sp
            fstar = fsp
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
