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
# DESTRUCTION FUNCTIONS
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
    sp = copy.deepcopy(s)
    l = len(sp) # nb of days
    items_removed = []
    index_of_items = []
    for _ in range(nb_destroyed_var):
        i = random.randint(0,l-1)
        while i in index_of_items: # no 2 same days
            i = random.randint(0,l-1)
        index_of_items.append(i)
        items_removed.append(sp[i])
        sp[i] = -1 # no production on that day
    return sp,items_removed,index_of_items

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
    range_from_center = nb_destroyed_var
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
    range_from_center = nb_destroyed_var + 7
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

def reconstruct_cp_zone(sp,s_init,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible by testing all possibilities for all the items removed placed at 
    all days where there is place in a time frame
    :param s: destroyed solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = s_init
    best_score = 1e10
    
    range_min, range_max = min(index_of_items), max(index_of_items)
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

def reconstruct_cp_zone_large(sp,s_init,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible by testing all possibilities for all the items removed placed at 
    all days where there is place in a bigger time frame
    :param s: destroyed solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = s_init
    best_score = 1e10
    # for new_items in itertools.permutations(items_removed):

    margin = 7
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

def reconstruct_random_1(sp,s_init,pyg,items_removed,index_of_items):
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
    start = time.time()
    while not valid:
        s = copy.deepcopy(sp)
        new_index = random.choice(permutations)
        if time.time() - start > 3:
            print("too long")
            new_index = index_of_items
            for i,index in enumerate(new_index):
                s[index] = items_removed[i]
            print("new")
            print(s)
            print(pyg.verify_solution(s,True))
            print("old")
            print(s_init)
            print(pyg.verify_solution(s_init,True))
            assert False
        for i,index in enumerate(new_index):
            s[index] = items_removed[i]
        valid = pyg.verify_solution(s)
    return s

def reconstruct_random_many(sp,s_init,pyg,items_removed,index_of_items):
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
    
    # for new_items in random.sample(permutations,min(20,len(permutations))):
    nb = 0
    while nb < 20:
        valid = False
        start = time.time()
        while not valid:
            s = copy.deepcopy(sp)
            new_index = random.choice(permutations)
            if time.time() - start > 3:
                print("too long")
                new_index = index_of_items
                for i,index in enumerate(new_index):
                    s[index] = items_removed[i]
                print("new")
                print(s)
                print(pyg.verify_solution(s,True))
                print("old")
                print(s_init)
                print(pyg.verify_solution(s_init,True))
                assert False
            for i,index in enumerate(new_index):
                s[index] = items_removed[i]
            if pyg.verify_solution(s):
                valid = True
                nb += 1
                score = pyg.solution_total_cost(s)
                if score < best_score:
                    best_score = score
                    best_s = s
    return best_s

#######################################
# ALNS FUNCTIONS
#######################################

def update_weights(rho_m,rho_p,i_d,i_r,psi,lambda_w):
    rho_m[i_d] = lambda_w*rho_m[i_d] + (1-lambda_w)*psi
    rho_p[i_r] = lambda_w*rho_p[i_r] + (1-lambda_w)*psi

def ALNS(pyg):
    # for restarts
    # best_s = None
    # best_f = 1e10

    #########################
    # HYPERPARAMETRES
    iterations = 2000
    nb_destroyed_var = 3
    temperature = 4
    alphaT = 0.8
    reheat = 5
    iterations_to_reheat = 150
    lambda_w = 0.8
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
    # W = [random.random() * 10 for _ in range(4)]
    W = [30,17,10,1]
    W = sorted(W,reverse=True) # w1 > w2 > w3 > w4
    print("W",W)

    for k in range(iterations):
        if k%20 == 0:
            print(f"Iter {k}, best solution's cost : {pyg.solution_total_cost(star)}")
            print('Rho -',rho_m)
            print('Rho +',rho_p)
        if no_change_iter > iterations_to_reheat:
            print("REHEAT")
            s = star
            no_change_iter = 0
            temperature += reheat
        
        i_d = random.choices(list(range(len(rho_m))),rho_m)[0]
        d = omega_m[i_d]
        i_r = random.choices(list(range(len(rho_p))),rho_p)[0]
        r = omega_p[i_r]

        sp,items_removed,index_of_items = d(s,pyg,nb_destroyed_var)
        sp = r(sp,s,pyg,items_removed,index_of_items)

        # acceptation of the solution with Simulated Annealing
        fsp = pyg.solution_total_cost(sp)
        fs = pyg.solution_total_cost(s)
        delta = fsp - fs
        if delta < 0:
            s = sp
            psi = W[1] # w2 if solution is better
        elif random.random() < math.exp(-delta/T):
            s = sp
            psi = W[2] # w3 if solution is accepted but not better
        else: 
            no_change_iter += 1
            psi = W[3] # w4 if solution is rejected
        
        if fsp < fstar: # check if sp is better    
            no_change_iter = 0
            star = sp
            fstar = fsp
            psi = W[0] # w1 if solution is the best found yet

        # if pyg.solution_total_cost(s) == fstar: 
        #     psi = W[0] # w1 if solution is the best found yet
        update_weights(rho_m,rho_p,i_d,i_r,psi,lambda_w) 

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
    return ALNS(pyg)
