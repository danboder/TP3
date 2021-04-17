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
                break # we stop at first better solution found
    return best_sol, best_score

def hill_climbing_fast(pyg,solution):
    star = solution
    fstar = pyg.solution_total_cost(star)
    better = True
    while better:
        better = False
        for i in range(len(star)-1):
            new_s = copy.deepcopy(star)
            new_s[i], new_s[i+1] = star[i+1],star[i]
            if pyg.verify_solution(new_s):
                fs = pyg.solution_total_cost(new_s)
                if fs < fstar:
                    star = new_s
                    fstar = fs
                    better = True
    return star,fstar

def hill_climbing_fast2(pyg,solution):
    star = solution
    fstar = pyg.solution_total_cost(star)
    better = True
    while better:
        better = False
        for i in range(pyg.nDays):
            if star[i] != -1:
                type_i = star[i]
                for j in reversed(range(i)):
                    if star[j] == type_i:
                        new_s = copy.deepcopy(star)
                        new_s[i] = -1 # remove at i
                        list.insert(new_s,j,type_i)
                        for k in reversed(range(j)):
                            if new_s[k] == -1:
                                new_ss = copy.deepcopy(new_s)
                                new_ss.pop(k)
                                score = pyg.solution_total_cost(new_ss)
                                if score < fstar:
                                    star = new_ss
                                    fstar = score
                                    better = True
                                    break
                        if better:
                            break
                if better:
                    break
    return star,fstar

#######################################
# Greedy Random Restart initial solution
#######################################

def solve_greedy_random(pyg, seed):
    """
    Greedy random
    -> Same as greedy solution, but includes a random seeded shuffle of queues
    """
    output = [-1 for _ in range(pyg.nDays)]
    queue = []
    for i in reversed(range(pyg.nDays)):
        for j in range(pyg.nProducts):
            if pyg.order(j, i):
                queue.append(j)
        # Add random process (shuffle list according to random seed) #
        random.Random(seed + i + j).shuffle(queue)
        if len(queue) != 0:
            output[i] = queue[0]
            queue.pop(0)
    return output, pyg.solution_total_cost(output)

def solve_restart_GR(pyg, n_trial, n_re):

    best_cost = 1000000
    best_solution = None

    for i in range(n_trial):
        # Greedy Random construction
        cur_sol, cur_cost = solve_greedy_random(pyg, (n_re + 1)*i + random.randint(0, 10))
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_solution = cur_sol

    assert best_solution is not None

    return best_solution, best_cost

def solve_grasp_hc(pyg, n_trial, n_re):
    best_cost = 1000000
    best_solution = None

    for i in range(n_re):
        # Greedy Random construction
        cur_sol, cur_cost = solve_restart_GR(pyg, n_trial, i)
        cur_sol, cur_cost = hill_climbing_fast(pyg, cur_sol)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_solution = cur_sol

    best_solution, best_cost = hill_climbing_fast2(pyg, best_solution)

    assert best_solution is not None

    return best_solution, best_cost

#######################################
# UTILS
#######################################

def softmax(l):
    s = 0
    for li in l:
        s += math.exp(li)
    return [math.exp(li)/s for li in l]

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
        start = time.time()
        while i in index_of_items or sp[i] == -1: # no 2 same days
            if time.time() - start > 2: # this prevents the while to loop infinitely
                # but behaviour will be like complete random
                center = random.randint(0,l-1)
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
    center = random.randint(0,l-1)
    range_from_center = nb_destroyed_var
    to_add = 0
    for i in range(max(0,center - range_from_center), min(l, center + range_from_center + 1)):
        if sp[i] == -1:
            to_add += 1
    range_from_center += to_add + 1 # we add space to avoid all the -1 spots
        
    for _ in range(nb_destroyed_var):

        i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 
        # taken by random but close to chosen center (+/- range)
        # allows destruction to be in a certain time zone
        start = time.time()
        while i in index_of_items or sp[i] == -1: # no 2 same days
            if time.time() - start > 2: # this prevents the while to loop infinitely
                # but behaviour will be like complete random
                center = random.randint(0,l-1)
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
    range_from_center = nb_destroyed_var + 5
    to_add = 0
    for i in range(max(0,center - range_from_center), min(l, center + range_from_center + 1)):
        if sp[i] == -1:
            to_add += 1
    range_from_center += to_add # we add space to avoid all the -1 spots

    for _ in range(nb_destroyed_var):

        i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 
        # taken by random but close to chosen center (+/- range)
        # allows destruction to be in a certain time zone
        start = time.time()
        while i in index_of_items: # no 2 same days
            if time.time() - start > 2: # this prevents the while to loop infinitely
                # but behaviour will be like complete random
                center = random.randint(0,l-1)
            i = random.randint(max(0,center - range_from_center), min(l-1, center + range_from_center)) 

        index_of_items.append(i)
        items_removed.append(sp[i])
        sp[i] = -1
    return sp,items_removed,index_of_items

def destroy_zone_short(s,pyg,nb_destroyed_var):
    """
    Destroy a solution by selecting succesive days in a little time zone and removing the production on these day
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
    center = random.randint(0,l-1)
    c = 0
    i = 0
    while c < nb_destroyed_var:
        if center + i < l:
            if sp[center+i] != -1:
                index_of_items.append(center+i)
                items_removed.append(sp[center+i])
                sp[center+i] = -1
                c+=1
        if c < nb_destroyed_var and center - i > 0:
            if sp[center-i] != -1:
                index_of_items.append(center-i)
                items_removed.append(sp[center-i])
                sp[center-i] = -1
                c+=1
        i += 1
    return sp,items_removed,index_of_items

#######################################
# CONSTRUCTION FUNCTIONS
#######################################

def reconstruct_cp_zone(sp,s_init,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible by testing all possibilities for all the items removed placed at 
    all days where there is place in a time frame
    :param sp: destroyed solution of the problem
    :param s_init: current solution of the problem
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
    :param sp: destroyed solution of the problem
    :param s_init: current solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = s_init
    best_score = 1e10
    # for new_items in itertools.permutations(items_removed):

    margin = 4
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
    :param sp: destroyed solution of the problem
    :param s_init: current solution of the problem
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
        for i,index in enumerate(new_index):
            s[index] = items_removed[i]
        valid = pyg.verify_solution(s)
        if time.time() - start > 2:
            return s_init
    return s

def reconstruct_random_many(sp,s_init,pyg,items_removed,index_of_items):
    """
    Find the best reconstruction possible from 20 different permutations
    :param sp: destroyed solution of the problem
    :param s_init: current solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: best valid solution found
    """
    best_s = s_init
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
            for i,index in enumerate(new_index):
                s[index] = items_removed[i]
            if pyg.verify_solution(s):
                valid = True
                nb += 1
                score = pyg.solution_total_cost(s)
                if score < best_score:
                    best_score = score
                    best_s = s
            if time.time() - start > 2:
                nb += 1
                valid = True
    return best_s

def reconstruct_types(sp,s_init,pyg,items_removed,index_of_items):
    """
    reconstruct a solution by trying to combine types
    This code will try all the combinations to fill the removed items in all the no production days contained in a range. 
    The filling will be based on the types of the items and its potential neighbors. If these match, we likely want them to be
    next to each other to avoid the transition of production fee. 
    :param sp: destroyed solution of the problem
    :param s_init: current solution of the problem
    :param pyg: object describing the input problem
    :param items_removed: list of the elements that were removed from s (we need to reinsert these to have a valid solution)
    :param index_of_items: list of the indexes of where the items were taken
    :return: solution computed
    """
    range_min, range_max = min(index_of_items), max(index_of_items) # range : between lowest and highest day
    no_production_days = [i for i,t in enumerate(sp) if t == -1 and i >= range_min and i <= range_max]
    # get the days where there is no production in the range of days chosen

    types = []
    for removed in items_removed: # for each item removed
        same = [0 for _ in range(len(no_production_days))]
        # check for each plausible future position, he will be put near a neighbour having same type of production
        for i,day in enumerate(no_production_days):
            if day > 0:
                if sp[day - 1] == removed:
                    same[i] += 1
            if day < pyg.nDays - 1:
                if sp[day + 1] == removed:
                    same[i] += 1
        types.append(same)
    

    s = sorted(zip(types,items_removed),key=lambda e: max(e[0]),reverse=True) 

    # sort by element with highest priority. Highest priority will be the elements where the have the more neighbors in common
    # we sort the items_removed list following the same order of the types list
    types = [x for x,_ in s]
    items_removed = [x for _,x in s]

    indexes = [list(map(lambda e : no_production_days[e[0]],sorted(enumerate(li),key=lambda elt: elt[1],reverse=True))) for li in types]
    # list containing, in order of priority, the day to try for an item, for all items_removed

    valid = False
    couter = 0
    index_for_each = [0 for _ in range(len(items_removed))] # will help trying the combinations
    l = len(index_for_each)
    start = time.time()
    while not valid:
        if time.time() - start > 10:
            print("Time Exceeded")
            s = copy.deepcopy(sp)
            for i,index in enumerate(index_of_items):
                s[index] = items_removed[i]
            print(pyg.verify_solution(s))

            if pyg.verify_solution(s_init): return s_init
            assert False
        m = map(lambda e:indexes[e[0]][e[1]],enumerate(index_for_each)) # check that each element will go at a different place
        if len(set(m)) == l:
            # print(index_for_each)
            s = copy.deepcopy(sp)
            for i,removed in enumerate(items_removed):
                s[indexes[i][index_for_each[i]]] = removed
            valid = pyg.verify_solution(s)
        if not valid:
            # changing the next combination to try by incrementing the counter
            couter += 1
            for i,index in enumerate(reversed(range(len(items_removed)))):
                if couter % len(no_production_days)**i == 0: 
                    index_for_each[index] = (index_for_each[index] + 1) % len(no_production_days) 
                    # we want last index to change the fastest
                    # first indexes (refereing to the elements having chance being placed near similar types) should change last
                    # last element will change every 1 iteration ( = go to next day to try )
                    # previous element will change every len(no_production_day). So that, we have been able to try all possibilities with element after
                    # element before will change every len(no_production_day)**2 iterations, etc for all other elements
    return s

#######################################
# ALNS FUNCTIONS
#######################################

def update_weights(rho_m,rho_p,i_d,i_r,psi,lambda_w):
    """
    Function to update the weights related to the destruction and construction functions
    :param rho_m: weights related to the destruction functions
    :param rho_p: weights related to the construction functions
    :param i_d: index of the destruction fonction used
    :param i_r: index of the reconstruction fonction used
    :param psi: Psi value used for the update
    :param lambda_w: defines by how much we update the variable 
    """
    rho_m[i_d] = lambda_w*rho_m[i_d] + (1-lambda_w)*psi
    rho_p[i_r] = lambda_w*rho_p[i_r] + (1-lambda_w)*psi

def ALNS(pyg):
    """
    Using ALNS to solve our instance problem
    :param pyg: instance of the problem
    """

    #########################
    # HYPERPARAMETRES
    time_allowed = 10 # in minutes
    nb_destroyed_var = 5
    temperature = 1
    alphaT = 0.8
    reheat = 1000
    iterations_to_reheat = 25
    lambda_w = 0.95
    hill_climbing_every = 10
    W = [4,3,2,0.5] # w1 > w2 > w3 > w4
    d_to_use = [0,1,0,1] # destruction fonctions to use (1 = we use it)
    r_to_use = [1,0,1,1,1] # same for reconstruction
    #########################
    print_every = 70

    destroy_functions = [destroy_random,destroy_zone,destroy_zone_large,destroy_zone_short]
    omega_m = [destroy_functions[i] for i,b in enumerate(d_to_use) if b == 1]
    rho_m = [1 for i,b in enumerate(d_to_use) if b == 1]
    
    construct_functions = [reconstruct_cp_zone,reconstruct_cp_zone_large,reconstruct_random_1,reconstruct_random_many,reconstruct_types]
    omega_p = [construct_functions[i] for i,b in enumerate(r_to_use) if b == 1]
    rho_p = [1 for i,b in enumerate(r_to_use) if b == 1]

    print(50*"=")
    print("Destroyed Variables :",nb_destroyed_var)
    print("Temperature :",temperature)
    print("alphaT :",alphaT)
    print("Reheat :",reheat)
    print("iterations_to_reheat :",iterations_to_reheat)
    print("lambda_w :",lambda_w)
    print("hill_climbing_every :",hill_climbing_every)
    print(50*"=")
    print("Destruct Functions",list(map(lambda f: f.__name__, omega_m)))
    print("Construction Functions", list(map(lambda f: f.__name__, omega_p)))
    print("W",W)
    print(50*"=")

    start = time.time()

    # get good first solution
    # s,fs = solve_greedy(pyg)
    # s,fs = hill_climbing_fast(pyg,s)
    # s,fs = hill_climbing_fast2(pyg,s)
    # s,fs = hill_climbing_fast(pyg,s)
    # s,fs = hill_climbing_fast2(pyg,s)
    s, fs = solve_grasp_hc(pyg, 1500, 4)

    star = s
    fstar = fs
    no_change_iter = 0
    T = temperature

    k = -1
    while time.time() - start < time_allowed * 60:
        k += 1
        if k % print_every == 0:
            print(f"Iter {k}, current solution : {fs}, best solution : {fstar}")
            print('Rho -',rho_m)
            print('Rho +',rho_p)
        if k % hill_climbing_every == 0:
            s,fs = hill_climbing_fast(pyg,s)
            if fs < fstar: # check if sp is better    
                no_change_iter = 0
                star = s
                fstar = fs
                print(f"{k} New best with HC : {fstar}")
        if no_change_iter > iterations_to_reheat:
            print("REHEAT")
            s = star
            no_change_iter = 0
            T += reheat
        
        i_d = random.choices(list(range(len(rho_m))),rho_m)[0]
        d = omega_m[i_d]
        i_r = random.choices(list(range(len(rho_p))),rho_p)[0]
        r = omega_p[i_r]

        # print(d)
        sp,items_removed,index_of_items = d(s,pyg,nb_destroyed_var)
        # print(r)
        sp = r(sp,s,pyg,items_removed,index_of_items)

        # acceptation of the solution with Simulated Annealing
        fsp = pyg.solution_total_cost(sp)
        # fs = pyg.solution_total_cost(s)
        delta = fsp - fs
        if delta < 0:
            no_change_iter = 0
            s = sp
            fs = fsp
            psi = W[1] # w2 if solution is better
        elif delta != 0 and random.random() < math.exp(-delta/T):
            print("accepted with probability",math.exp(-delta/T))
            s = sp
            fs = fsp
            psi = W[2] # w3 if solution is accepted but not better
        else: 
            no_change_iter += 1
            psi = W[3] # w4 if solution is rejected
        
        if fsp < fstar: # check if sp is better    
            no_change_iter = 0
            star = sp
            fstar = fsp
            psi = W[0] # w1 if solution is the best found yet
            print(f"{k} New best : {fstar}")

        update_weights(rho_m,rho_p,i_d,i_r,psi,lambda_w) 

        T *= alphaT
    return star,pyg.solution_total_cost(star)



def solve_advance(pyg):
    """
    Advanced solution of the problem
    :param pyg: object describing the input problem
    :return: a tuple (solution, cost) where solution is a
    list of the produced items and cost is the cost of the
    solution
    """
    # SET SEED
    seed = random.randint(0,10000000)
    print("SEED",seed)
    random.seed(seed)


    # TODO implement here your solution
    # return solve_greedy(pygment)

    # start = time.time()
    # s,fs = solve_greedy(pyg)
    # s,fs = hill_climbing_fast(pyg,s)
    # s,fs = hill_climbing_fast2(pyg,s)
    # s,fs = hill_climbing_fast(pyg,s)
    # s,fs = hill_climbing_fast2(pyg,s)
    # print(time.time() - start)
    s,fs = ALNS(pyg)
    return s,fs
