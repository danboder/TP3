import eternity_puzzle
import math
import random

import numpy as np
import copy


def solve_GRASP(eternity_puzzle, n_trial):
    """
    Random solution of the problem (best of n_trial random solution generated)
    :param eternity_puzzle: object describing the input
    :param n_trial: number of random solution generated
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution, the solution is the best among the n_trial generated ones
    """
    best_n_conflict = 1000000
    best_solution = None

    for i in range(n_trial):
        # Greedy Random construction
        cur_sol, cur_n_conflict = solve_greedyRandom(i, eternity_puzzle)
        # Local Search
        N = fast_neighborhood(cur_sol, eternity_puzzle)
        loc_best_conflict = eternity_puzzle.get_total_n_conflict(N[0])
        # Best choice from neighborhood
        for j in range(len(N)):
            loc_cur_conflict = eternity_puzzle.get_total_n_conflict(N[j])
            if loc_cur_conflict < loc_best_conflict:
                loc_best_conflict = loc_cur_conflict
                cur_sol = N[j]
                cur_n_conflict = loc_cur_conflict
        # Random Choice from neighborhood (Faster)
            # cur_sol = N[random.randint(0, (len(N) - 1))]
            # cur_n_conflict = eternity_puzzle.get_total_n_conflict(cur_sol)
        # Update Best
        if cur_n_conflict < best_n_conflict:
            best_n_conflict = cur_n_conflict
            best_solution = cur_sol

    assert best_solution != None

    return best_solution, best_n_conflict

def solve_greedyRandom(alpha, puzzle):

    solution = []
    remaining_piece = copy.deepcopy(puzzle.piece_list)
    # rnd_color = 0
    rnd_color = alpha % 26
    for i in range(puzzle.n_piece):
        # Greedy choice
        best_n_conflict = 1000
        for j in range(len(remaining_piece)):
            for k in range(4):
                cur_sol = copy.deepcopy(solution)
                append_piece = copy.deepcopy(remaining_piece[j])
                append_piece.rotate(k)
                cur_sol.append(append_piece)
                # Append grey tiles to pad the solution
                temp_sol = copy.deepcopy(cur_sol)
                for l in range(len(remaining_piece)):
                    grey_idx = i + l
                    grey_piece = eternity_puzzle.Piece(rnd_color, (rnd_color, rnd_color, rnd_color, rnd_color))
                    temp_sol.append(grey_piece)

                cur_n_conf = puzzle.get_total_n_conflict(temp_sol)
                if cur_n_conf < best_n_conflict:
                    piece_idx = j
                    permutation_idx = k
                    best_n_conflict = cur_n_conf

        piece = remaining_piece[piece_idx]

        piece.rotate(permutation_idx)

        solution.append(piece)
        remaining_piece.remove(piece)

    return solution, puzzle.get_total_n_conflict(solution)

def solve_random(eternity_puzzle):
    """
    Random solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    solution = []
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)

        piece = remaining_piece[piece_idx]

        permutation_idx = np.random.choice(np.arange(4))

        piece.rotate(permutation_idx)

        solution.append(piece)

        remaining_piece.remove(piece)

    return solution, eternity_puzzle.get_total_n_conflict(solution)


def solve_best_random(eternity_puzzle, n_trial):
    """
    Random solution of the problem (best of n_trial random solution generated)
    :param eternity_puzzle: object describing the input
    :param n_trial: number of random solution generated
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution, the solution is the best among the n_trial generated ones
    """
    best_n_conflict = 1000000

    best_solution = None

    for i in range(n_trial):

        cur_sol, cur_n_conflict = solve_random(eternity_puzzle)

        if cur_n_conflict < best_n_conflict:
            best_n_conflict = cur_n_conflict
            best_solution = cur_sol

    assert best_solution != None

    return best_solution, best_n_conflict


####################################################
# SIMULATED ANNEALING

def neighborhood(s,puzzle):
    neighbors = []
    list_index = [i for i in range(puzzle.n_piece)]
    for i,piece in enumerate(s):
        to_switch_with = random.choice(list_index) # select element random
        for rot in puzzle.generate_rotation(piece): # get all rotations of the piece (not the random one)
            new_s = copy.deepcopy(s)
            new_s[i] = new_s[to_switch_with] #random element take place at i
            # This returns a tuple and not a piece
            # new_s[to_switch_with] = rot # we set the piece with rotation at the random element's place
            rot_piece = eternity_puzzle.Piece(i, rot)
            new_s[to_switch_with] = rot_piece
            neighbors.append(new_s)
    return neighbors

def fast_neighborhood(s,puzzle):
    neighbors = []
    list_index = [i for i in range(puzzle.n_piece)]
    for i,piece in enumerate(s):
        to_switch_with = random.choice(list_index) # select element random
        rot = random.choice(puzzle.generate_rotation(piece)) # get one rotation of the piece (not the random one)
        new_s = copy.deepcopy(s)
        new_s[i] = new_s[to_switch_with] #random element take place at i
        rot_piece = eternity_puzzle.Piece(i, rot)
        new_s[to_switch_with] = rot_piece
        neighbors.append(new_s)
    return neighbors

def simulated_annealing(s, fs, puzzle):

    best_s = None
    best_f = 1e10

    #########################################
    # HYPER PARAMETERS
    nb_restart = 20
    T = 3
    alphaT = 0.92
    betaT = 1
    re_lim = 20
    fast_neighbor = True
    #########################################

    for _ in range(nb_restart):

        star = s
        fstar = fs

        re_count = 0
        maxT = T

        change = True
        for i in range(1000):
            if re_count >= re_lim:
                G = fast_neighborhood(star,puzzle) if fast_neighbor else neighborhood(star,puzzle)
                # V = validate_neighboorhood(G, clients, W)
                re_count = 0 # reset counter
                T = min(T + betaT, maxT)
            if change:
                G = fast_neighborhood(s,puzzle) if fast_neighbor else neighborhood(s,puzzle)
                change = False
            c = G[random.randint(0,len(G)-1)]
            fc = puzzle.get_total_n_conflict(c)
            delta = fc - fs
            if delta < 0 or random.random() < math.exp(-delta/T):
                change = True
                s = c
                fs = fc
                if fs < fstar:
                    re_count = 0
                    star = s
                    fstar = fs
                    # print("improvement at",i,":",fstar)
            else:
                re_count += 1
            T = alphaT * T

        if fstar < best_f:
            print("update:", fstar)
            best_f = fstar
            best_s = star
    
    return best_s,best_f

####################################################


####################################################
# GENETIC ALGORITHM
####################################################

def children(parent1,parent2,piece_list):
    n = len(piece_list)
    # # IDEA 1 : split in half
    # child1 = parent1[:int(n/2)] + parent2[int(n/2):]
    # child2 = parent2[:int(n/2)] + parent1[int(n/2):]
    # child1,child2 = equilibrate(copy.deepcopy(child1),copy.deepcopy(child2),piece_list)
    # #IDEA 2 : random
    # child1 = []
    # child2 = []
    # for i in range(n):
    #     if random.randint(0,1) == 0:
    #         child1.append(parent1[i])
    #         child2.append(parent2[i])
    #     else:
    #         child2.append(parent1[i])
    #         child1.append(parent2[i])
    # child1,child2 = equilibrate(child1,child2,piece_list)
    # IDEA 3 : PMX
    child1,child2 = PMX(parent1,parent2,piece_list)
    return child1,child2

def PMX(parent1,parent2,piece_list):
    n = len(piece_list)
    inter1 = random.randint(0,n-2)
    inter2 = random.randint(inter1+1, n-1)

    cross1 = parent1[inter1:inter2+1]
    cross2 = parent2[inter1:inter2+1]
    to_switch = dict()
    for i in range(len(cross1)):
        keys = list(to_switch.keys())
        p1id = cross1[i].id
        p2id = cross2[i].id
        if p1id in keys:
            to_switch[p2id] = to_switch[p1id]
            to_switch[to_switch[p1id]] = p2id
            to_switch.pop(p1id)
        elif p2id in keys:
            to_switch[p1id] = to_switch[p2id]
            to_switch[to_switch[p2id]] = p1id
            to_switch.pop(p2id)
        else:
            to_switch[p1id] = p2id
            to_switch[p2id] = p1id
    keys = list(to_switch.keys())
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    for i in range(n):
        if i < inter1 or i > inter2:
            id1 = child1[i].id
            if id1 in keys:
                child1[i] == piece_list[to_switch[id1]]
            id2 = child2[i].id
            if id2 in keys:
                child2[i] == piece_list[to_switch[id2]]
    
    child1 = child1[:inter1] + cross2 + child1[inter2+1:]
    child2 = child2[:inter1] + cross1 + child2[inter2+1:]
    return child1, child2


def equilibrate(child1,child2,piece_list):
    # remove double pieces and replace with missing pieces
    n = len(piece_list)
    piece_inChild1 = [p.id for p in child1]
    missing_inChild1 = [p.id for p in piece_list if p.id not in piece_inChild1]

    indexes = [[] for _ in range(n)]
    for i,p in enumerate(child1):
        indexes[p.id].append(i) # check if some piece are in double in the child
    for l in indexes:
        if len(l) > 1: # if double
            # random for which one we change
            random.shuffle(l) 
            e = l[0]
            # random chosen element that is missing is child
            i = random.randint(0,len(missing_inChild1) - 1)
            id = missing_inChild1.pop(i)
            p = piece_list[id]
            # replace
            child1[e] = p
    
    # same for child 2
    piece_inChild2 = [p.id for p in child2]
    missing_inChild2 = [p.id for p in piece_list if p.id not in piece_inChild2]

    indexes = [[] for _ in range(n)]
    for i,p in enumerate(child2):
        indexes[p.id].append(i)
    for l in indexes:
        if len(l) > 1: 
            random.shuffle(l) 
            e = l[0]
            i = random.randint(0,len(missing_inChild2) - 1)
            id = missing_inChild2.pop(i)
            p = piece_list[id]
            child2[e] = p

    return child1,child2


def mutation(child):
    # rotate (or not) for each piece
    for p in child:
        rot = random.randint(0,3)
        p.rotate(rot)

def genetic_algorithm(puzzle):

    print("="*30)
    print("USING GENETIC ALGORITHMS")
    print("="*30)

    piece_list = puzzle.piece_list

    ###############################
    # HYPERPARAMETERS
    size_population = 300
    assert size_population % 2 == 0, "Size population should be an even number"
    mutation_probability = 0.3
    ###############################
    print("Size population:",size_population)
    print("Mutation probability:",mutation_probability)
    print("="*30)

    # initialize population
    population = []
    for _ in range(size_population):
        population.append(solve_best_random(puzzle,20)[0])
    
    population = sorted(population,key=lambda s: puzzle.get_total_n_conflict(s))

    for k in range(200): # will be replace with time

        # classify from best to worst our population
        if k % 20 == 0:
            print(k,"Best solution",puzzle.get_total_n_conflict(population[0]))

        new_children = []
        for i in range(0,len(population),2):
            parent1,parent2 = population[i],population[i+1] # get two parents
            child1,child2 = children(parent1,parent2,piece_list) # get 2 childs
            # assert puzzle.verify_solution(child1), f"At {i}\nparent1 : {list(map(lambda p: p.id,parent1))}\nparent2 : {list(map(lambda p: p.id,parent2))}\nchild : {list(map(lambda p: p.id,child1))}"
            # assert puzzle.verify_solution(child2), f"At {i}\nparent1 : {list(map(lambda p: p.id,parent1))}\nparent2 : {list(map(lambda p: p.id,parent2))}\nchild : {list(map(lambda p: p.id,child2))}"
            if random.random() < mutation_probability: # mutate
                mutation(child1)
            if random.random() < mutation_probability: # mutate
                mutation(child2)
            new_children.append(child1)
            new_children.append(child2)
        
        population = population + new_children
        population = sorted(population,key=lambda s: puzzle.get_total_n_conflict(s))
        population = population[:int(len(population)/2)]

    population = sorted(population,key=lambda s: puzzle.get_total_n_conflict(s))
    best = population[0]
    return best, puzzle.get_total_n_conflict(best)



def solve_advance(eternity_puzzle):
    """
    Advanced solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    # TODO implement here your solution

    # solution = simulated_annealing(eternity_puzzle)
    solution = genetic_algorithm(eternity_puzzle)

    return solution
