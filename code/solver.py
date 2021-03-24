import math
import random

import numpy as np
import copy


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
####################################################

def neighborhood(s,puzzle):
    neighbors = []
    list_index = [i for i in range(puzzle.n_piece)]
    for i,piece in enumerate(s):
        to_switch_with = random.choice(list_index) # select element random
        for rot in puzzle.generate_rotation(piece): # get all rotations of the piece (not the random one)
            new_s = copy.deepcopy(s)
            new_s[i] = new_s[to_switch_with] #random element take place at i
            new_s[to_switch_with] = rot # we set the piece with rotation at the random element's place
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
        new_s[to_switch_with] = rot # we set the piece with rotation at the random element's place
        neighbors.append(new_s)
    return neighbors

def neighborhood_placed(s,puzzle):
    neighbors = []
    list_index = [i for i in range(puzzle.n_piece)]
    for i,piece in enumerate(s):
        to_switch_with = random.choice(list_index) # select element random
        while puzzle.piece_list[to_switch_with].getType() != piece.getType(): # but we want the same type (corner, wall or center)
            to_switch_with = random.choice(list_index)
        for rot in puzzle.generate_rotation(piece): # get all rotations of the piece (not the random one)
            new_s = copy.deepcopy(s)
            new_s[i] = new_s[to_switch_with] #random element take place at i
            new_s[to_switch_with] = rot # we set the piece with rotation at the random element's place
            neighbors.append(new_s)
    return neighbors

def fast_neighborhood_placed(s,puzzle):
    neighbors = []
    list_index = [i for i in range(puzzle.n_piece)]
    for i,piece in enumerate(s):
        to_switch_with = random.choice(list_index) # select element random
        while puzzle.piece_list[to_switch_with].getType() != piece.getType(): # but we want the same type (corner, wall or center)
            to_switch_with = random.choice(list_index)
        rot = random.choice(puzzle.generate_rotation(piece)) # get one rotation of the piece (not the random one)
        new_s = copy.deepcopy(s)
        new_s[i] = new_s[to_switch_with] #random element take place at i
        new_s[to_switch_with] = rot # we set the piece with rotation at the random element's place
        neighbors.append(new_s)
    return neighbors

def greedy_construction(alpha,puzzle):
    pieces = puzzle.piece_list
    size = puzzle.board_size
    ids_to_take = [i for i in range(len(pieces))]

    center_pieces = []
    wall_pieces = []
    corner_pieces = []

    for p in pieces:
        if p.isCorner():
            corner_pieces.append(p)
        elif p.isWall():
            wall_pieces.append(p)
        else:
            center_pieces.append(p)
    
    new_list = [None for _ in range(size**2)]
    p = random.choice(center_pieces)
    center_pieces.pop(center_pieces.index(p))
    line,col = 1,1
    new_list[size * line + col] = p


    ### POSITIONNING THE CENTER PIECES
    for line in range(1,size-1):
        for col in range(1,size-1): # for all center pieces
            if not (line == 1 and col == 1): # already filled
                k = size * line + col
                k_east = size * line + (col + 1)
                k_south = size * (line - 1) + col
                k_west = size * line + (col - 1)
                k_north = size * (line + 1) + col
                around = [
                    new_list[k_north] , # get piece north
                    new_list[k_south], # get piece south
                    new_list[k_east], # get piece east
                    new_list[k_west] # get piece west
                ]
                conflicts = []
                for p in center_pieces:
                    p.rotate(random.randint(0,3)) # rotate piece at random
                    nb_conflicts = 0
                    if around[0] != None: # north
                        if around[0].colors[1] != p.colors[0]: nb_conflicts += 1
                    if around[1] != None: # south
                        if around[1].colors[0] != p.colors[1]: nb_conflicts += 1
                    if around[2] != None: # west
                        if around[2].colors[3] != p.colors[2]: nb_conflicts += 1
                    if around[3] != None: # east
                        if around[3].colors[2] != p.colors[3]: nb_conflicts += 1
                    conflicts.append(nb_conflicts)
                
                conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, center_pieces),key=lambda e: e[0]))
                # this sorts the conflicts list and sorts the list of the pieces following the same order
                minf = conflicts_sorted[0]
                maxf = conflicts_sorted[-1] # get min and max value
                RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
                # get RCL
                chosen = random.choice(RCL) # choose by random in RCL
                center_pieces.pop(center_pieces.index(chosen))
                new_list[k] = chosen
     
    ### POSITIONNING THE WALL PIECES
    for line in range(size):
        for col in range(size):
            if ((line == 0 or line == size-1) and col != 0 and col != size - 1) or \
                ((col == 0 or col == size-1) and line != 0 and line != size - 1): # get only the wall places
                k = size * line + col
                k_east = size * line + (col + 1)
                k_south = size * (line - 1) + col
                k_west = size * line + (col - 1)
                k_north = size * (line + 1) + col
                # print(line,col,size)
                around = [
                    new_list[k_north] if line < size - 1 else None, new_list[k_south] if line > 0 else None, 
                    new_list[k_east] if col < size - 1 else None, new_list[k_west] if col > 0 else None ]
                conflicts = []
                for p in wall_pieces:
                    p.rotate(random.randint(0,3)) # rotate piece at random
                    nb_conflicts = 0
                    if around[0] != None: # north
                        if around[0].colors[1] != p.colors[0]: nb_conflicts += 1
                    if around[1] != None: # south
                        if around[1].colors[0] != p.colors[1]: nb_conflicts += 1
                    if around[2] != None: # west
                        if around[2].colors[3] != p.colors[2]: nb_conflicts += 1
                    if around[3] != None: # east
                        if around[3].colors[2] != p.colors[3]: nb_conflicts += 1
                    conflicts.append(nb_conflicts)
                
                conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, wall_pieces),key=lambda e: e[0]))
                minf = conflicts_sorted[0]
                maxf = conflicts_sorted[-1]
                RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
                chosen = random.choice(RCL)
                wall_pieces.pop(wall_pieces.index(chosen))
                new_list[k] = chosen
    
    ### POSITIONNING THE CORNER PIECES
    for line in [0,size-1]:
        for col in [0,size-1]: # corners
            k = size * line + col
            k_east = size * line + (col + 1)
            k_south = size * (line - 1) + col
            k_west = size * line + (col - 1)
            k_north = size * (line + 1) + col
            around = [
                new_list[k_north] if line < size - 1 else None, new_list[k_south] if line > 0 else None, 
                new_list[k_east] if col < size - 1 else None, new_list[k_west] if col > 0 else None ]
            conflicts = []
            for p in corner_pieces:
                p.rotate(random.randint(0,3))
                nb_conflicts = 0
                if around[0] != None:
                    if around[0].colors[1] != p.colors[0]: nb_conflicts += 1
                if around[1] != None:
                    if around[1].colors[0] != p.colors[1]: nb_conflicts += 1
                if around[2] != None:
                    if around[2].colors[3] != p.colors[2]: nb_conflicts += 1
                if around[3] != None:
                    if around[3].colors[2] != p.colors[3]: nb_conflicts += 1
                conflicts.append(nb_conflicts)
            
            conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, corner_pieces),key=lambda e: e[0]))
            minf = conflicts_sorted[0]
            maxf = conflicts_sorted[-1]
            RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
            chosen = random.choice(RCL)
            corner_pieces.pop(corner_pieces.index(chosen))
            new_list[k] = chosen
    
    return new_list

def hill_climbing(s,steps,puzzle):
    best_s = s
    best_f = puzzle.get_total_n_conflict(best_s)
    for _ in range(steps):
        neighbors = fast_neighborhood_placed(best_s,puzzle) # keep center pieces in center same for wall pieces and corner pieces
        for n in neighbors:
            f = puzzle.get_total_n_conflict(n)
            if f < best_f:
                best_f = f
                best_s = n
    return best_s,best_f

def grasp(puzzle,alpha,nb_trials):
    star = None
    fstar = 1e8
    for _ in range(nb_trials):
        s = greedy_construction(alpha,puzzle)
        s,f = hill_climbing(s,20,puzzle)
        if f < fstar:
            fstar = f
            star = s
    return star,fstar

def simulated_annealing(puzzle):

    best_s = None
    best_f = 1e10

    #########################################
    # HYPER PARAMETERS
    nb_restart = 10
    T = 3
    alphaT = 0.92
    betaT = 1
    re_lim = 20
    fast_neighbor = True
    alpha_grasp = 0.2
    nb_tries_grasp = 10
    #########################################

    for _ in range(nb_restart):
        print("START")
        # s,fs = solve_best_random(puzzle,20)
        s,fs = grasp(puzzle,alpha_grasp,nb_tries_grasp)
        star = s
        fstar = fs

        re_count = 0
        maxT = T

        change = True
        for i in range(1000): # TODO : change with time
            if re_count >= re_lim:
                G = fast_neighborhood_placed(star,puzzle) if fast_neighbor else neighborhood_placed(star,puzzle)
                # V = validate_neighboorhood(G, clients, W)
                re_count = 0 # reset counter
                T = min(T + betaT, maxT)
            if change:
                G = fast_neighborhood_placed(s,puzzle) if fast_neighbor else neighborhood_placed(s,puzzle)
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
                    print("improvement at",i,":",fstar)
            else:
                re_count += 1
            T = alphaT * T

        if fstar < best_f:
            print("NEW BEST",fstar)
            best_f = fstar
            best_s = star
    
    return best_s,best_f



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

    solution = simulated_annealing(eternity_puzzle)
    # solution = genetic_algorithm(eternity_puzzle)

    return solution
