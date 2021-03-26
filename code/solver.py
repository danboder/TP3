import math
import random
import time

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
# GRASP
####################################################
class Piece:
    def __init__(self,id,colors):
        self.id = id
        self.colors = colors
    def __str__(self):
        return f"{self.id} {self.colors}"
    def __repr__(self):
        return str(self.id)
    def __eq__(self, other):
        if other == None: return False
        return self.id == other.id

def greedy_construction(alpha,puzzle):
    """
    Greedy Construction for Grasp
    Separating all pieces by their type (Corner, Wall or Center).
    Construction starts by filling the center pieces, then wall pieces, then the corner pieces.
    We try to minimze conflicts.
    :param alpha: float to determine if we keep bad solutions or not
    :param puzzle: object describing the puzzle
    :return: a list of Pieces, being a solution for the puzzle    
    """

    WALLPIECE = Piece(-1,(0,0,0,0))

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

    new_list = [WALLPIECE for _ in range(size**2)]
    p = random.choice(center_pieces)
    center_pieces.pop(center_pieces.index(p))
    line,col = 1,1
    new_list[size * line + col] = p

    ### POSITIONNING THE CORNER PIECES
    for line in [0,size-1]:
        for col in [0,size-1]: # corners
            k = size * line + col
            k_east = size * line + (col + 1)
            k_south = size * (line - 1) + col
            k_west = size * line + (col - 1)
            k_north = size * (line + 1) + col
            around = [
                new_list[k_north] if line < size - 1 else WALLPIECE, new_list[k_south] if line > 0 else WALLPIECE, 
                new_list[k_west] if col > 0 else WALLPIECE, new_list[k_east] if col < size - 1 else WALLPIECE]
            
            wall_is = []
            if line >= size - 1: wall_is.append(0) # wall is north
            if line <= 0: wall_is.append(1) # south
            if col <= 0: wall_is.append(2) # west
            if col >= size - 1: wall_is.append(3) # east

            conflicts = []
            all_pieces = []
            for p in corner_pieces:
                for pi in puzzle.generate_rotation(p):
                    if pi.colors[wall_is[0]] == 0 and pi.colors[wall_is[1]] == 0: # check that rotation matches the corner
                        nb_conflicts = 0
                        if around[0].colors[1] != pi.colors[0]: nb_conflicts += 1 # north
                        if around[1].colors[0] != pi.colors[1]: nb_conflicts += 1 # south
                        if around[2].colors[3] != pi.colors[2]: nb_conflicts += 1 # west
                        if around[3].colors[2] != pi.colors[3]: nb_conflicts += 1 # east
                        conflicts.append(nb_conflicts)
                        all_pieces.append(pi)
            
            conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, all_pieces),key=lambda e: e[0]))
            minf = conflicts_sorted[0]
            maxf = conflicts_sorted[-1]
            RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
            chosen = random.choice(RCL)
            corner_pieces.pop(corner_pieces.index(chosen))
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
                    new_list[k_north] if line < size - 1 else WALLPIECE, new_list[k_south] if line > 0 else WALLPIECE, 
                    new_list[k_west] if col > 0 else WALLPIECE, new_list[k_east] if col < size - 1 else WALLPIECE]

                if line >= size - 1: wall_is = 0 # wall is north
                elif line <= 0: wall_is = 1 # south
                elif col <= 0: wall_is = 2 # west
                elif col >= size - 1: wall_is = 3 # east

                conflicts = []
                all_pieces = []
                for p in wall_pieces:
                    for pi in puzzle.generate_rotation(p):
                        if pi.colors[wall_is] == 0: # check that rotation matches the wall
                            nb_conflicts = 0
                            if around[0].colors[1] != pi.colors[0]: nb_conflicts += 1 # north
                            if around[1].colors[0] != pi.colors[1]: nb_conflicts += 1 # south
                            if around[2].colors[3] != pi.colors[2]: nb_conflicts += 1 # west
                            if around[3].colors[2] != pi.colors[3]: nb_conflicts += 1 # east
                            conflicts.append(nb_conflicts)
                            all_pieces.append(pi)
                
                conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, all_pieces),key=lambda e: e[0]))
                minf = conflicts_sorted[0]
                maxf = conflicts_sorted[-1]
                RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
                chosen = random.choice(RCL)
                wall_pieces.pop(wall_pieces.index(chosen))
                new_list[k] = chosen
        
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
                    new_list[k_west], # get piece west
                    new_list[k_east] # get piece east
                ]
                conflicts = []
                all_pieces = []
                for p in center_pieces:
                    for pi in puzzle.generate_rotation(p):
                    # p.rotate(random.randint(0,3)) # rotate piece at random
                        nb_conflicts = 0
                        if around[0].colors[1] != pi.colors[0]: nb_conflicts += 1 # north
                        if around[1].colors[0] != pi.colors[1]: nb_conflicts += 1 # south
                        if around[2].colors[3] != pi.colors[2]: nb_conflicts += 1 # west
                        if around[3].colors[2] != pi.colors[3]: nb_conflicts += 1 # east
                        conflicts.append(nb_conflicts)
                        all_pieces.append(pi)
                
                conflicts_sorted, pieces_sorted = zip(*sorted(zip(conflicts, all_pieces),key=lambda e: e[0]))
                # this sorts the conflicts list and sorts the list of the pieces following the same order
                minf = conflicts_sorted[0]
                maxf = conflicts_sorted[-1] # get min and max value
                RCL = [p for i,p in enumerate(pieces_sorted) if conflicts_sorted[i] <= minf + alpha * (maxf - minf)]
                # get RCL
                chosen = random.choice(RCL) # choose by random in RCL
                center_pieces.pop(center_pieces.index(chosen))
                new_list[k] = chosen
     

    return new_list

def grasp_construction(puzzle,alpha,nb_trials):
    """
    Returns the best Greedy Construction
    :param puzzle: object describing the puzzle
    :param alpha: float to determine if we keep bad solutions or not
    :param nb_trials: number of trials for greedy construction
    :return: a list of Pieces, being a solution for the puzzle    
    """
    star = None
    fstar = 1e8
    print("*"*50)
    print("Construction with Grasp")
    for _ in range(nb_trials):
        s = greedy_construction(alpha,puzzle)
        f = puzzle.get_total_n_conflict(s)
        print(f,end=" ")
        if f < fstar:
            fstar = f
            star = s
    print("best",fstar)
    return star,fstar

####################################################
# SIMULATED ANNEALING
####################################################

def fast_neighborhood_placed(s,puzzle):
    """
    Returns the neighborhood for a solution. For each corner and wall pieces, we also select a center piece.
    The neighbor will be 3 switches where all the 3 pieces takes the places of another one from the same type
    :param s: a solution of the puzzle
    :param puzzle: object describing the puzzle
    :return: a list of solutions    
    """
    neighbors = []
    list_i_corners = []
    list_i_wall = []
    list_i_center = []
    for i,p in enumerate(s):
        t = p.getType()
        if t == 'corner':
            list_i_corners.append(i)
        elif t == 'wall':
            list_i_wall.append(i)
        else:
            list_i_center.append(i)

    for ico,corner in enumerate(s):
        if corner.getType() == 'corner':
            for iwa,wall in enumerate(s):
                if wall.getType() == 'wall':
                    # for ice,center in enumerate(s):
                    #     if center.getType() == 'center':

                            ice = random.choice(list_i_center)
                            center = s[ice]
                            corner_to_switch_with = random.choice(list_i_corners) # select element random with the right type
                            wall_to_switch_with = random.choice(list_i_wall)
                            center_to_switch_with = random.choice(list_i_center)
                            rot_corner = random.choice(puzzle.generate_rotation(corner)) # get one rotation of the piece (not the random one)
                            rot_wall = random.choice(puzzle.generate_rotation(wall))
                            rot_center = random.choice(puzzle.generate_rotation(center))
                            new_s = copy.deepcopy(s)
                            new_s[ico] = new_s[corner_to_switch_with] #random element take place at ico, we swith for the same types
                            new_s[corner_to_switch_with] = rot_corner # we set the piece with rotation at the random element's place
                            new_s[iwa] = new_s[wall_to_switch_with]
                            new_s[wall_to_switch_with] = rot_wall
                            new_s[ice] = new_s[center_to_switch_with]
                            new_s[center_to_switch_with] = rot_center
                            neighbors.append(new_s)

    return neighbors

def get_one_neighbor_placed(s,puzzle):
    """
    Returns one neighbor for one solution. 3 pairs of pieces (1 per type) selected by random are switched.
    :param s: a solution of the puzzle
    :param puzzle: object describing the puzzle
    :return: one solution 
    """
    list_i_corners = []
    list_i_wall = []
    list_i_center = []
    for i,p in enumerate(s):
        t = p.getType()
        if t == 'corner':
            list_i_corners.append(i)
        elif t == 'wall':
            list_i_wall.append(i)
        else:
            list_i_center.append(i)
    
    new_s = copy.deepcopy(s)

    ico = random.choice(list_i_corners) # select first element to switch
    iwa = random.choice(list_i_wall)
    ice = random.choice(list_i_center)
    corner = new_s[ico]
    wall = new_s[iwa]
    center = new_s[ice]
    
    zeros_ico = [i for i,c in enumerate(corner.colors) if c == 0] # remember the rotation of the corner element
    zero_iwa = [i for i,c in enumerate(wall.colors) if c == 0] # remember the rotation of the second element

    ico2 = random.choice(list_i_corners) # select second element to switch with the right type
    iwa2 = random.choice(list_i_wall)
    ice2 = random.choice(list_i_center)
    corner2 = new_s[ico2]
    wall2 = new_s[iwa2]
    center2 = new_s[ice2]

    zeros_ico2 = [i for i,c in enumerate(corner2.colors) if c == 0] # remember the rotation of the corner element
    zero_iwa2 = [i for i,c in enumerate(wall2.colors) if c == 0] # remember the rotation of the second element

    rot_corner = None
    for rc in puzzle.generate_rotation(corner):
        if rc.colors[zeros_ico2[0]] == 0 and rc.colors[zeros_ico2[1]] == 0: # find the right rotation for the first element
            rot_corner = rc
            break
    rot_corner2 = None
    for rc in puzzle.generate_rotation(corner2):
        if rc.colors[zeros_ico[0]] == 0 and rc.colors[zeros_ico[1]] == 0: # find the right rotation for the second element element
            rot_corner2 = rc
            break
    # same for the 2 wall pieces
    rot_wall = None
    for rw in puzzle.generate_rotation(wall):
        if rw.colors[zero_iwa2[0]] == 0: # same
            rot_wall = rw
            break
    rot_wall2 = None
    for rw in puzzle.generate_rotation(wall2):
        if rw.colors[zero_iwa[0]] == 0: # same
            rot_wall2 = rw
            break
   

    # rotate centers to best choice
    size = puzzle.board_size
    center_north = new_s[ice+size]
    center_south = new_s[ice-size]
    center_west = new_s[ice-1]
    center_east = new_s[ice+1]
    center2_north = new_s[ice2+size]
    center2_south = new_s[ice2-size]
    center2_west = new_s[ice2-1]
    center2_east = new_s[ice2+1]

    rot_center = None
    best_conflicts = 5
    for rw in puzzle.generate_rotation(center):
        conflicts = 0
        if rw.colors[0] != center2_north.colors[1]: conflicts += 1
        if rw.colors[1] != center2_south.colors[0]: conflicts += 1
        if rw.colors[2] != center2_west.colors[3]: conflicts += 1
        if rw.colors[3] != center2_east.colors[2]: conflicts += 1
        if conflicts < best_conflicts:
            rot_center = rw
            best_conflicts = conflicts

    # same for second center
    rot_center2 = None
    best_conflicts = 5
    for rw in puzzle.generate_rotation(center2):
        conflicts = 0
        if rw.colors[0] != center_north.colors[1]: conflicts += 1
        if rw.colors[1] != center_south.colors[0]: conflicts += 1
        if rw.colors[2] != center_west.colors[3]: conflicts += 1
        if rw.colors[3] != center_east.colors[2]: conflicts += 1
        if conflicts < best_conflicts:
            rot_center2 = rw
            best_conflicts = conflicts

    # make the switch
    new_s[ico] = rot_corner2
    new_s[ico2] = rot_corner
    new_s[iwa] = rot_wall2
    new_s[iwa2] = rot_wall
    new_s[ice] = rot_center2
    new_s[ice2] = rot_center
                
    return new_s

def simulated_annealing(puzzle):

    print("="*30)
    print("USING SIMULATED ANNEALING WITH GRASP")
    print("="*30)

    best_s = None
    best_f = 1e10

    #########################################
    # HYPER PARAMETERS
    nb_restart = 3
    T = 0.05
    maxT = 10
    alphaT = 0.75
    betaT = 5
    re_lim = 1500
    fast_neighbor = True
    alpha_grasp = 0.2
    nb_tries_grasp = 20
    #########################################
    nb_minutes = 3

    total_time = time.time() # get time, algorithm can run during 10 minutes

    for _ in range(nb_restart):
        print("NEW START")
        start_time = time.time()
        # s,fs = solve_best_random(puzzle,20)
        s,fs = grasp_construction(puzzle,alpha_grasp,nb_tries_grasp)
        star = s
        fstar = fs

        if fs == 0:
            print("Optimal solution found")
            return s,fs

        re_count = 0
        T = maxT

        change = True
        i = -1
        while time.time() - start_time < nb_minutes * 60 / nb_restart:
            i += 1
            if i % 500 == 0:
                print(f"{i} current best: {fs}")
            if re_count >= re_lim:
                # G = fast_neighborhood_placed(star,puzzle) if fast_neighbor else neighborhood_placed(star,puzzle)
                s = star
                re_count = 0 # reset counter
                T = min(T + betaT, maxT)
            # if change:
            #     G = fast_neighborhood_placed(s,puzzle) if fast_neighbor else neighborhood_placed(s,puzzle)
            #     change = False
            # c = G[random.randint(0,len(G)-1)]
            c = get_one_neighbor_placed(s,puzzle)
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
                    if fstar == 0:
                        print("Optimal solution found")
                        return s,fs
            else:
                re_count += 1
            T = alphaT * T

        if fstar < best_f:
            print("NEW BEST",fstar)
            best_f = fstar
            best_s = star
        print(f"Time taken: {(time.time() - start_time).__round__()}s")
    
    print(f"Total time: {(time.time() - total_time).__round__()}s")
    
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
    child1 = parent1
    child2 = parent2
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
    size_population = 20
    assert size_population % 2 == 0, "Size population should be an even number"
    mutation_probability = 0.6
    ###############################
    print("Size population:",size_population)
    print("Mutation probability:",mutation_probability)
    print("="*30)

    start_time = time.time()

    print("Contruction of initial population")
    # initialize population
    population = []
    for _ in range(size_population):
        population.append(solve_best_random(puzzle,20)[0])
    
    population = sorted(population,key=lambda s: puzzle.get_total_n_conflict(s))

    nb_minutes = 1
    k = -1
    print("="*30)
    print("Start of Algorithm")
    best_s = None
    best_f = 1e8
    while time.time() - start_time < nb_minutes * 60:
        k += 1
        # classify from best to worst our population
        if k % 50 == 0:
            print(k,"Best solution",puzzle.get_total_n_conflict(population[0]))

        new_children = []
        for i in range(0,len(population),2):
            parent1,parent2 = population[i],population[i+1] # get two parents
            child1,child2 = children(copy.deepcopy(parent1),copy.deepcopy(parent2),piece_list) # get 2 childs
            if random.random() < mutation_probability: # mutate
                mutation(child1)
            if random.random() < mutation_probability: # mutate
                mutation(child2)
            new_children.append(child1)
            new_children.append(child2)
        
        population = new_children
        population = sorted(population,key=lambda s: puzzle.get_total_n_conflict(s))
        s = population[0]
        f = puzzle.get_total_n_conflict(s)
        if f < best_f:
            best_f = f
            best_s = s

    return best_s, best_f

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
