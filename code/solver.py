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

        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]

        solution.append(piece_permuted)

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

def simulated_annealing(puzzle):

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
        print("START")
        s,fs = solve_best_random(puzzle,20)
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
                    print("improvement at",i,":",fstar)
            else:
                re_count += 1
            T = alphaT * T

        if fstar < best_f:
            print("NEW BEST",fstar)
            best_f = fstar
            best_s = star
    
    return best_s,best_f





def solve_advance(eternity_puzzle):
    """
    Advanced solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    # TODO implement here your solution

    solution = simulated_annealing(eternity_puzzle)

    return solution
