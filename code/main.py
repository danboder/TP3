import eternity_puzzle
import argparse
import copy
import solver
import time


def grasp(exec, e, alpha):
    best_cost = 100000
    trial = 0
    while time.time() - start_time < exec * 60:
        print("Trial", trial)
        cur_solution, cur_cost_sol = solver.solve_GRASP(e, alpha)
        if cur_cost_sol < best_cost:
            print("NEW BEST", cur_cost_sol)
            best_cost = cur_cost_sol
            best_solution = copy.deepcopy(cur_solution)
        trial += 1
    return best_solution, best_cost


def grasp_and_sa(exec, e, alpha):
    best_cost = 100000
    trial = 0
    while time.time() - start_time < exec * 60:
        print("Trial", trial)
        cur_solution, cur_cost_sol = solver.solve_GRASP(e, alpha)
        cur_solution, cur_cost_sol = solver.simulated_annealing(cur_solution, cur_cost_sol, e)
        if cur_cost_sol < best_cost:
            print("NEW BEST", cur_cost_sol)
            best_cost = cur_cost_sol
            best_solution = copy.deepcopy(cur_solution)
        trial += 1

    return best_solution, best_cost


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    # parser.add_argument('--infile', type=str, default='instances/eternity_trivialA.txt')
    parser.add_argument('--infile', type=str, default='instances/eternity_A.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_B.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_C.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_Complete.txt')
    parser.add_argument('--outfile', type=str, default='output')
    parser.add_argument('--visufile', type=str, default='sol.png')
    # parser.add_argument('--greedytriv', type=str, default='soltriv.png')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] Start the solving of the VRP instance")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visu file: %s" % args.visufile)
    print("***********************************************************")
    alpha = 26
    execution_time = 10

    e = eternity_puzzle.EternityPuzzle(args.infile)
    start_time = time.time()
    # solution, cost_sol = solver.solve_random(e)
    # solution, cost_sol = solver.solve_best_random(e, 10)
    # solution, cost_sol = solver.solve_advance(e)
    # solution, cost_sol = solver.solve_greedyRandom(1, e)
    solution, cost_sol = grasp(execution_time, e, alpha)
    # solution, cost_sol = grasp_and_sa(execution_time, e, alpha)
    end_time = time.time()

    print(solution)
    print(cost_sol)
    print('Time elapsed: ', end_time - start_time)

    e.display_solution(solution, args.visufile)
    e.print_solution(solution, args.outfile)

    print(e.verify_solution(solution))



