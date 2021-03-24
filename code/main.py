import eternity_puzzle
import argparse
import copy
import solver
import time



def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    # parser.add_argument('--infile', type=str, default='instances/eternity_trivialA.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_A.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_B.txt')
    parser.add_argument('--infile', type=str, default='instances/eternity_C.txt')
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

    e = eternity_puzzle.EternityPuzzle(args.infile)
    start_time = time.time()
    # solution, cost_sol = solver.solve_random(e)
    # solution, cost_sol = solver.solve_best_random(e, 10)
    # solution, cost_sol = solver.solve_advance(e)
    execution_time = 5
    best_cost = 100000
    trial = 0
    while time.time() - start_time < execution_time * 60:
        print("Trial", trial)
        solution, cost_sol = solver.solve_GRASP(e, 26)
        solution, cost_sol = solver.simulated_annealing(solution, cost_sol, e)
        if cost_sol < best_cost:
            print("NEW BEST", cost_sol)
            best_cost = cost_sol
            best_solution = copy.deepcopy(solution)
        trial += 1

    # solution, cost_sol = solver.solve_greedyRandom(1, e)
    end_time = time.time()

    print(best_solution)
    print(best_cost)
    print('Time elapsed: ', end_time - start_time)

    e.display_solution(best_solution, args.visufile)
    e.print_solution(best_solution, args.outfile)

    print(e.verify_solution(best_solution))