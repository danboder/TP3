import pygment
import argparse
import solver


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--infile', type=str, default='instances/instanceB')
    parser.add_argument('--outfile', type=str, default='output')
    parser.add_argument('--visufile', type=str, default='sol.png')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] Start the solving of the VRP instance")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visu file: %s" % args.visufile)
    print("***********************************************************")

    e = pygment.Pygment(args.infile)

    # solution_greedy, cost_sol = solver.solve_greedy(e)
    solution_greedy, cost_sol = solver.solve_advance(e)
    print(solution_greedy)
    print(cost_sol)

    e.display_solution(solution_greedy, args.visufile)
    e.print_solution(solution_greedy, args.outfile)

    print(e.verify_solution(solution_greedy,True))
