import eternity_puzzle
import argparse
import solver


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    # parser.add_argument('--infile', type=str, default='instances/eternity_trivialA.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_A.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_B.txt')
    # parser.add_argument('--infile', type=str, default='instances/eternity_C.txt')
    parser.add_argument('--infile', type=str, default='instances/eternity_Complete.txt')
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

    e = eternity_puzzle.EternityPuzzle(args.infile)
    print(e.piece_list)

    # solution, cost_sol = solver.solve_best_random(e, 10)
    solution, cost_sol = solver.solve_advance(e)

    print(solution)
    print(args.infile,cost_sol)
    e.display_solution(solution,args.visufile)
    e.print_solution(solution, args.outfile)

    print(e.verify_solution(solution))
