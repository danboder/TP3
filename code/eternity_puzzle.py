import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
import numpy as np

import math
import copy

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

N = 0 # north
S = 1 # south
W = 2 # west
E = 3 # est


class Piece:
    def __init__(self,id,colors):
        self.id = id
        self.colors = colors
    def __str__(self):
        return f"{self.id} {self.colors}"
    def __repr__(self):
        return self.__str__()
    
    def isWall(self):
        return 0 in self.colors
    def isCorner(self):
        c = self.colors
        return (c[N] == 0 and c[E] == 0) or (c[E] == 0 and c[S] == 0) or (c[S] == 0 and c[W] == 0) or (c[W] == 0 and c[N] == 0)
    def getType(self):
        if self.isCorner(): return 'corner'
        elif self.isWall(): return 'wall'
        else: return 'center'
    
    def rotate(self,to):
        # to : value between 0 and 3 = describes which rotation we do
        c = self.colors
        rotation_0 = c
        rotation_90 = (c[2], c[3], c[1], c[0])
        rotation_180 = (c[1], c[0], c[3], c[2])
        rotation_270 = (c[3], c[2], c[0], c[1])
        rots = [rotation_0, rotation_90, rotation_180, rotation_270]
        self.colors = rots[to]
        

class EternityPuzzle:

    def __init__(self, instance_file):

        with open(instance_file) as file:
            lines = file.readlines()

            self.board_size = int(lines[0])
            self.n_piece = self.board_size ** 2
            self.n_internal_connection = 2 * self.board_size * (self.board_size - 1)
            self.n_total_connection = self.n_internal_connection + self.board_size * 4

            flatten = lambda l: [item for sublist in l for item in sublist]

            self.piece_list = [Piece(i,(int(x.split()[0]), int(x.split()[1]), int(x.split()[2]), int(x.split()[3]))) for i,line in
                               enumerate(lines[1:]) for x in line.strip().split('\n')]

            self.n_color = max(flatten(list(map(lambda p: p.colors, self.piece_list)))) + 1

            assert (len(self.piece_list) == self.n_piece)

            for p in self.piece_list:
                assert (len(p.colors) == 4)

    def generate_rotation(self, piece):
        c = piece.colors
        initial_shape = c
        rotation_90 = (c[2], c[3], c[1], c[0])
        rotation_180 = (c[1], c[0], c[3], c[2])
        rotation_270 = (c[3], c[2], c[0], c[1])

        return [Piece(piece.id,initial_shape), Piece(piece.id,rotation_90), Piece(piece.id,rotation_180), Piece(piece.id,rotation_270)]
    
    def generate_rotation_colors(self, piece):
        c = piece.colors
        initial_shape = c
        rotation_90 = (c[2], c[3], c[1], c[0])
        rotation_180 = (c[1], c[0], c[3], c[2])
        rotation_270 = (c[3], c[2], c[0], c[1])

        return [initial_shape, rotation_90, rotation_180, rotation_270]
    

    def get_total_n_conflict(self, solution):

        n_conflict = 0

        for j in range(self.board_size):
            for i in range(self.board_size):

                k = self.board_size * j + i
                k_east = self.board_size * j + (i - 1)
                k_south = self.board_size * (j - 1) + i

                if i > 0 and solution[k].colors[W] != solution[k_east].colors[E]:
                    n_conflict += 1

                if i == 0 and solution[k].colors[W] != GRAY:
                    n_conflict += 1

                if i == self.board_size - 1 and solution[k].colors[E] != GRAY:
                    n_conflict += 1

                if j > 0 and solution[k].colors[S] != solution[k_south].colors[N]:
                    n_conflict += 1

                if j == 0 and solution[k].colors[S] != GRAY:
                    n_conflict += 1

                if j == self.board_size - 1 and solution[k].colors[N] != GRAY:
                    n_conflict += 1

        return n_conflict

    def display_solution(self, solution, output_file):

        if len(solution) < self.n_piece:
            solution = solution + [(WHITE, WHITE, WHITE, WHITE)] * (self.n_piece - len(solution))

        origin = 0
        size = self.board_size + 2

        color_dict = self.build_color_dict()

        fig, ax = plt.subplots()

        n_total_conflict = self.get_total_n_conflict(solution)

        n_internal_conflict = 0

        for j in range(size):  # y-axis
            for i in range(size):  # x-axis
                valid_draw = [0, size - 1]
                if i in valid_draw or j in valid_draw:
                    ax.add_patch(patches.Rectangle((i, j), i + 1, j + 1, fill=True, facecolor=color_dict[GRAY],
                                                   edgecolor=color_dict[BLACK]))
                else:
                    # ax.add_patch(patches.Rectangle((i, j), i + 1, j + 1, fill=True, facecolor='white', edgecolor='k'))

                    left_bot = (i, j)
                    right_bot = (i + 1, j)
                    right_top = (i + 1, j + 1)
                    left_top = (i, j + 1)
                    middle = (i + 0.5, j + 0.5)

                    instructions = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

                    triangle_south_path = Path([left_bot, middle, right_bot, left_bot], instructions)
                    triangle_east_path = Path([right_top, middle, right_bot, right_top], instructions)
                    triangle_north_path = Path([right_top, middle, left_top, right_top], instructions)
                    triangle_west_path = Path([left_bot, middle, left_top, left_bot], instructions)

                    is_triangle_south_valid = True
                    is_triangle_north_valid = True
                    is_triangle_east_valid = True
                    is_triangle_west_valid = True

                    k = self.board_size * (j - 1) + (i - 1)
                    k_east = self.board_size * (j - 1) + (i - 2)
                    k_south = self.board_size * (j - 2) + (i - 1)

                    if i == 1:
                        is_triangle_west_valid = (solution[k].colors[W] == GRAY)  # 1 for Gray
                    elif i == size - 2:
                        is_triangle_east_valid = (solution[k].colors[E] == GRAY)
                        is_triangle_west_valid = solution[k].colors[W] == solution[k_east].colors[E]
                    else:
                        is_triangle_west_valid = solution[k].colors[W] == solution[k_east].colors[E]

                    if j == 1:
                        is_triangle_south_valid = (solution[k].colors[S] == GRAY)
                    elif j == size - 2:
                        is_triangle_north_valid = (solution[k].colors[N] == GRAY)
                        is_triangle_south_valid = solution[k].colors[S] == solution[k_south].colors[N]
                    else:
                        is_triangle_south_valid = solution[k].colors[S] == solution[k_south].colors[N]

                    patch_south = patches.PathPatch(triangle_south_path, facecolor=color_dict[solution[k].colors[S]],
                                                    edgecolor=color_dict[BLACK])

                    patch_north = patches.PathPatch(triangle_north_path, facecolor=color_dict[solution[k].colors[N]],
                                                    edgecolor=color_dict[BLACK])

                    patch_east = patches.PathPatch(triangle_east_path, facecolor=color_dict[solution[k].colors[E]],
                                                   edgecolor=color_dict[BLACK])

                    patch_west = patches.PathPatch(triangle_west_path, facecolor=color_dict[solution[k].colors[W]],
                                                   edgecolor=color_dict[BLACK])

                    if not is_triangle_south_valid:
                        line_zip = list(zip(left_bot, right_bot))
                        line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                        ax.add_line(line)

                        if j != 1:
                            n_internal_conflict += 1

                    if not is_triangle_north_valid:
                        line_zip = list(zip(left_top, right_top))
                        line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                        ax.add_line(line)

                        if j != size - 2:
                            n_internal_conflict += 1

                    if not is_triangle_west_valid:
                        line_zip = list(zip(left_bot, left_top))
                        line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                        ax.add_line(line)

                        if i != 1:
                            n_internal_conflict += 1

                    if not is_triangle_east_valid:
                        line_zip = list(zip(right_bot, right_top))
                        line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                        ax.add_line(line)

                        if i != size - 2:
                            n_internal_conflict += 1

                    ax.add_patch(patch_south)
                    ax.add_patch(patch_north)
                    ax.add_patch(patch_east)
                    ax.add_patch(patch_west)

                    k += 1

        plt.xlim(origin, size)
        plt.ylim(origin, size)

        title = 'Eternity of size %d X %d\n' \
                'Total connections: %d    Internal connections: %d\n' \
                'Total Valid connections: %d     Internal valid internal connections: %d\n' \
                'Total Invalid connections: %d    Internal invalid connections: %d' % \
                (self.board_size, self.board_size,
                 self.n_total_connection, self.n_internal_connection,
                 self.n_total_connection - n_total_conflict, self.n_internal_connection - n_internal_conflict,
                 n_total_conflict, n_internal_conflict,
                 )
        ax.set_title(title)

        plt.savefig(output_file)

    def print_solution(self, solution, output_file):
        with open(output_file, "w") as file:
            file.write(str(self.get_total_n_conflict(solution)) + "\n")
            file.write(str(self.board_size))
            for piece in solution:
                file.write("\n")
                for c in piece.colors:
                    file.write(str(c) + " ")

    def build_color_dict(self):

        color_dict = {
            GRAY: 'gray',
            1: 'lightcoral',
            2: 'tab:blue',
            3: 'tab:orange',
            4: 'tab:green',
            5: 'gold',
            6: 'tab:purple',
            7: 'tab:brown',
            8: 'tab:pink',
            9: 'tab:olive',
            10: 'tab:cyan',
            11: 'deeppink',
            12: 'blue',
            13: 'slateblue',
            14: 'darkslateblue',
            15: 'darkviolet',
            16: 'teal',
            17: 'wheat',
            18: 'darkkhaki',
            19: 'indigo',
            20: 'fuchsia',
            21: 'lime',
            22: 'rosybrown',
            BLACK: 'black',
            RED: 'tab:red',
            WHITE: 'white'
        }
        return color_dict

    def hash_piece(self, piece):
        all = self.generate_rotation_colors(piece)
        return min(all)

    def verify_solution(self,solution):
        hash_init = sorted([self.hash_piece(p) for p in self.piece_list])
        hash_sol = sorted([self.hash_piece(p) for p in solution])

        return hash_init == hash_sol

