import argparse
from instance_generator import *
from cbs import CBSSolver

from visualize import Animation

NUM_OBS = 10
NUM_AGENTS = 10
WIDTH = 10
TIME = 60
SOLVER = "CBS"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs test suite for cbs')
    parser.add_argument('--agents', type=int, default=NUM_AGENTS,
                        help='The number of agents, defaults to ' + str(NUM_AGENTS))
    parser.add_argument('--obs', type=int, default=NUM_OBS,
                        help='The number of obstacles on the grid, defaults to ' + str(NUM_OBS))
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--time', type=int, default=TIME,
                        help='The time limit of the solver (seconds), defaults to ' + str(TIME))
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Sequential}), defaults to ' + str(SOLVER))

    args = parser.parse_args()

    my_map = gen_map(size=3, obstacle=[(0, 0), (0, 1), (1,1),(2, 0), (2, 2)])
    starts = [(1, 0)]
    goals = [(1, 2)]
    starts, goals = reflush_data(starts, goals)

    cbs = CBSSolver(my_map, starts, goals)
    paths = cbs.find_solution(args.disjoint, args.time)


    print(paths)
    # print("***Test paths on a simulation***")
    # animation = Animation(my_map, starts, goals, paths)
    # # animation.save("output.mp4", 1.0)
    # animation.show()
