import random
from collections import deque
from pathlib import Path

NUM_WALKS = 10000


def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def is_connected(my_map, start, goal):
    open_list = deque()
    closed_list = set()
    open_list.append(start)
    closed_list.add(start)
    while len(open_list) > 0:
        curr = open_list.popleft()
        if curr == goal:
            return True
        for i in range(4):
            next = move(curr, i)
            if my_map[next[0]][next[1]] or next in closed_list:
                continue
            open_list.append(next)
            closed_list.add(next)
    return False


def gen_map(old_size, obstacle):
    size = old_size[0] + 2
    glo_map = [[False for i in range(size)] for j in range(size)]
    for i in range(size):
        glo_map[0][i] = True
        glo_map[i][0] = True
        glo_map[size - 1][i] = True
        glo_map[i][size - 1] = True
    for i in range(len(obstacle)):
        glo_map[int(obstacle[i][0]) + 1][int(obstacle[i][1]) + 1] = True
    return glo_map


def gen_random_starts_and_goals(my_map, num_of_agents, random_walk_steps=NUM_WALKS):
    """Generate random start and goal locations by random walk.
    my_map          - binary obstacle maps
    num_of_agents   - number of agents
    """

    size_x = len(my_map)
    size_y = len(my_map[1])
    # Generate the initial positions of the robots
    starts = []
    goals = []
    used4starts = [[False for _ in range(size_y)] for _ in range(size_x)]
    used4goals = [[False for _ in range(size_y)] for _ in range(size_x)]
    while len(starts) < num_of_agents:
        # Generate possible initial and goal positions
        start = (random.randint(0, size_x - 1), random.randint(0, size_y - 1))
        if my_map[start[0]][start[1]] or used4starts[start[0]][start[1]]:
            continue
        curr = start
        i = 0
        while i < random_walk_steps or used4goals[curr[0]][curr[1]]:
            r = random.randint(0, 3)
            next = move(curr, r)
            if my_map[next[0]][next[1]] is False:
                curr = next
                i += 1
        goal = curr
        starts.append(start)
        used4starts[start[0]][start[1]] = True
        goals.append(goal)
        used4goals[goal[0]][goal[1]] = True

    return starts, goals


def reflush_data(starts, goals):
    new_starts = []
    new_goals = []
    for i in range(len(starts)):
        new_starts.append((starts[i][0] + 1, starts[i][1] + 1))
    for i in range(len(goals)):
        new_goals.append((goals[i][0] + 1, goals[i][1] + 1))
    return new_starts, new_goals


def reflush_solution(solution):
    new_solution = []
    if solution != -1:
        for i in range(len(solution)):
            temp = []
            for j in range(len(solution[i])):
                temp.append((solution[i][j][0]-1, solution[i][j][1]-1))
            new_solution.append(temp)
    return new_solution
