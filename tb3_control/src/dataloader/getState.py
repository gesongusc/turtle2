import numpy as np
import math
import torch
from operator import itemgetter


def getAgentPos(num_agents, agent_pos, step):
    current_agent_pos = np.zeros((num_agents, 2))
    for i in range(num_agents):
        current_agent_pos[i][0] = agent_pos[step][i][0]
        current_agent_pos[i][1] = agent_pos[step][i][1]
    return current_agent_pos


def setSpecialFeatures(var):
    var[0] = var[1] = -1


def varNormalization(toVar, var, currentPos, range):
    toVar[0] = var[0]
    toVar[1] = var[1]


def getDistance(var1, var2):
    return math.sqrt((var1[0] - var2[0]) ** 2 + (var1[1] - var2[1]) ** 2)


def checkInFOV(var, center, R):
    if abs(var[0] - center[0]) <= R and abs(var[1] - center[1]) <= R:
        return True
    else:
        return False


def getState(makespan, num_agents, nearest_step_num, nearest_robot_num, agent_pos, R, final_goal, current_step):
    feat_own_path = np.zeros((makespan, num_agents, nearest_step_num, 2))

    feat_agent_path = np.zeros((makespan, num_agents, nearest_robot_num, nearest_step_num, 2))

    feat_final_goal = np.zeros((makespan, num_agents, nearest_robot_num, 2))

    for step in range(makespan):
        current_agent_pos = getAgentPos(num_agents, agent_pos, step)
        for agent in range(num_agents):

            if makespan == 1:
                ps = current_step
            else:
                ps = step

            for i in range(nearest_step_num):
                if ps >= 0:
                    varNormalization(feat_own_path[step][agent][i], agent_pos[ps][agent], current_agent_pos[agent], R)
                    ps = ps - 1
                else:  
                    setSpecialFeatures(feat_own_path[step][agent][i])


            sorted_data = []
            for agent_i in range(len(current_agent_pos)):

                if agent_i == agent:
                    continue

                if checkInFOV(current_agent_pos[agent_i], current_agent_pos[agent], R) and current_agent_pos[agent_i][0] != final_goal[agent_i][0] \
                        and current_agent_pos[agent_i][1] != final_goal[agent_i][1]:
                    dis = getDistance(current_agent_pos[agent_i], current_agent_pos[agent])
                else:
                    dis = 999
                sorted_data.append([dis, agent_i])
            sorted_data.sort(key=itemgetter(0, 1))

            for other_robot in range(nearest_robot_num):
                if sorted_data[other_robot][0] == 999:
                    for i in range(nearest_step_num):
                        setSpecialFeatures(feat_agent_path[step][agent][other_robot][i])
                    setSpecialFeatures(feat_final_goal[step][agent][other_robot])
                else:
                    selected_id = sorted_data[other_robot][1]
                    if makespan == 1:
                        ps = current_step
                    else:
                        ps = step
                    for i in range(nearest_step_num):
                        if ps >= 0:
                            if not checkInFOV(agent_pos[ps][selected_id], current_agent_pos[agent], R):
                                setSpecialFeatures(feat_agent_path[step][agent][other_robot][i])
                            else:
                                varNormalization(feat_agent_path[step][agent][other_robot][i], agent_pos[ps][selected_id], current_agent_pos[agent], R)
                            ps = ps - 1
                        else:  
                            setSpecialFeatures(feat_agent_path[step][agent][other_robot][i])
                    new_dis = getDistance(final_goal[selected_id], current_agent_pos[agent])
                    varNormalization(feat_final_goal[step][agent][other_robot], final_goal[selected_id], current_agent_pos[agent], new_dis)

    if makespan == 1:
        feat_own_path = feat_own_path.squeeze(0)
        feat_agent_path = feat_agent_path.squeeze(0)
        feat_final_goal = feat_final_goal.squeeze(0)

    feat_own_path_list = torch.FloatTensor(feat_own_path).cpu().detach().numpy()
    feat_agent_path_list = torch.FloatTensor(feat_agent_path).cpu().detach().numpy()
    feat_final_goal_list = torch.FloatTensor(feat_final_goal).cpu().detach().numpy()
    return feat_own_path_list, feat_agent_path_list, feat_final_goal_list
