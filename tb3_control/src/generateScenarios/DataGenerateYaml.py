import csv
import os
import sys
import shutil
import numpy as np
import scipy.io as sio
import yaml
import argparse
from os.path import dirname, realpath, pardir
from hashids import Hashids
import hashlib

sys.path.append(os.path.join(dirname(realpath(__file__)), pardir))

import utils.graphUtils.graphTools as graph

from dataloader.statetransformer_Guidance import AgentState
from scipy.spatial.distance import squareform, pdist
from multiprocessing import Queue

parser = argparse.ArgumentParser("Input width and #Agent")
parser.add_argument('--num_agents', type=int, default=4)
parser.add_argument('--map_w', type=int, default=10)
parser.add_argument('--map_density', type=float, default=0.1)
parser.add_argument('--dir_SaveData', type=str, default='../MultiAgentDataset/DataSource_DMap_LG')
parser.add_argument('--loadmap_TYPE', type=str, default='map')
parser.add_argument('--solCases_dir', type=str, default='../MultiAgentDataset/Solution_DMap')
parser.add_argument('--chosen_solver', type=str, default='ECBS')

parser.add_argument('--id_start', type=int, default=0)
parser.add_argument('--div_train', type=int, default=21000)
parser.add_argument('--div_valid', type=int, default=200)
parser.add_argument('--div_test', type=int, default=4500)

parser.add_argument('--div_train_IDMap', type=int, default=0)
parser.add_argument('--div_test_IDMap', type=int, default=427)
parser.add_argument('--div_valid_IDMap', type=int, default=800)
parser.add_argument('--maxNum_Map', type=int, default=1000)

parser.add_argument('--div_extend_valid', type=int, default=0)

parser.add_argument('--FOV', type=int, default=9)
parser.add_argument('--guidance', type=str, default='')
parser.add_argument('--dynamic_commR', action='store_true', default=False)
parser.add_argument('--symmetric_norm', action='store_true', default=False)
parser.add_argument('--commR', type=int, default=7)

args = parser.parse_args()


class DataTransformer:
    def __init__(self, config):
        self.config = config
        self.PROCESS_NUMBER = 4
        self.num_agents = self.config.num_agents
        self.size_map = [self.config.map_w, self.config.map_w]
        self.label_density = str(self.config.map_density).split('.')[-1]
        self.AgentState = AgentState(self.config)
        self.communicationRadius = self.config.commR  # 通信半径
        self.zeroTolerance = 1e-9
        self.delta = [[-1, 0],  # go up
                      [0, -1],  # go left
                      [1, 0],  # go down
                      [0, 1],  # go right
                      [0, 0]]  # stop
        self.num_actions = 5
        self.list_seqtrain_file = []
        self.list_train_file = []
        self.list_seqvalid_file = []
        self.list_validStep_file = []
        self.list_valid_file = []
        self.list_test_file = []
        self.hashids = Hashids(alphabet='01234567789abcdef', min_length=5)
        self.pathtransformer = self.pathtransformer_RelativeCoordinate
        self.label_setup = '{}{:02d}x{:02d}_density_p{}/{}_Agent'.format(self.config.loadmap_TYPE, self.size_map[0],
                                                                         self.size_map[1],
                                                                         self.label_density,
                                                                         self.num_agents)
        self.dirName_parent = os.path.join(self.config.solCases_dir, self.label_setup)
        self.dirName_Store = os.path.join(self.config.dir_SaveData, self.label_setup)
        self.dirName_input = os.path.join(self.dirName_parent, 'input')
        self.dirName_output = os.path.join(self.dirName_parent, 'output_{}'.format(config.chosen_solver))
        self.set_up()

        if self.config.dynamic_commR:
            # comm radius that ensure initial graph connected
            print("run on multirobotsim (radius dynamic) with collision shielding")
            self.getAdjacencyMatrix = self.computeAdjacencyMatrix
        else:
            # comm radius fixed
            print("run on multirobotsim (radius fixed) with collision shielding")
            self.getAdjacencyMatrix = self.computeAdjacencyMatrix_fixedCommRadius

    def set_up(self):

        self.list_failureCases_solution = self.search_failureCases(self.dirName_output)
        self.list_failureCases_input = self.search_failureCases(self.dirName_input)
        self.nameprefix_input = self.list_failureCases_input[0].split('input/')[-1].split('ID')[0]
        self.list_failureCases_solution = sorted(self.list_failureCases_solution)

        self.list_sol_training = []
        self.list_sol_valid = []
        self.list_sol_test = []

        for i in range(self.config.div_train_IDMap, self.config.div_test_IDMap):
            for case in self.list_failureCases_solution:
                if "_IDMap{:05d}".format(i) in case:
                    self.list_sol_training.append(case)

        for i in range(self.config.div_test_IDMap, self.config.div_valid_IDMap):
            for case in self.list_failureCases_solution:
                if "_IDMap{:05d}".format(i) in case:
                    self.list_sol_test.append(case)

        for i in range(self.config.div_valid_IDMap, self.config.maxNum_Map):
            for case in self.list_failureCases_solution:
                if "_IDMap{:05d}".format(i) in case:
                    self.list_sol_valid.append(case)

        self.list_sol_training = sorted(self.list_sol_training)
        self.list_sol_valid = sorted(self.list_sol_valid)
        self.list_sol_valid = sorted(self.list_sol_valid)
        self.len_failureCases_solution = len(self.list_failureCases_solution)

    def reset(self):
        self.task_queue = Queue()
        dirpath = self.dirName_Store
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        self.path_save_solDATA = self.dirName_Store

        try:
            os.makedirs(self.path_save_solDATA)
            os.makedirs(os.path.join(self.path_save_solDATA, 'train'))
            os.makedirs(os.path.join(self.path_save_solDATA, 'valid'))
            os.makedirs(os.path.join(self.path_save_solDATA, 'test'))
        except FileExistsError:
            pass

    def solutionTransformer(self):
        div_train = self.config.div_train
        div_valid = self.config.div_valid
        div_test = self.config.div_test
        if self.config.div_extend_valid != 0:
            num_used_data = div_train + div_valid + div_test + self.config.div_extend_valid
        else:
            num_used_data = div_train + div_valid + div_test

        num_data_loop = min(num_used_data, self.len_failureCases_solution)

        for id_sol in range(0, div_train):
            mode = "train"
            case_config = (mode, id_sol)
            self.task_queue.put(case_config)

        for id_sol in range(0, div_valid):
            mode = "valid"
            case_config = (mode, id_sol)
            self.task_queue.put(case_config)

        for id_sol in range(0, div_test):
            mode = "test"
            case_config = (mode, id_sol)
            self.task_queue.put(case_config)

        while not self.task_queue.empty():
            try:
                case_config = self.task_queue.get(block=False)
                (mode, id_sol) = case_config
                self.pipeline(case_config)
            except Exception as e:
                print(e)
                return



    def pipeline(self, case_config):
        (mode, id_sol) = case_config
        agents_schedule, agents_goal, makespan, map_data, id_case = self.load_ExpertSolution(mode, id_sol)
        log_str = 'Transform_failureCases_ID_#{} in MAP_ID{}'.format(id_case[1], id_case[0])
        print('############## {} ###############'.format(log_str))
        if mode == "train" or mode == "valid":
            self.pathtransformer(map_data, agents_schedule, agents_goal, makespan + 1, id_case, mode)
        else:
            self.pathtransformer_test(map_data, agents_schedule, agents_goal, makespan + 1, id_case, mode)

    def load_ExpertSolution(self, mode, ID_case):
        if mode == 'train':
            name_solution_file = self.list_sol_training[ID_case]
        elif mode == 'valid':
            name_solution_file = self.list_sol_valid[ID_case]
        elif mode == 'test':
            name_solution_file = self.list_sol_test[ID_case]

        # id_solved_case = name_solution_file.split('_ID')[-1].split('.yaml')[0]
        map_setup = name_solution_file.split('output_')[-1].split('_IDMap')[0]
        id_sol_map = name_solution_file.split('_IDMap')[-1].split('_IDCase')[0]
        id_sol_case = name_solution_file.split('_IDCase')[-1].split('_')[0]

        name_inputfile = os.path.join(self.dirName_input,
                                      'input_{}_IDMap{}_IDCase{}.yaml'.format(map_setup, id_sol_map, id_sol_case))

        print(name_inputfile)
        print(name_solution_file)

        with open(name_inputfile, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(name_solution_file, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        agentsConfig = data_config['agents']
        num_agent = len(agentsConfig)
        list_posObstacle = data_config['map']['obstacles']

        if list_posObstacle == None:
            map_data = np.zeros(self.size_map, dtype=np.int64)
        else:
            map_data = self.setup_map(list_posObstacle)

        schedule = data_output['schedule']
        makespan = data_output['statistics']['makespan']

        goal_allagents = np.zeros([num_agent, 2])
        schedule_agentsState = np.zeros([makespan + 1, num_agent, 2])
        schedule_agentsActions = np.zeros([makespan + 1, num_agent, self.num_actions])
        schedule_agents = [schedule_agentsState, schedule_agentsActions]
        hash_ids = np.zeros(self.num_agents)
        for id_agent in range(num_agent):
            goalX = agentsConfig[id_agent]['goal'][0]
            goalY = agentsConfig[id_agent]['goal'][1]
            goal_allagents[id_agent][:] = [goalX, goalY]

            schedule_agents = self.obtainSchedule(id_agent, schedule, schedule_agents, goal_allagents, makespan + 1)

            str_id = '{}_{}_{}'.format(id_sol_map, id_sol_case, id_agent)
            int_id = int(hashlib.sha256(str_id.encode('utf-8')).hexdigest(), 16) % (10 ** 5)
            # hash_ids[id_agent]=np.divide(int_id,10**5)
            hash_ids[id_agent] = int_id

        # print(id_sol_map, id_sol_case, hash_ids)
        return schedule_agents, goal_allagents, makespan, map_data, (id_sol_map, id_sol_case, hash_ids)

    def load_ExpertSolution_(self, ID_case):

        name_solution_file = self.list_failureCases_solution[ID_case]
        id_sol_case = name_solution_file.split('_ID')[-1].split('.yaml')[0]

        map_setup = 'demo'
        id_sol_map = '0'

        name_inputfile = os.path.join(self.dirName_input,
                                      'failureCases_ID{}.yaml'.format(id_sol_case))

        # print(name_inputfile)
        # print(name_solution_file)

        with open(name_inputfile, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(name_solution_file, 'r') as stream:
            try:
                # print(yaml.safe_load(stream))
                data_output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        agentsConfig = data_config['agents']
        num_agent = len(agentsConfig)
        list_posObstacle = data_config['map']['obstacles']

        if list_posObstacle == None:
            map_data = np.zeros(self.size_map, dtype=np.int64)
        else:
            map_data = self.setup_map(list_posObstacle)

        schedule = data_output['schedule']
        makespan = data_output['statistics']['makespan']

        # print(schedule)
        goal_allagents = np.zeros([num_agent, 2])
        schedule_agentsState = np.zeros([makespan + 1, num_agent, 2])
        schedule_agentsActions = np.zeros([makespan + 1, num_agent, self.num_actions])
        schedule_agents = [schedule_agentsState, schedule_agentsActions]
        hash_ids = np.zeros(self.num_agents)
        for id_agent in range(num_agent):
            goalX = agentsConfig[id_agent]['goal'][0]
            goalY = agentsConfig[id_agent]['goal'][1]
            goal_allagents[id_agent][:] = [goalX, goalY]

            schedule_agents = self.obtainSchedule(id_agent, schedule, schedule_agents, goal_allagents, makespan + 1)

            str_id = '{}_{}_{}'.format(id_sol_map, id_sol_case, id_agent)
            int_id = int(hashlib.sha256(str_id.encode('utf-8')).hexdigest(), 16) % (10 ** 5)
            # hash_ids[id_agent]=np.divide(int_id,10**5)
            hash_ids[id_agent] = int_id
        print(schedule_agents)
        # print(id_sol_map, id_sol_case, hash_ids)
        return schedule_agents, goal_allagents, makespan, map_data, (id_sol_map, id_sol_case, hash_ids)

    def obtainSchedule(self, id_agent, agentplan, schedule_agents, goal_allagents, teamMakeSpan):

        name_agent = "agent{}".format(id_agent)
        [schedule_agentsState, schedule_agentsActions] = schedule_agents

        planCurrentAgent = agentplan[name_agent]
        pathLengthCurrentAgent = len(planCurrentAgent)

        actionKeyListAgent = []

        for step in range(teamMakeSpan):
            if step < pathLengthCurrentAgent:
                currentX = planCurrentAgent[step]['x']
                currentY = planCurrentAgent[step]['y']
            else:
                currentX = goal_allagents[id_agent][0]
                currentY = goal_allagents[id_agent][1]

            schedule_agentsState[step][id_agent][:] = [currentX, currentY]
            # up left down right stop
            actionVectorTarget = [0, 0, 0, 0, 0]

            # map action with respect to the change of position of agent
            if step < (pathLengthCurrentAgent - 1):
                nextX = planCurrentAgent[step + 1]['x']
                nextY = planCurrentAgent[step + 1]['y']
                # actionCurrent = [nextX - currentX, nextY - currentY]

            elif step >= (pathLengthCurrentAgent - 1):
                nextX = goal_allagents[id_agent][0]
                nextY = goal_allagents[id_agent][1]

            actionCurrent = [nextX - currentX, nextY - currentY]

            actionKeyIndex = self.delta.index(actionCurrent)
            actionKeyListAgent.append(actionKeyIndex)

            actionVectorTarget[actionKeyIndex] = 1
            schedule_agentsActions[step][id_agent][:] = actionVectorTarget

        return [schedule_agentsState, schedule_agentsActions]

    def setup_map(self, list_posObstacle):
        num_obstacle = len(list_posObstacle)
        map_data = np.zeros(self.size_map)
        for ID_obs in range(num_obstacle):
            obstacleIndexX = list_posObstacle[ID_obs][0]
            obstacleIndexY = list_posObstacle[ID_obs][1]
            map_data[obstacleIndexX][obstacleIndexY] = 1

        return map_data

    def pathtransformer_RelativeCoordinate(self, map_data, agents_schedule, agents_goal, makespan, ID_case, mode):
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}
        GSO, communicationRadius = self.getAdjacencyMatrix(schedule_agentsState, self.communicationRadius)
        (id_sol_map, id_sol_case, _) = ID_case
        # transform into relative Coordinate, loop "makespan" times
        self.AgentState.setmap(map_data)
        input_seq_tensor = self.AgentState.toSeqInputTensor(agents_goal, schedule_agentsState, makespan)

        list_input = input_seq_tensor.cpu().detach().numpy()
        save_PairredData.update({'map': map_data, 'goal': agents_goal, 'inputState': schedule_agentsState,
                                 'inputTensor': list_input, 'target': schedule_agentsActions,
                                 'GSO': GSO, 'makespan': makespan, 'HashIDs': ID_case[2], 'ID_Map': int(id_sol_map),
                                 'ID_case': int(id_sol_case)})
        # print(save_PairredData)
        self.save(mode, save_PairredData, ID_case, makespan)
        print("Save as  {}set_#{} from MAP ID_{}.".format(mode, ID_case[1], ID_case[0]))

    def pathtransformer_test(self, map_data, agents_schedule, agents_goal, makespan, ID_case, mode):
        # input: start and goal position,
        # output: a set of file,
        #         each file consist of state (map. goal, state) and target (action for current state)
        (id_sol_map, id_sol_case, _) = ID_case
        [schedule_agentsState, schedule_agentsActions] = agents_schedule
        save_PairredData = {}
        save_PairredData.update({'map': map_data, 'goal': agents_goal,
                                 'inputState': schedule_agentsState[0],
                                 'target': schedule_agentsActions,
                                 'makespan': makespan, 'HashIDs': ID_case[2], 'ID_Map': int(id_sol_map),
                                 'ID_case': int(id_sol_case)})
        # print(save_PairredData)
        self.save(mode, save_PairredData, ID_case, makespan)
        print("Save as  {}set_#{} from MAP ID_{}.".format(mode, ID_case[1], ID_case[0]))

    def save(self, mode, save_PairredData, ID_case, makespan):

        (id_sol_map, id_sol_case, _) = ID_case

        file_name = os.path.join(self.path_save_solDATA, mode,
                                 '{}_IDMap{}_IDCase{}_MP{}.mat'.format(mode, id_sol_map, id_sol_case, makespan))
        # print(file_name)

        sio.savemat(file_name, save_PairredData)

    def record_pathdata(self, mode, ID_case, makespan):
        (id_sol_map, id_sol_case) = ID_case
        data_name_mat = '{}_IDMap{}_IDCase{}_MP{}.mat'.format(mode, id_sol_map, id_sol_case, makespan)

        if mode == "train":
            self.list_seqtrain_file.append([data_name_mat, makespan, 0])
            # print("\n train --", self.list_seqtrain_file)
            for step in range(makespan):
                self.list_train_file.append([data_name_mat, step, 0])
        elif mode == 'validStep':
            self.list_seqvalid_file.append([data_name_mat, makespan, 0])
            for step in range(makespan):
                self.list_validStep_file.append([data_name_mat, step, 0])
        elif mode == "valid":
            self.list_valid_file.append([data_name_mat, makespan, 0])  # 0
        elif mode == "test":
            self.list_test_file.append([data_name_mat, makespan, 0])  # 0

    def save_filepath(self):
        dirName = self.path_save_solDATA

        file_seqtrain_name = os.path.join(dirName, '{}seq_filename.csv'.format('train'))
        with open(file_seqtrain_name, "w", newline="") as f:
            writer = csv.writer(f)
            print("\n train hello --", self.list_seqtrain_file)
            writer.writerows(self.list_seqtrain_file)

        file_train_name = os.path.join(dirName, '{}_filename.csv'.format('train'))
        with open(file_train_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.list_train_file)

        file_seqvalid_name = os.path.join(dirName, '{}seq_filename.csv'.format('valid'))
        with open(file_seqvalid_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.list_seqvalid_file)

        file_validStep_name = os.path.join(dirName, '{}_filename.csv'.format('validStep'))
        with open(file_validStep_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.list_validStep_file)

        file_valid_name = os.path.join(dirName, '{}_filename.csv'.format('valid'))
        with open(file_valid_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.list_valid_file)

        file_test_name = os.path.join(dirName, '{}_filename.csv'.format('test'))
        with open(file_test_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.list_test_file)

    def search_failureCases(self, dir):
        # make a list of file name of input yaml
        list_path = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    path = os.path.join(root, fname)
                    list_path.append(path)

        return list_path

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.yaml']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)

    def computeAdjacencyMatrix(self, pos, CommunicationRadius, connected=True):

        # First, transpose the axis of pos so that the rest of the code follows
        # through as legible as possible (i.e. convert the last two dimensions
        # from 2 x nNodes to nNodes x 2)
        # pos: TimeSteps x nAgents x 2 (X, Y)

        # Get the appropriate dimensions
        nSamples = pos.shape[0]
        len_TimeSteps = pos.shape[0]  # length of timesteps
        nNodes = pos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices
        W = np.zeros([len_TimeSteps, nNodes, nNodes])
        threshold = CommunicationRadius  # We compute a different
        # threshold for each sample, because otherwise one bad trajectory
        # will ruin all the adjacency matrices

        for t in range(len_TimeSteps):
            # Compute the distances
            distances = squareform(pdist(pos[t]))  # nNodes x nNodes
            # Threshold them
            W[t] = (distances < threshold).astype(pos.dtype)
            # And get rid of the self-loops
            W[t] = W[t] - np.diag(np.diag(W[t]))
            # Now, check if it is connected, if not, let's make the
            # threshold bigger
            while (not graph.isConnected(W[t])) and (connected):
                # while (not graph.isConnected(W[t])) and (connected):
                # Increase threshold
                threshold = threshold * 1.1  # Increase 10%
                # Compute adjacency matrix
                W[t] = (distances < threshold).astype(pos.dtype)
                W[t] = W[t] - np.diag(np.diag(W[t]))

        # And since the threshold has probably changed, and we want the same
        # threshold for all nodes, we repeat:
        W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
        for t in range(len_TimeSteps):
            # Initial matrix
            allagentPos = pos[t]
            distances = squareform(pdist(allagentPos, 'euclidean'))  # nNodes x nNodes

            W_t = (distances < threshold).astype(allagentPos.dtype)
            W_t = W_t - np.diag(np.diag(W_t))

            if np.any(W):
                # if W is all non-zero matrix, do normalization
                if self.config.symmetric_norm:
                    deg = np.sum(W_t, axis=0)  # nNodes (degree vector)
                    zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)
                    deg[zeroDeg] = 1.
                    invSqrtDeg = np.sqrt(1. / deg)
                    invSqrtDeg[zeroDeg] = 0.
                    Deg = np.diag(invSqrtDeg)
                    W_t = Deg @ W_t @ Deg

                maxEigenValue = self.get_maxEigenValue(W_t)
                W_norm[t] = W_t / maxEigenValue
            else:
                # if W is all zero matrix, don't do any normalization
                W_norm[t] = W
        return W_norm, threshold

    def get_maxEigenValue(self, matrix):

        isSymmetric = np.allclose(matrix, np.transpose(matrix, axes=[1, 0])) 
        if isSymmetric: 
            W = np.linalg.eigvalsh(matrix)
        else:
            W = np.linalg.eigvals(matrix)

        maxEigenvalue = np.max(np.real(W), axis=0)
        return maxEigenvalue
        # return np.max(np.abs(np.linalg.eig(matrix)[0]))

    def computeAdjacencyMatrix_fixedCommRadius(self, pos, CommunicationRadius, connected=True):
        len_TimeSteps = pos.shape[0]  # length of timesteps
        nNodes = pos.shape[1]  # Number of nodes
        # Create the space to hold the adjacency matrices

        W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
        for t in range(len_TimeSteps):
            # Initial matrix
            allagentPos = pos[t]
            distances = squareform(pdist(allagentPos, 'euclidean'))  # nNodes x nNodes

            W = (distances < CommunicationRadius).astype(allagentPos.dtype)  
            W = W - np.diag(np.diag(W))  

            if np.any(W):
                # if W is all non-zero matrix, do normalization
                if self.config.symmetric_norm:
                    deg = np.sum(W, axis=0)  # nNodes (degree vector)
                    zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)
                    deg[zeroDeg] = 1.
                    invSqrtDeg = np.sqrt(1. / deg)
                    invSqrtDeg[zeroDeg] = 0.
                    Deg = np.diag(invSqrtDeg)
                    W = Deg @ W @ Deg

                maxEigenValue = self.get_maxEigenValue(W)
                W_norm[t] = W / maxEigenValue
            else:
                # if W is all zero matrix, don't do any normalization
                print('No robot are connected at this moment, all zero matrix.')
                W_norm[t] = W

        return W_norm, CommunicationRadius


if __name__ == '__main__':
    DataTransformer = DataTransformer(args)

    DataTransformer.reset()
    DataTransformer.solutionTransformer()
