import scipy.io as sio
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from dataloader.getState import getState
import argparse
from utils.config import *
from agents import *


class DataTransformer:
    def __init__(self, config):
        self.config = config
        self.num_agents = self.config.num_agents
        self.R = (self.config.FOV-1)/2
        self.size_map = [self.config.map_w, self.config.map_w]
        self.label_density = str(self.config.map_density).split('.')[-1]

        self.nearest_obstacle_num = self.config.nearest_obstacle_num
        self.nearest_robot_num = self.config.nearest_robot_num
        self.nearest_step_num = self.config.nearest_step_num
        self.label_setup = 'map{:02d}x{:02d}_density_p{}/{}_Agent'.format(self.size_map[0], self.size_map[1], self.label_density, self.num_agents)
        self.dirName_parent = os.path.join(self.config.data_root, self.label_setup)
        self.dirName_train = os.path.join(self.dirName_parent, 'train')
        self.dirName_test = os.path.join(self.dirName_parent, 'test')
        self.dirName_valid = os.path.join(self.dirName_parent, 'valid')
        self.list_train = self.search_cases(self.dirName_train)
        self.list_test = self.search_cases(self.dirName_test)
        self.list_valid = self.search_cases(self.dirName_valid)



    def solutionTransformer(self):
        for address in self.list_train:
            self.dealFeatureVector(address)

        for address in self.list_valid:
            self.dealFeatureVector(address)

        # for address in self.list_test:
        #     self.dealFeatureVector_testDatset(address)

    def dealFeatureVector(self, address):
        data_contents = sio.loadmat(address)
        global_map = data_contents['map']  # W x H
        final_goal = data_contents['goal']  # num_agent x 2
        gso = data_contents['GSO']  # step x num_agent x num_agent
        agent_pos = data_contents['inputState']  # makespan x num_agent x 2
        agent_action = data_contents['target']  # makespan x num_agent x 5
        makespan = data_contents['makespan'][0][0]
        id_sol_map = data_contents['ID_Map'][0][0]
        id_sol_case = data_contents['ID_case'][0][0]
        inputState = data_contents['inputState']
        inputTensor = data_contents['inputTensor']

        feat_own_path_list, feat_agent_path_list, feat_final_goal_list = getState(makespan, self.num_agents, self.nearest_step_num, self.nearest_robot_num, agent_pos, self.R, final_goal, 0)

        save_PairredData = {}
        save_PairredData.update({'goal': final_goal, 'target': agent_action, 'GSO': gso, 'map': global_map,
                                 'inputState': inputState,'inputTensor': inputTensor,
                                 'makespan': makespan, 'ID_Map': int(id_sol_map), 'ID_case': int(id_sol_case),
                                 'feat_own_path_list': feat_own_path_list,
                                 'feat_agent_path_list': feat_agent_path_list, 'feat_final_goal_list': feat_final_goal_list})
        sio.savemat(address, save_PairredData)
        print("Save as {}.".format(address))

    def dealFeatureVector_testDatset(self, address):
        data_contents = sio.loadmat(address)
        global_map = data_contents['map']  # W x H
        final_goal = data_contents['goal']  # num_agent x 2
        agent_pos = data_contents['inputState']  # num_agent x 2
        agent_action = data_contents['target']  # makespan x num_agent x 5
        makespan = data_contents['makespan'][0][0]
        id_sol_map = data_contents['ID_Map'][0][0]
        id_sol_case = data_contents['ID_case'][0][0]
        inputState = data_contents['inputState']


        save_PairredData = {}
        save_PairredData.update({'goal': final_goal, 'target': agent_action, 'map': global_map,
                                 'inputState': inputState,
                                 'makespan': makespan, 'ID_Map': int(id_sol_map), 'ID_case': int(id_sol_case)})
        sio.savemat(address, save_PairredData)
        print("Save as {}.".format(address))

    def search_cases(self, dir):
        list_path = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    path = os.path.join(root, fname)
                    list_path.append(path)

        return list_path

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.mat']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)


if __name__ == '__main__':
    begin_time = time.time()
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    configs, _ = get_config_from_json(args.config)

    DataTransformer = DataTransformer(configs)
    DataTransformer.solutionTransformer()
    time.sleep(5)
    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
