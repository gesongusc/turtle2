import os
import numpy as np
import scipy.io as sio
import random
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from dataloader.statetransformer_Guidance import AgentState


class DecentralPlannerDataLoader:
    def __init__(self, config):
        self.config = config
        if config.mode == "train":
            train_set = CreateDataset(self.config, "train")
            test_trainingSet = CreateDataset(self.config, "test_trainingSet")
            valid_set = CreateDataset(self.config, "valid")

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)

            self.test_trainingSet_loader = DataLoader(test_trainingSet, batch_size=self.config.valid_batch_size, shuffle=True,
                                                      num_workers=self.config.data_loader_workers,
                                                      pin_memory=self.config.pin_memory)

            self.valid_loader = DataLoader(valid_set, batch_size=self.config.valid_batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
        elif config.mode == "test":
            if self.config.test_on_ValidSet:
                testset = CreateDataset(self.config, 'valid')
            else:
                testset = CreateDataset(self.config, 'test')

            self.test_loader = DataLoader(testset, batch_size=self.config.test_batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
        else:
            raise Exception("Please specify in the json a specified mode in mode")

    def finalize(self):
        pass


class CreateDataset(data.Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.datapath_exp = '{}{:02d}x{:02d}_density_p{}/{}_Agent/'.format(self.config.map_type, self.config.map_w, self.config.map_h, self.config.map_density, self.config.num_agents)
        self.dirName = os.path.join(self.config.data_root, self.datapath_exp)
        self.AgentState = AgentState(self.config)

        if mode == "train":
            self.dir_data = os.path.join(self.dirName, 'train')
            self.search_files = self.search_target_files_withStep
            self.data_paths, self.id_stepdata = self.getTrainingsetData(self.dir_data)
            self.load_data = self.load_train_data
        elif mode == "test_trainingSet":
            self.dir_data = os.path.join(self.dirName, 'train')
            data_paths, id_stepdata = self.search_target_files(self.dir_data)
            paths_total = list(zip(data_paths, id_stepdata))
            random.shuffle(paths_total)
            data_paths, id_stepdata = zip(*paths_total)
            self.data_paths = data_paths[:self.config.num_test_trainingSet]
            self.id_stepdata = id_stepdata[:self.config.num_test_trainingSet]
            self.load_data = self.load_data_during_training
        elif mode == "valid":
            self.dir_data = os.path.join(self.dirName, 'valid')
            self.data_paths, self.id_stepdata = self.obtain_data_path_validset(self.dir_data, self.config.num_validset)
            self.load_data = self.load_data_during_training
        elif mode == "test":
            self.dir_data = os.path.join(self.dirName, mode)
            self.data_paths, self.id_stepdata = self.obtain_data_path_validset(self.dir_data, self.config.num_testset)
            self.load_data = self.load_test_data

        self.data_size = len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index % self.data_size]
        id_step = int(self.id_stepdata[index % self.data_size])
        input, agentPath, target, GSO, map_tensor, ID_Map, ID_case = self.load_data(path, id_step)
        return input, agentPath, target, (ID_Map, ID_case, id_step), GSO, map_tensor

    def getTrainingsetData(self, dir_data):
        data_paths_total = []
        step_paths_total = []

        data_paths, step_paths = self.search_files(dir_data)
        data_paths_total.extend(data_paths)
        step_paths_total.extend(step_paths)

        paths_total = list(zip(data_paths_total, step_paths_total))
        random.shuffle(paths_total)
        data_paths_total, step_paths_total = zip(*paths_total)
        return data_paths_total, step_paths_total

    def obtain_data_path_validset(self, dir_data, case_limit):
        data_paths, id_stepdata = self.search_target_files(dir_data)
        paths_bundle = list(zip(data_paths, id_stepdata))
        paths_bundle = sorted(paths_bundle)
        if self.config.shuffle_testSet:
            random.shuffle(paths_bundle)
            random.shuffle(paths_bundle)
        data_paths, id_stepdata = zip(*paths_bundle)
        data_paths = data_paths[:case_limit]
        id_stepdata = id_stepdata[:case_limit]
        return data_paths, id_stepdata

    def load_train_data(self, path, id_step):
        data_contents = sio.loadmat(path)
        map_channel = data_contents['map']  # W x H

        input_tensor = data_contents['inputTensor']  # step x num_agent x 3 x 11 x 11
        target_sequence = data_contents['target']  # step x num_agent x 5
        input_GSO_sequence = data_contents['GSO']  # Step x num_agent x num_agent
        feat_agent_path = data_contents['feat_agent_path_list']
        feat_final_goal = data_contents['feat_final_goal_list']
        feat_own_path = data_contents['feat_own_path_list']
        all_agent_pos = data_contents['inputState']
        ID_Map = data_contents['ID_Map']
        ID_case = data_contents['ID_case']
        tensor_map = torch.from_numpy(map_channel).float()

        step_input_tensor = torch.from_numpy(input_tensor[id_step][:]).float()
        step_input_GSO = torch.from_numpy(input_GSO_sequence[id_step, :, :]).float()
        step_target = torch.from_numpy(target_sequence[id_step, :, :]).long()


        agent_path_reshaped_pre = feat_agent_path[id_step, :, :, :, :].reshape(
            (feat_agent_path.shape[1], feat_agent_path.shape[2], feat_agent_path.shape[3] * 2), order='F')
        agent_path_reshaped = agent_path_reshaped_pre.reshape(
            (agent_path_reshaped_pre.shape[0], agent_path_reshaped_pre.shape[1] * agent_path_reshaped_pre.shape[2]), order='C')
        final_goal_reshaped = feat_final_goal[id_step, :, :, :].reshape((feat_final_goal.shape[1], feat_final_goal.shape[2] * 2), order='F')
        own_path_reshaped = feat_own_path[id_step, :, :, :].reshape((feat_own_path.shape[1], feat_own_path.shape[2] * 2), order='F')
        final = np.hstack((own_path_reshaped, agent_path_reshaped))

        step_final = torch.from_numpy(final).float()
        return step_input_tensor, final, step_target, step_input_GSO, tensor_map, ID_Map, ID_case

    def load_data_during_training(self, path, _):
        data_contents = sio.loadmat(path)
        map_channel = data_contents['map']  # W x H
        goal_allagents = data_contents['goal']  # num_agent x 2

        input_sequence = data_contents['inputState'][0]  # from step x num_agent x 2 to # initial pos x num_agent x 2
        target_sequence = data_contents['target']  # step x num_agent x 5
        ID_Map = data_contents['ID_Map'][0][0]
        ID_case = data_contents['ID_case'][0][0]
        self.AgentState.setmap(map_channel)
        step_input_tensor = self.AgentState.stackinfo(goal_allagents, input_sequence)

        step_target = torch.from_numpy(target_sequence).long()
        # from step x num_agent x action (5) to  id_agent x step x action(5)
        step_target = step_target.permute(1, 0, 2)
        step_input_rs = step_input_tensor.squeeze(0)
        step_target_rs = step_target.squeeze(0)

        tensor_map = torch.from_numpy(map_channel).float()
        GSO_none = torch.zeros(1)
        return step_input_rs, "", step_target_rs, GSO_none, tensor_map, ID_Map, ID_case


    def load_test_data(self, path, _):
        # load dataset into test mode - only initial position, predict action towards goal

        data_contents = sio.loadmat(path)
        map_channel = data_contents['map']  # W x H
        goal_allagents = data_contents['goal']  # num_agent x 2

        input_sequence = data_contents['inputState']  # num_agent x 2
        target_sequence = data_contents['target']  # step x num_agent x 5
        ID_Map = data_contents['ID_Map']
        ID_case = data_contents['ID_case']

        self.AgentState.setmap(map_channel)
        step_input_tensor = self.AgentState.stackinfo(goal_allagents, input_sequence)

        step_target = torch.from_numpy(target_sequence).long()
        # from step x num_agent x action (5) to  id_agent x step x action(5)
        step_target = step_target.permute(1, 0, 2)
        step_input_rs = step_input_tensor.squeeze(0)
        step_target_rs = step_target.squeeze(0)

        tensor_map = torch.from_numpy(map_channel).float()
        GSO_none = torch.zeros(1)
        return step_input_rs, "", step_target_rs, GSO_none, tensor_map, ID_Map, ID_case


    def search_target_files(self, dir):
        list_path = []
        list_path_stepdata = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    makespan = int(fname.split('_MP')[-1].split('.mat')[0])
                    path = os.path.join(root, fname)
                    list_path.append(path)
                    list_path_stepdata.append(makespan)
        return list_path, list_path_stepdata

    def search_target_files_withStep(self, dir):
        list_path = []
        list_path_stepdata = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_target_file(fname):
                    makespan = int(fname.split('_MP')[-1].split('.mat')[0])
                    path = os.path.join(root, fname)
                    for step in range(makespan):
                        list_path.append(path)
                        list_path_stepdata.append(step)

        return list_path, list_path_stepdata

    def is_target_file(self, filename):
        DATA_EXTENSIONS = ['.mat']
        return any(filename.endswith(extension) for extension in DATA_EXTENSIONS)

    def __len__(self):
        return self.data_size
