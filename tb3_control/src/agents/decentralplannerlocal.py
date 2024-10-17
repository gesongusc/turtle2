import shutil
import os
import time
import torch.optim as optim
from agents.base import BaseAgent
from graphs.models.decentralplanner import *
from utils.metrics import MonitoringMultiAgentPerformance
from graphs.models.decentralplanner import DecentralPlannerNet
#from rosnode.pub import Publisher
#from rosnode.sub import Subscriber, PosSubscriber
from policy.env import pos, move, getstate, update_path
import numpy as np
import torch
import torch.nn as nn
#import rclpy
import cv2

class DecentralPlannerAgentLocal(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #self.test = self.test_single

        self.config.device = 'cuda'
        
        self.model = DecentralPlannerNet(self.config).to(self.config.device)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()                    
        self.obstacle_map = torch.zeros(30,30)            
        self.pos_map = torch.zeros(30,30)    
        self.path = torch.zeros((1,2,60))        
        
    def load_checkpoint(self):
        filename = "/home/ge_orin/catkin_ws/src/tb3_control/src/checkpoint/model_best.pth.tar"
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['state_dict'])
                
    def update_obs(self, obstacle_y, obstacle_x):         
        for i in range(obstacle_x.shape[0]):
            self.obstacle_map[obstacle_x[i]+5,obstacle_y[i]+5] = 1 
            
    def update_target(self, target_x_1, target_y_1, target_x_2, target_y_2):
        self.target_x_1 = target_x_1 ## cell index
        self.target_y_1 = target_y_1
        self.target_x_2 = target_x_2 ## cell index
        self.target_y_2 = target_y_2
        
    def run(self, current_x, current_y, other_x, other_y):
        self.model.eval()                     
                
        state, gso = getstate(current_x, current_y, other_x, other_y, self.target_x_2, self.target_y_2, self.pos_map, self.obstacle_map)
        if other_x == 0.0 and other_y == 0.0:
            otherstate = torch.zeros((1,1,3,11,11))
        else:
            otherstate,_ = getstate(other_x, other_y, current_x, current_y, self.target_x_1, self.target_y_1, self.pos_map, self.obstacle_map)

        input = torch.cat((state, otherstate),1).to(self.config.device)
        #input = state.to(self.config.device)
        gso = gso.to(self.config.device)
                
        self.model.addGSO(gso)
        actionVec_predict = self.model(input, self.path).reshape(gso.shape[1], 5)
        actions = nn.functional.softmax(actionVec_predict[:1,:],dim=1).detach().cpu().numpy()
        index = np.argmax(actions)

        move_x, move_y = move(index, current_x, current_y, max_len=4.4, numcell=20)
        
        nowx, nowy = pos(current_x, current_y, max_len=4.4, numcell=20)
        self.path = update_path(self.path, nowx, nowy)
        return np.array([current_x, current_y, move_x, move_y, move_x, move_y, move_x, move_y])
            
            
