import numpy as np
import torch

def pos(pos_x, pos_y, max_len=4.4, numcell=20):
    each_len = max_len/numcell
    idx_x = int(pos_x/each_len)
    idx_y = int(pos_y/each_len)
    return idx_x, idx_y
    
def move(action, pos_x, pos_y, max_len=4.4, numcell=20):
    each_len = max_len/numcell
    if action == 2 and pos_y<=(max_len-each_len):
        pos_y += each_len
    elif action == 1 and pos_x>=each_len:
        pos_x -= each_len
    elif action == 0 and pos_y>=each_len:
        pos_y -= each_len
    elif action == 3 and pos_x<=(max_len-each_len):
        pos_x += each_len
    return pos_x, pos_y
    
def getstate(current_x, current_y, other_x, other_y, target_x, target_y, pos_map, obstacle_map):
    
    current_x, current_y = pos(current_x, current_y, max_len=4.4, numcell=20)
    other_x, other_y = pos(other_x, other_y, max_len=4.4, numcell=20)
    #channel_pos = pos_map[current_x:current_x+11,current_y:current_y+11]
    channel_pos = torch.zeros(11,11)
    channel_pos[5,5] = 1
    if other_x != 0 and other_y != 0:
        if abs(current_x-other_x)<=5 and abs(current_y-other_y)<=5:
            channel_pos[int(5+(other_x-current_x)),int(5+(other_y-current_y))] = 1
  
    channel_obstacle = obstacle_map[current_x:current_x+11,current_y:current_y+11]
    
    
    channel_goal = torch.zeros(11,11)
    diff_x = target_x - current_x
    diff_y = target_y - current_y      
    channel_goal[int(5+np.sign(diff_y)*min(5,abs(diff_y))),int(5+np.sign(diff_x)*min(5,abs(diff_x)))] = 1
    
    state = torch.cat((channel_obstacle.view(1,1,1,11,11), channel_goal.view(1,1,1,11,11), channel_pos.view(1,1,1,11,11)),2)
    
    gso = torch.zeros((1,2,2))
    gso[0,0,0] = 1
    if abs(current_x-other_x)<=5 and abs(current_y-other_y)<=5:
        gso[0,0,1] = 1
        gso[0,1,0] = 1
           
    return state, gso
    
def update_path(old_path, newx, newy):
    new_cor = torch.tensor([[newx,newy]])
    old_path[0,0,:] = torch.cat((old_path[0,0:1,2:],new_cor),1)
    return old_path
    

