#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:25:53 2021

@author: junlinou
"""

import numpy as np
import math
from numba import cuda, types
from numba import jit, njit
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
#from numba.cuda.random import xoroshiro128p_normal_float64, xoroshiro128p_uniform_float32
#import numba
#import random
@njit
def pop_init(paths_p, number_candidate, number_of_genes):
    number_population = paths_p.shape[0]
    new_population = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    for i in range(number_population):
        new_population[i, :, 0] = paths_p[i,0]
        new_population[i, :, 1] = paths_p[i,1]
        new_population[i, :, number_of_genes - 2] = paths_p[i,number_of_genes - 2]
        new_population[i, :,number_of_genes - 1] = paths_p[i,number_of_genes - 1]
    return new_population


@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def population_path_free_G(new_population_out, obstacles_out, num_edge_out, rng_states, pop_out):
    x, y = cuda.grid(2)
    if x < new_population_out.shape[0] and y < new_population_out.shape[1]:
        # Thread id in a 2D block
        blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
        threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
        for i in range(2, new_population_out.shape[2] - 2, 2):
            c = 1
            while c%2 == 1:
                n2 = 0
                pop_x = (pop_out[x,i] + xoroshiro128p_normal_float32(rng_states, threadId))%4.4
                pop_y = (pop_out[x,i+1]+ xoroshiro128p_normal_float32(rng_states, threadId))%4.4

    
                for m in range(num_edge_out.shape[0]):
                    c = 0
                    for n in range(num_edge_out[m]):
                        n1 = (n+1)%num_edge_out[m]
                        A_x = obstacles_out[n2+n, 0]
                        A_y = obstacles_out[n2+n, 1]
                        B_x = obstacles_out[n2+n1, 0]
                        B_y = obstacles_out[n2+n1, 1]
    
                        if (((pop_y > A_y) != (pop_y > B_y)) and ((pop_x - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (pop_y - A_y)*(B_y - A_y))):
                            c += 1
                    if c%2 == 1:
                        break
                    n2 += num_edge_out[m]
            if y == 0:
                new_population_out[x,y,i] = pop_out[x,i]
                new_population_out[x,y,i + 1] = pop_out[x,i+1]
            else:
                new_population_out[x,y,i] = pop_x
                new_population_out[x,y,i + 1] = pop_y


@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def population_path_free_G_old(new_population_out, obstacles_out, num_edge_out, rng_states, pop_out):
    x, y = cuda.grid(2)
    if x < new_population_out.shape[0] and y < new_population_out.shape[1]:
        # Thread id in a 2D block
        blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
        threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
        L_limit = 0.52
        for i in range(2, new_population_out.shape[2] - 2, 2):
            c = 1
            while c%2 == 1:
                n2 = 0
                pop_x = (pop_out[x,i] + xoroshiro128p_normal_float32(rng_states, threadId))%4.4
                pop_y = (pop_out[x,i+1]+ xoroshiro128p_normal_float32(rng_states, threadId))%4.4
                if i == 2:
                    L = math.sqrt((pop_x-pop_out[x,0])**2 + (pop_y-pop_out[x,1])**2)
                    if L>L_limit:
                        pop_x = L_limit/L*(pop_x-pop_out[x,0]) + pop_out[x,0]
                        pop_y = L_limit/L*(pop_y-pop_out[x,1]) + pop_out[x,1]
    
                for m in range(num_edge_out.shape[0]):
                    c = 0
                    for n in range(num_edge_out[m]):
                        n1 = (n+1)%num_edge_out[m]
                        A_x = obstacles_out[n2+n, 0]
                        A_y = obstacles_out[n2+n, 1]
                        B_x = obstacles_out[n2+n1, 0]
                        B_y = obstacles_out[n2+n1, 1]
    
                        if (((pop_y > A_y) != (pop_y > B_y)) and ((pop_x - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (pop_y - A_y)*(B_y - A_y))):
                            c += 1
                    if c%2 == 1:
                        break
                    n2 += num_edge_out[m]
            if y == 0:
                new_population_out[x,y,i] = pop_out[x,i]
                new_population_out[x,y,i + 1] = pop_out[x,i+1]
            else:
                new_population_out[x,y,i] = pop_x
                new_population_out[x,y,i + 1] = pop_y


@cuda.jit('(float32[:, :, :],  float32[:, :], int32[:], float32[:, :], int32[:], float32[:, :], int32[:], int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:])')
def fitness(pop, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out):

    # 2 D coordinates
    x, y = cuda.grid(2)
    
    if x < pop.shape[0] and y < pop.shape[1]:
    
   
        # lamda_m = 0.3
        # lamda_s = 0.2

        ob_speed = 0.2
        ratio = 1
        speed = 0.26
        time_int = 0.2
        pp = 10000
        summ = 0
        safety = 0
  
        risk = 0
        risk_t = 0
        sum_dis = 0
        sum_dir = 0
        total_penalty = 0.0

        sum_risk = 0
        risk_det = 0
        
        start_x = pop[x, y, 0]
        start_y = pop[x, y, 1]
        
        waypoint_1_x = pop[x, y, 2]
        waypoint_1_y = pop[x, y, 3]

        waypoint_2_x = pop[x, y, 4]
        waypoint_2_y = pop[x, y, 5]

        end_x = pop[x, y, 6]
        end_y = pop[x, y, 7]

        length_1 = math.sqrt((waypoint_1_x-start_x)**2+(waypoint_1_y-start_y)**2)
        length_2 = math.sqrt((waypoint_2_x-waypoint_1_x)**2+(waypoint_2_y-waypoint_1_y)**2)
        length_3 = math.sqrt((end_x -waypoint_2_x)**2+(end_y-waypoint_2_y)**2)
        
        cumulative_length_1 = length_1
        cumulative_length_2 = cumulative_length_1 + length_2
        cumulative_length_3 = cumulative_length_2 + length_3

        length = length_p_out[x]
        smoothness = smoothness_p_out[x]
        
        
        distance_travelled_1 = speed*time_int
        if  distance_travelled_1 <= cumulative_length_1:
            ratio = distance_travelled_1 / length_1
            future_x_1 = start_x + ratio * (waypoint_1_x - start_x)
            future_y_1 = start_y + ratio * (waypoint_1_y - start_y)
        elif distance_travelled_1 <= cumulative_length_2:
            ratio = (distance_travelled_1 - length_1) / length_2
            future_x_1 = waypoint_1_x + ratio * (waypoint_2_x - waypoint_1_x)
            future_y_1 = waypoint_1_y + ratio * (waypoint_2_y - waypoint_1_y)
        elif distance_travelled_1 <= cumulative_length_3:
            ratio = (distance_travelled_1 - cumulative_length_2) / length_3
            future_x_1 = waypoint_2_x + ratio * (end_x - waypoint_2_x)
            future_y_1 = waypoint_2_y + ratio * (end_y - waypoint_2_y)
        else:
            future_x_1 = end_x
            future_y_1 = end_y

        distance_travelled_2 = speed*time_int*2
        if  distance_travelled_2 <= cumulative_length_1:
            ratio = distance_travelled_2 / length_1
            future_x_2 = start_x + ratio * (waypoint_1_x - start_x)
            future_y_2 = start_y + ratio * (waypoint_1_y - start_y)
        elif distance_travelled_2 <= cumulative_length_2:
            ratio = (distance_travelled_2 - length_1) / length_2
            future_x_2 = waypoint_1_x + ratio * (waypoint_2_x - waypoint_1_x)
            future_y_2 = waypoint_1_y + ratio * (waypoint_2_y - waypoint_1_y)
        elif distance_travelled_2 <= cumulative_length_3:
            ratio = (distance_travelled_2 - cumulative_length_2) / length_3
            future_x_2 = waypoint_2_x + ratio * (end_x - waypoint_2_x)
            future_y_2 = waypoint_2_y + ratio * (end_y - waypoint_2_y)
        else:
            future_x_2 = end_x
            future_y_2 = end_y

        distance_travelled_3 = speed*time_int*3
        if  distance_travelled_3 <= cumulative_length_1:
            ratio = distance_travelled_3 / length_1
            future_x_3 = start_x + ratio * (waypoint_1_x - start_x)
            future_y_3 = start_y + ratio * (waypoint_1_y - start_y)
        elif distance_travelled_3 <= cumulative_length_2:
            ratio = (distance_travelled_3 - length_1) / length_2
            future_x_3 = waypoint_1_x + ratio * (waypoint_2_x - waypoint_1_x)
            future_y_3 = waypoint_1_y + ratio * (waypoint_2_y - waypoint_1_y)
        elif distance_travelled_3 <= cumulative_length_3:
            ratio = (distance_travelled_3 - cumulative_length_2) / length_3
            future_x_3 = waypoint_2_x + ratio * (end_x - waypoint_2_x)
            future_y_3 = waypoint_2_y + ratio * (end_y - waypoint_2_y)
        else:
            future_x_3 = end_x
            future_y_3 = end_y

        distance_travelled_4 = speed*time_int*4
        if  distance_travelled_4 <= cumulative_length_1:
            ratio = distance_travelled_4 / length_1
            future_x_4 = start_x + ratio * (waypoint_1_x - start_x)
            future_y_4 = start_y + ratio * (waypoint_1_y - start_y)
        elif distance_travelled_4 <= cumulative_length_2:
            ratio = (distance_travelled_4 - length_1) / length_2
            future_x_4 = waypoint_1_x + ratio * (waypoint_2_x - waypoint_1_x)
            future_y_4 = waypoint_1_y + ratio * (waypoint_2_y - waypoint_1_y)
        elif distance_travelled_4 <= cumulative_length_3:
            ratio = (distance_travelled_4 - cumulative_length_2) / length_3
            future_x_4 = waypoint_2_x + ratio * (end_x - waypoint_2_x)
            future_y_4 = waypoint_2_y + ratio * (end_y - waypoint_2_y)
        else:
            future_x_4 = end_x
            future_y_4 = end_y

        distance_travelled_5 = speed*time_int*5
        if  distance_travelled_5 <= cumulative_length_1:
            ratio = distance_travelled_5 / length_1
            future_x_5 = start_x + ratio * (waypoint_1_x - start_x)
            future_y_5 = start_y + ratio * (waypoint_1_y - start_y)
        elif distance_travelled_5 <= cumulative_length_2:
            ratio = (distance_travelled_5 - length_1) / length_2
            future_x_5 = waypoint_1_x + ratio * (waypoint_2_x - waypoint_1_x)
            future_y_5 = waypoint_1_y + ratio * (waypoint_2_y - waypoint_1_y)
        elif distance_travelled_5 <= cumulative_length_3:
            ratio = (distance_travelled_5 - cumulative_length_2) / length_3
            future_x_5 = waypoint_2_x + ratio * (end_x - waypoint_2_x)
            future_y_5 = waypoint_2_y + ratio * (end_y - waypoint_2_y)
        else:
            future_x_5 = end_x
            future_y_5 = end_y
                    
        
        obstacles_m_out = obstacles_m_out[0:4]
        position_1_out = obstacles_m_out[4:8]
        position_2_out = obstacles_m_out[8:12]
        position_3_out = obstacles_m_out[12:16]
        position_4_out = obstacles_m_out[16:20]
        
        
        obstacles_m_2_out = obstacles_m_out[20:24]
        position_2_1_out = obstacles_m_out[24:28]
        position_2_2_out = obstacles_m_out[28:32]
        position_2_3_out = obstacles_m_out[32:36]
        position_2_4_out = obstacles_m_out[36:40]
        
        
        center_ob_m_current_x = (obstacles_m_out[0][0]+obstacles_m_out[2][0])/2
        center_ob_m_current_y = (obstacles_m_out[0][1]+obstacles_m_out[2][1])/2

        center_ob_m_fu_x_1 = (position_1_out[0][0]+position_1_out[2][0])/2
        center_ob_m_fu_y_1 = (position_1_out[0][1]+position_1_out[2][1])/2

        center_ob_m_fu_x_2 = (position_2_out[0][0]+position_2_out[2][0])/2
        center_ob_m_fu_y_2 = (position_2_out[0][1]+position_2_out[2][1])/2

        center_ob_m_fu_x_3 = (position_3_out[0][0]+position_3_out[2][0])/2
        center_ob_m_fu_y_3 = (position_3_out[0][1]+position_3_out[2][1])/2

        center_ob_m_fu_x_4 = (position_4_out[0][0]+position_4_out[2][0])/2
        center_ob_m_fu_y_4 = (position_4_out[0][1]+position_4_out[2][1])/2

        
        center_ob_m_2_c_x = (obstacles_m_2_out[0][0]+obstacles_m_2_out[2][0])/2
        center_ob_m_2_c_y = (obstacles_m_2_out[0][1]+obstacles_m_2_out[2][1])/2
        
        center_ob_m_2_fu_x_1 = (position_2_1_out[0][0]+position_2_1_out[2][0])/2
        center_ob_m_2_fu_y_1 = (position_2_1_out[0][1]+position_2_1_out[2][1])/2

        center_ob_m_2_fu_x_2 = (position_2_2_out[0][0]+position_2_2_out[2][0])/2
        center_ob_m_2_fu_y_2 = (position_2_2_out[0][1]+position_2_2_out[2][1])/2

        center_ob_m_2_fu_x_3 = (position_2_3_out[0][0]+position_2_3_out[2][0])/2
        center_ob_m_2_fu_y_3 = (position_2_3_out[0][1]+position_2_3_out[2][1])/2

        center_ob_m_2_fu_x_4 = (position_2_4_out[0][0]+position_2_4_out[2][0])/2
        center_ob_m_2_fu_y_4 = (position_2_4_out[0][1]+position_2_4_out[2][1])/2
        

        ob_m_safe_dis = math.sqrt((center_ob_m_current_x-obstacles_m_out[0][0])**2 + (center_ob_m_current_y-obstacles_m_out[0][1])**2)
        
        
        start_t_ob_length_c = math.sqrt((start_x-center_ob_m_current_x)**2 + (start_y-center_ob_m_current_y)**2)
        start_t_ob_length_1 = math.sqrt((future_x_1-center_ob_m_fu_x_1)**2 + (future_y_1-center_ob_m_fu_y_1)**2)
        start_t_ob_length_2 = math.sqrt((future_x_2-center_ob_m_fu_x_2)**2 + (future_y_2-center_ob_m_fu_y_2)**2)
        start_t_ob_length_3 = math.sqrt((future_x_3-center_ob_m_fu_x_3)**2 + (future_y_3-center_ob_m_fu_y_3)**2)
        start_t_ob_length_4 = math.sqrt((future_x_4-center_ob_m_fu_x_4)**2 + (future_y_4-center_ob_m_fu_y_4)**2)

        start_t_ob_length_c_2 = math.sqrt((start_x-center_ob_m_2_c_x)**2 + (start_y-center_ob_m_2_c_y)**2)
        start_t_ob_length_2_1 = math.sqrt((future_x_1-center_ob_m_2_fu_x_1)**2 + (future_y_1-center_ob_m_2_fu_y_1)**2)
        start_t_ob_length_2_2 = math.sqrt((future_x_2-center_ob_m_2_fu_x_2)**2 + (future_y_2-center_ob_m_2_fu_y_2)**2)
        start_t_ob_length_2_3 = math.sqrt((future_x_3-center_ob_m_2_fu_x_3)**2 + (future_y_3-center_ob_m_2_fu_y_3)**2)
        start_t_ob_length_2_4 = math.sqrt((future_x_4-center_ob_m_2_fu_x_4)**2 + (future_y_4-center_ob_m_2_fu_y_4)**2)        

        start_t_obe_length_c = start_t_ob_length_c - ob_m_safe_dis
        start_t_obe_length_1 = start_t_ob_length_1 - ob_m_safe_dis
        start_t_obe_length_2 = start_t_ob_length_2 - ob_m_safe_dis
        start_t_obe_length_3 = start_t_ob_length_3 - ob_m_safe_dis
        start_t_obe_length_4 = start_t_ob_length_4 - ob_m_safe_dis

        start_t_obe_length_c_2 = start_t_ob_length_c_2 - ob_m_safe_dis
        start_t_obe_length_2_1 = start_t_ob_length_2_1 - ob_m_safe_dis
        start_t_obe_length_2_2 = start_t_ob_length_2_2 - ob_m_safe_dis
        start_t_obe_length_2_3 = start_t_ob_length_2_3 - ob_m_safe_dis
        start_t_obe_length_2_4 = start_t_ob_length_2_4 - ob_m_safe_dis        
        # dist = cuda.shared.array((9,64,2), dtype=types.float32)
        # a = np.zeros(2)
        #length_out[x,y] = 0
        #safety_out[x,y] = 0
        #smoothness_out[x,y] = 0
        intersection_value_out[x, y] = 1
        dx_1 = math.cos(init_dir_out[0])
        dy_1 = math.sin(init_dir_out[0])
        
        a_value = 23.4741478173084
        k_value = 1.0
        c_value = -4.2377903618239365        
        
        if start_t_obe_length_c > 1.5:
            sum_risk += 0
        elif start_t_obe_length_c < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_c) + c_value

        if start_t_obe_length_1 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_1 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_1) + c_value

        if start_t_obe_length_2 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_2 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_2) + c_value

        if start_t_obe_length_3 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_3 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_3) + c_value

        if start_t_obe_length_4 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_4 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_4) + c_value
#######    ob_2    #######
        
        if start_t_obe_length_c_2 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_c_2 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_c_2) + c_value            

        if start_t_obe_length_2_1 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_2_1 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_2_1) + c_value   

        if start_t_obe_length_2_2 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_2_2 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_2_2) + c_value

        if start_t_obe_length_2_3 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_2_3 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_2_3) + c_value

        if start_t_obe_length_2_4 > 1.5:
            sum_risk += 0
        elif start_t_obe_length_2_4 < 0.25:
            sum_risk += 10000
        else:
            sum_risk += a_value * math.exp(-k_value * start_t_obe_length_2_4) + c_value
       
        
        
        for z in range(int(pop.shape[2]/2) - 1):
            dx_2 = pop[x,y,2*z+2]-pop[x,y,2*z+0]
            dy_2 = pop[x,y,2*z+3]-pop[x,y,2*z+1]
            direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
            if dx_2!=0 or dy_2!=0:
                dx_1 = dx_2
                dy_1 = dy_2
            # if direction > math.pi/2:
            #     direction += 20
            smoothness += direction
            
            A_x = pop[x,y,2*z]
            A_y = pop[x,y,2*z+1]
            B_x = pop[x,y,2*z+2]
            B_y = pop[x,y,2*z+3]
            AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
            #min_value = 10000
            n = 0
            
            #for i in range(num_edge_out.shape[0]):
            for i in range(num_edge_s_out.shape[0]):
                #for j in range(num_edge_out[i]):
                    #n1 = (j+1)%num_edge_out[i]
                for j in range(num_edge_s_out[i]):
                    n1 = (j+1)%num_edge_s_out[i] 
 
                    C_x = obstacles_s_out[n+j, 0]
                    C_y = obstacles_s_out[n+j, 1]
                    D_x = obstacles_s_out[n+n1, 0]
                    D_y = obstacles_s_out[n+n1, 1]                                      
                    
                    #C_x = obstacles_out[n+j, 0]
                    #C_y = obstacles_out[n+j, 1]
                    #D_x = obstacles_out[n+n1, 0]
                    #D_y = obstacles_out[n+n1, 1]
                    
                    j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                    j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                    j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                    j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)
    
                        
                    if j1*j2<=0 and j3*j4<=0 and (B_x-A_x)**2+(B_y-A_y)**2!=0:
                        summ += 1
                n += num_edge_s_out[i]
                #n += num_edge_out[i]
            length += AB
        if summ == 0:
            intersection_value_out[x, y] = 0
        dx_2 = math.cos(dir_final_out[x])
        dy_2 = math.sin(dir_final_out[x])
        direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
        if dir_final_out[x] == 1000:
            direction = 0
        smoothness += direction
        # if the path does not intersect with the obstacles, we need to calculate the shortest distance between segments of path and obtacles
        if intersection_value_out[x, y] == 0:
            for z in range(3):
                # set initil value for counting the times (close to obstacle)
                time_o = 0
                # set initial value for n
                n = 0
                #coodinates of points A, B
                A_x = pop[x,y,2*z]
                A_y = pop[x,y,2*z+1]
                B_x = pop[x,y,2*z+2]
                B_y = pop[x,y,2*z+3]
                # length of AB
                AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
                for i in range(num_edge_m_out.shape[0]):
                    for j in range(num_edge_m_out[i]):
                        n1 = (j+1)%num_edge_m_out[i]
                        #coodinates of points C, D
                        C_x = obstacles_m_out[n+j, 0]
                        C_y = obstacles_m_out[n+j, 1]
                        D_x = obstacles_m_out[n+n1, 0]
                        D_y = obstacles_m_out[n+n1, 1]
                        #length of CD,......
                        CD = math.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
                        AC = math.sqrt((A_x - C_x)**2 + (A_y - C_y)**2)
                        AD = math.sqrt((A_x - D_x)**2 + (A_y - D_y)**2)
                        BC = math.sqrt((B_x - C_x)**2 + (B_y - C_y)**2)
                        BD = math.sqrt((B_x - D_x)**2 + (B_y - D_y)**2)
                        #condition 1
                        r_1 = ((C_x - A_x) * (B_x - A_x) + (C_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_1 <= 0:
                            temp1 = AC
                        elif r_1 >= 1:
                            temp1 = BC
                        else:
                            tem = r_1 * AB
                            temp1 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 2
                        r_2 = ((D_x - A_x) * (B_x - A_x) + (D_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_2 <= 0:
                            temp2 = AD
                        elif r_2 >= 1:
                            temp2 = BD
                        else:
                            tem = r_2 * AB
                            temp2 = math.sqrt(max(10**(-4),AD * AD - tem * tem))    
                        #condition 3
                        r_3 = ((A_x - C_x) * (D_x - C_x) + (A_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_3 <= 0:
                            temp3 = AC
                        elif r_3 >= 1:
                            temp3 = AD
                        else:
                            tem = r_3 * CD
                            temp3 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 4
                        r_4 = ((B_x - C_x) * (D_x - C_x) + (B_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_4 <= 0:
                            temp4 = BC
                        elif r_4 >= 1:
                            temp4 = BD
                        else:
                            tem = r_4 * CD
                            temp4 = math.sqrt(max(10**(-4),BC * BC - tem * tem))
                        # minimum distance between two segments
                        temp = min(temp1, temp2, temp3, temp4)
                        # calculate the path safety for dynamic obstacles
                        if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p_out[x]!=0 and time_o<2:# find the obstacle which has the connection with the path
                            temp = 0.1
                            time_o += 1
                        #if temp<0.4:
                            #safety += 1/temp-2.5

                    # for the next moving obstacle
                    n += num_edge_m_out[i]

                # set initial value for n
                n = 0
                for i in range(num_edge_s_out.shape[0]):
                    for j in range(num_edge_s_out[i]):
                        n1 = (j+1)%num_edge_s_out[i]
                        #dist_out[x][y][i][j][k] = 0
                        #coodinates of points C, D
                        C_x = obstacles_s_out[n+j, 0]
                        C_y = obstacles_s_out[n+j, 1]
                        D_x = obstacles_s_out[n+n1, 0]
                        D_y = obstacles_s_out[n+n1, 1]
                        #length of CD,......
                        CD = math.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
                        AC = math.sqrt((A_x - C_x)**2 + (A_y - C_y)**2)
                        AD = math.sqrt((A_x - D_x)**2 + (A_y - D_y)**2)
                        BC = math.sqrt((B_x - C_x)**2 + (B_y - C_y)**2)
                        BD = math.sqrt((B_x - D_x)**2 + (B_y - D_y)**2)
                        #condition 1
                        r_1 = ((C_x - A_x) * (B_x - A_x) + (C_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_1 <= 0:
                            temp1 = AC
                        elif r_1 >= 1:
                            temp1 = BC
                        else:
                            tem = r_1 * AB
                            temp1 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 2
                        r_2 = ((D_x - A_x) * (B_x - A_x) + (D_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_2 <= 0:
                            temp2 = AD
                        elif r_2 >= 1:
                            temp2 = BD
                        else:
                            tem = r_2 * AB
                            temp2 = math.sqrt(max(10**(-4),AD * AD - tem * tem)) 
                        #condition 3
                        r_3 = ((A_x - C_x) * (D_x - C_x) + (A_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_3 <= 0:
                            temp3 = AC
                        elif r_3 >= 1:
                            temp3 = AD
                        else:
                            tem = r_3 * CD
                            temp3 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 4
                        r_4 = ((B_x - C_x) * (D_x - C_x) + (B_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_4 <= 0:
                            temp4 = BC
                        elif r_4 >= 1:
                            temp4 = BD
                        else:
                            tem = r_4 * CD
                            temp4 = math.sqrt(max(10**(-4),BC * BC - tem * tem))
                        # minimum distance between two segments
                        temp = min(temp1, temp2, temp3, temp4)
                        # calculate the path safety for static obstacles
                        if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p_out[x]!=0 and time_o<2:# find the obstacle which has the connection with the path
                            temp = 0.1
                            time_o += 1
                        if temp<0.2:
                            safety += 1/temp-5
                        
                    # for the next static obstacle
                    n += num_edge_s_out[i]
                    
        smoothness_out[x,y] = smoothness
        length_out[x,y] = time_o
        
        sum_risk = sum_risk/6
        fitness_value_out[x][y] = length/0.26 + smoothness/1.82 + safety + 10000 * intersection_value_out[x, y] + sum_risk


@cuda.jit('(float32[:, :, :], float32[:, :], int32[:], float32[:, :], int32[:], float32[:, :], int32[:], int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:], float32[:], float32[:])')
def fitness_obstacles(pop, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    if x < pop.shape[0] and y < pop.shape[1]:
        # lamda_m = 0.3
        # lamda_s = 0.2
        summ = 0
        safety = 0
        length = length_p_out[x]
        smoothness = smoothness_p_out[x]
        # dist = cuda.shared.array((9,64,2), dtype=types.float32)
        # a = np.zeros(2)
        #length_out[x,y] = 0
        #safety_out[x,y] = 0
        #smoothness_out[x,y] = 0
        intersection_value_out[x, y] = 1
        dx_1 = math.cos(init_dir_out[0])
        dy_1 = math.sin(init_dir_out[0])
    
        for z in range(int(pop.shape[2]/2) - 1):
            dx_2 = pop[x,y,2*z+2]-pop[x,y,2*z+0]
            dy_2 = pop[x,y,2*z+3]-pop[x,y,2*z+1]
            direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
            if dx_2!=0 or dy_2!=0:
                dx_1 = dx_2
                dy_1 = dy_2
            # if direction > math.pi/2:
            #     direction += 20
            smoothness += direction
            
            A_x = pop[x,y,2*z]
            A_y = pop[x,y,2*z+1]
            B_x = pop[x,y,2*z+2]
            B_y = pop[x,y,2*z+3]
            AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
            #min_value = 10000
            n = 0
            
            for i in range(num_edge_out.shape[0]):
                for j in range(num_edge_out[i]):
                    n1 = (j+1)%num_edge_out[i]
                
                    C_x = obstacles_out[n+j, 0]
                    C_y = obstacles_out[n+j, 1]
                    D_x = obstacles_out[n+n1, 0]
                    D_y = obstacles_out[n+n1, 1]
                    
                    j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                    j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                    j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                    j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)
    
                        
                    if j1*j2<=0 and j3*j4<=0 and (B_x-A_x)**2+(B_y-A_y)**2!=0:
                        summ += 1
                n += num_edge_out[i]
            length += AB
        if summ == 0:
            intersection_value_out[x, y] = 0
        dx_2 = math.cos(dir_final_out[x])
        dy_2 = math.sin(dir_final_out[x])
        direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
        if dir_final_out[x] == 1000:
            direction = 0
        smoothness += direction
        # if the path does not intersect with the obstacles, we need to calculate the shortest distance between segments of path and obtacles
        if intersection_value_out[x, y] == 0:
            for z in range(3):
                # set a large value for sub
                sub = np.inf
                # set initil value for counting the times (close to obstacle)
                time_o = 0
                # set initial value for n
                n = 0
                #coodinates of points A, B
                A_x = pop[x,y,2*z]
                A_y = pop[x,y,2*z+1]
                B_x = pop[x,y,2*z+2]
                B_y = pop[x,y,2*z+3]
                # length of AB
                AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
                for i in range(num_edge_m_out.shape[0]):
                    sub = np.inf
                    for j in range(num_edge_m_out[i]):
                        n1 = (j+1)%num_edge_m_out[i]
                        #dist_out[x][y][i][j][k] = 0
                        #coodinates of points C, D
                        C_x = obstacles_m_out[n+j, 0]
                        C_y = obstacles_m_out[n+j, 1]
                        D_x = obstacles_m_out[n+n1, 0]
                        D_y = obstacles_m_out[n+n1, 1]
                        #length of CD,......
                        CD = math.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
                        AC = math.sqrt((A_x - C_x)**2 + (A_y - C_y)**2)
                        AD = math.sqrt((A_x - D_x)**2 + (A_y - D_y)**2)
                        BC = math.sqrt((B_x - C_x)**2 + (B_y - C_y)**2)
                        BD = math.sqrt((B_x - D_x)**2 + (B_y - D_y)**2)
                        #condition 1
                        r_1 = ((C_x - A_x) * (B_x - A_x) + (C_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_1 <= 0:
                            temp1 = AC
                        elif r_1 >= 1:
                            temp1 = BC
                        else:
                            tem = r_1 * AB
                            # if AC * AC - tem * tem < 0:
                            #     temp1 = 0
                            # else:
                            temp1 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 2
                        r_2 = ((D_x - A_x) * (B_x - A_x) + (D_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_2 <= 0:
                            temp2 = AD
                        elif r_2 >= 1:
                            temp2 = BD
                        else:
                            tem = r_2 * AB
                            # if AD * AD - tem * tem < 0:
                            #     temp2 = 0
                            # else:
                            temp2 = math.sqrt(max(10**(-4),AD * AD - tem * tem))    
                        #condition 3
                        r_3 = ((A_x - C_x) * (D_x - C_x) + (A_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_3 <= 0:
                            temp3 = AC
                        elif r_3 >= 1:
                            temp3 = AD
                        else:
                            tem = r_3 * CD
                            # if AC * AC - tem * tem < 0:
                            #     temp3 = 0
                            # else:
                            temp3 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 4
                        r_4 = ((B_x - C_x) * (D_x - C_x) + (B_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_4 <= 0:
                            temp4 = BC
                        elif r_4 >= 1:
                            temp4 = BD
                        else:
                            tem = r_4 * CD
                            # if BC * BC - tem * tem < 0:
                            #     temp4 = 0
                            # else:
                            temp4 = math.sqrt(max(10**(-4),BC * BC - tem * tem))
                        # minimum distance between two segments
                        temp = min(temp1, temp2, temp3, temp4)
                        # find the closest obstacle
                        if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p_out[x]!=0:
                            time_o += 1
                        # obtain the shortest the distance
                        if sub > temp:
                            sub = temp
                    # for the next moving obstacle
                    n += num_edge_m_out[i]
                    if z==2 and length_p_out[x]!=0 and time_o==2:
                        # sub = 0.05
                        sub = 0.1
                        time_o += 1
                    if sub<0.5:
                        safety += 2/sub-4
                
                # calculate the safety value for the moving obstacles
                # safety += 2/sub
                # if sub<lamda_m:
                #     safety += math.exp(lamda_m-sub)*3
                # if sub<0.2:
                #     safety += 1/sub**2
                # else:
                #     safety += 1/sub
                # set a large value for sub
                sub = np.inf
                # set initial value for n
                n = 0
                for i in range(num_edge_s_out.shape[0]):
                    sub = np.inf
                    for j in range(num_edge_s_out[i]):
                        n1 = (j+1)%num_edge_s_out[i]
                        #dist_out[x][y][i][j][k] = 0
                        #coodinates of points C, D
                        C_x = obstacles_s_out[n+j, 0]
                        C_y = obstacles_s_out[n+j, 1]
                        D_x = obstacles_s_out[n+n1, 0]
                        D_y = obstacles_s_out[n+n1, 1]
                        #length of CD,......
                        CD = math.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
                        AC = math.sqrt((A_x - C_x)**2 + (A_y - C_y)**2)
                        AD = math.sqrt((A_x - D_x)**2 + (A_y - D_y)**2)
                        BC = math.sqrt((B_x - C_x)**2 + (B_y - C_y)**2)
                        BD = math.sqrt((B_x - D_x)**2 + (B_y - D_y)**2)
                        #condition 1
                        r_1 = ((C_x - A_x) * (B_x - A_x) + (C_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_1 <= 0:
                            temp1 = AC
                        elif r_1 >= 1:
                            temp1 = BC
                        else:
                            tem = r_1 * AB
                            # if AC * AC - tem * tem < 0:
                            #     temp1 = 0
                            # else:
                            temp1 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 2
                        r_2 = ((D_x - A_x) * (B_x - A_x) + (D_y - A_y) * (B_y - A_y))/(AB * AB)
                        if r_2 <= 0:
                            temp2 = AD
                        elif r_2 >= 1:
                            temp2 = BD
                        else:
                            tem = r_2 * AB
                            # if AD * AD - tem * tem < 0:
                            #     temp2 = 0
                            # else:
                            temp2 = math.sqrt(max(10**(-4),AD * AD - tem * tem)) 
                        #condition 3
                        r_3 = ((A_x - C_x) * (D_x - C_x) + (A_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_3 <= 0:
                            temp3 = AC
                        elif r_3 >= 1:
                            temp3 = AD
                        else:
                            tem = r_3 * CD
                            # if AC * AC - tem * tem < 0:
                            #     temp3 = 0
                            # else:
                            temp3 = math.sqrt(max(10**(-4),AC * AC - tem * tem))
                        #condition 4
                        r_4 = ((B_x - C_x) * (D_x - C_x) + (B_y - C_y) * (D_y - C_y))/(CD * CD)
                        if r_4 <= 0:
                            temp4 = BC
                        elif r_4 >= 1:
                            temp4 = BD
                        else:
                            tem = r_4 * CD
                            # if BC * BC - tem * tem < 0:
                            #     temp4 = 0
                            # else:
                            temp4 = math.sqrt(max(10**(-4),BC * BC - tem * tem))
                        # minimum distance between two segments
                        temp = min(temp1, temp2, temp3, temp4)
                        # find the closest obstacle
                        if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p_out[x]!=0:
                            time_o += 1
                        #assign the minimum to distance_out[x][i][j][0]
                    
                        # dist_out[x][y][i][j][k][0] = temp1
                        # dist_out[x][y][i][j][k][1] = temp2
                        # dist_out[x][y][i][j][k][2] = temp3
                        # dist_out[x][y][i][j][k][3] = temp4
                        # obtain the shortest the distance
                        if sub > temp:
                            sub = temp
                    # for the next static obstacle
                    n += num_edge_s_out[i]
                    if z==2 and length_p_out[x]!=0 and time_o==2:
                        # sub = 0.025
                        sub = 0.1
                        time_o += 1
                    if sub<0.2:
                        safety += 1/sub-5
                
                # calculate the safety value for the static obstacles
                # safety += 1/sub
                # if sub<lamda_s:
                #     safety += math.exp(lamda_s-sub)*3
                # if sub<0.2:
                #     safety += 0.5/sub**2
                # else:
                #     safety += 0.5/sub
            # calculate the safety value
            # for i in range(2):
            #     if dist[x,y,i]<lamda:
            #         safety += math.exp(lamda-dist[x,y,i])
            # safety = min(dist[0],dist[1])
            # distance_out[x][y][0] = sum
        smoothness_out[x,y] = smoothness
        length_out[x,y] = time_o
        
        ob_m_1 = obstacles_m_out[0:4]
        
        fitness_value_out[x][y] = length/0.26 + smoothness/1.82 + safety + 10000 * intersection_value_out[x, y]




@cuda.jit('(float32[:, :, :], float32[:, :], float32[:, :], float32[:, :, :])')
def selection(new_population_out, fitness_value_out, fitness_out, parents_out):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x, y = cuda.grid(2)
    if x < new_population_out.shape[0] and y < new_population_out.shape[1]:
        summ = 0
    
        for i in range(fitness_value_out.shape[1]):
            if fitness_value_out[x][i] < fitness_value_out[x][y] or fitness_value_out[x][i] == fitness_value_out[x][y] and i < y:
                summ += 1
        fitness_out[x][summ] = fitness_value_out[x][y]
        for j in range(new_population_out.shape[2]):
            parents_out[x][summ][j] = new_population_out[x][y][j]
        

@cuda.jit('(float32[:, :], int32, float32[:], int32[:])')
def selection2(fitness_out, generation, trend_out,order):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    x = cuda.grid(1)
    if x < fitness_out.shape[0]:
        summ = 0
    
        for i in range(fitness_out.shape[0]):
            if fitness_out[i][0] < fitness_out[x][0] or fitness_out[i][0] == fitness_out[x][0] and i < x:
                summ += 1
        if summ == 0:
            trend_out[generation] = fitness_out[x][0]
            order[0] = x
            

@cuda.jit('(float32[:, :, :], float32[:, :, :])')
def crossover(parents_out, offspring_out):

    # The point at which crossover takes place between two parents. Usually it is at the center.
    x, y = cuda.grid(2)
    if x < parents_out.shape[0] and y < parents_out.shape[1]:
        num_offspring = offspring_out.shape[1]
        crossover_point = 2
        # Index of the first parent to mate.
        parent1_idx = (2 * y)%num_offspring
        # Index of the second parent to mate.
        parent2_idx = (2 * y + 1)%num_offspring
        for i in range(crossover_point):
            # The first new offspring will have its first half of its genes taken from the first parent.
            offspring_out[x][2 * y][2 * i] = parents_out[x][parent1_idx][2 * i]
            offspring_out[x][2 * y][2 * i + 1] = parents_out[x][parent1_idx][2 * i + 1]
            # The first offspring will have its second half of its genes taken from the second parent.
            offspring_out[x][2 * y][2 * (crossover_point + i)] = parents_out[x][parent2_idx][2 * (crossover_point + i)]
            offspring_out[x][2 * y][2 * (crossover_point + i) + 1] = parents_out[x][parent2_idx][2 * (crossover_point + i) + 1]
            # The second offspring will have its first half of its genes taken from the first parent.
            offspring_out[x][2 * y + 1][2 * i] = parents_out[x][parent2_idx][2 * i]
            offspring_out[x][2 * y + 1][2 * i + 1] = parents_out[x][parent2_idx][2 * i + 1]
            # The second offspring will have its second half of its genes taken from the second parent.
            offspring_out[x][2 * y + 1][2 * (crossover_point + i)] = parents_out[x][parent1_idx][2 * (crossover_point + i)]
            offspring_out[x][2 * y + 1][2 * (crossover_point + i) + 1] = parents_out[x][parent1_idx][2 * (crossover_point + i) + 1]



@cuda.jit('(float32[:, :, :],  float32[:, :, :], float32[:, :, :])')
def new_popul(parents_out, offspring_out, new_population_out):
    x, y = cuda.grid(2)
    if x < parents_out.shape[0] and y < parents_out.shape[1]:
        num_offspring = offspring_out.shape[1]
        if y<num_offspring * 0.5:
            for i in range(parents_out.shape[2]):
                new_population_out[x][y][i] = parents_out[x][y][i]
        else:
            for i in range(parents_out.shape[2]):
                new_population_out[x][y][i] = offspring_out[x][y-int(num_offspring * 0.5)][i]



@cuda.jit#('(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :], int32[:], int32)')
def mutation_free(rng_states, new_population_out, obstacles_out, num_edge_out):
    # Mutation changes a single gene in each offspring randomly.
    idx, idy = cuda.grid(2)
    if idx < new_population_out.shape[0] and idy < new_population_out.shape[1]:
        # Thread id in a 2D block
        blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
        threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
        temp = 0.3

        c = 1
        if idy >= int(0.1 * new_population_out.shape[1]):
            for m in range(2, new_population_out.shape[2]-2):
                if xoroshiro128p_uniform_float32(rng_states, threadId)<0.1:
                    while c%2 == 1:
                        bx = m
                        if bx%2 == 0:
                            y = bx + 1
                            x = bx
                            nx = (new_population_out[idx,idy,x] + temp*xoroshiro128p_normal_float32(rng_states, threadId))%4.4
                            ny = new_population_out[idx,idy,y]
                        else:
                            y = bx
                            x = y - 1
                            nx = new_population_out[idx,idy,x]
                            ny = (new_population_out[idx,idy,y] + temp*xoroshiro128p_normal_float32(rng_states, threadId))%4.4
                        
                        n = 0
                        for i in range(num_edge_out.shape[0]):
                            c = 0
                            for j in range(num_edge_out[i]):
                                n1 = (j+1)%num_edge_out[i]
                                
                                A_x = obstacles_out[n+j, 0]
                                A_y = obstacles_out[n+j, 1]
                                B_x = obstacles_out[n+n1, 0]
                                B_y = obstacles_out[n+n1, 1]
    
                                if (((ny > A_y) != (ny > B_y)) and ((nx - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (ny - A_y)*(B_y - A_y))):
                                    c += 1
    
                            if c%2 == 1:
                                break
                            n += num_edge_out[i]
                    new_population_out[idx,idy,x] = nx
                    new_population_out[idx,idy,y] = ny      
    
@cuda.jit#('(float32[:, :, :], float32[:, :, :], float32[:, :, :], float32[:, :], int32[:], int32)')
def mutation_free_old(rng_states, new_population_out, obstacles_out, num_edge_out):
    # Mutation changes a single gene in each offspring randomly.
    idx, idy = cuda.grid(2)
    if idx < new_population_out.shape[0] and idy < new_population_out.shape[1]:
        # Thread id in a 2D block
        blockId = (cuda.gridDim.x * cuda.blockIdx.y) + cuda.blockIdx.x
        threadId = (blockId * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x
        temp = 1
        L_limit = 0.52
        c = 1
        if idy >= int(0.1 * new_population_out.shape[1]):
            for m in range(2, new_population_out.shape[2]-2):
                if xoroshiro128p_uniform_float32(rng_states, threadId)<0.1:
                    while c%2 == 1:
                        bx = m
                        if bx%2 == 0:
                            y = bx + 1
                            x = bx
                            nx = (new_population_out[idx,idy,x] + temp*xoroshiro128p_normal_float32(rng_states, threadId))%4.5
                            ny = new_population_out[idx,idy,y]
                        else:
                            y = bx
                            x = y - 1
                            nx = new_population_out[idx,idy,x]
                            ny = (new_population_out[idx,idy,y] + temp*xoroshiro128p_normal_float32(rng_states, threadId))%4.5
                        
                        if x == 2:
                            L = math.sqrt((nx-new_population_out[idx,idy,0])**2 + (ny-new_population_out[idx,idy,1])**2)
                            if L>L_limit:
                                nx = L_limit/L*(nx-new_population_out[idx,idy,0]) + new_population_out[idx,idy,0]
                                ny = L_limit/L*(ny-new_population_out[idx,idy,1]) + new_population_out[idx,idy,1]
                        n = 0
                        for i in range(num_edge_out.shape[0]):
                            c = 0
                            for j in range(num_edge_out[i]):
                                n1 = (j+1)%num_edge_out[i]
                                
                                A_x = obstacles_out[n+j, 0]
                                A_y = obstacles_out[n+j, 1]
                                B_x = obstacles_out[n+n1, 0]
                                B_y = obstacles_out[n+n1, 1]
    
                                if (((ny > A_y) != (ny > B_y)) and ((nx - A_x)*(B_y - A_y)*(B_y - A_y) < (B_x - A_x) * (ny - A_y)*(B_y - A_y))):
                                    c += 1
    
                            if c%2 == 1:
                                break
                            n += num_edge_out[i]
                    new_population_out[idx,idy,x] = nx
                    new_population_out[idx,idy,y] = ny      

