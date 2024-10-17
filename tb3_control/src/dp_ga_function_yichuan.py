#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:55:16 2022

@author: ouj
"""

#%%load module
# import os
#use the specified gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import rospy
from numba import cuda
import Extension
import INTERSECTION as inters
import updation
import GA
import math
from numba.cuda.random import create_xoroshiro128p_states
#from sklearn.preprocessing import MinMaxScaler

#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#from torch.utils.data import Dataset, DataLoader
#from sklearn.model_selection import train_test_split

def dp_ga_planner(obstacle_s, num_edge_s, obstacle_m, num_edge_m, start_point, goal_point, init_dir):
    # expansion of obstacles
    
    #start_fur_init = np.zeros(((forecast_horizon+1), 2)).astype(np.float32)
    
    #ob_m_vars = {}
    
    #for i in range(0, len(obstacle_m), 4):
    #    globals()[f'ob_m_{i//4 + 1}'] = obstacle_m[i:i+4]
    ob_m_1 = obstacle_m[0,4]
    ob_m_2 = obstacle_m[4,8]
    ob_m_3 = obstacle_m[8,12]
    ob_m_4 = obstacle_m[12,16]
    ob_m_5 = obstacle_m[16,20]
    ob_m_6 = obstacle_m[20,24]
    ob_m_7 = obstacle_m[24,28]
    ob_m_8 = obstacle_m[28,32]
    ob_m_9 = obstacle_m[32,36]
    ob_m_10 = obstacle_m[36,40]
    
    L = -0.22
    num_edge_p = np.int32([4]).astype(np.int32)
    
    #for i in range(1, int(len(obstacle_m)/4+1)):
    #    var_name = f'ob_m_{i}'
    #    globals()[var_name] = Extension.exten(globals()[var_name],num_edge_p,L)
    ob_m_1 = Extension.exten(ob_m_1,num_edge_p,L)
    ob_m_2 = Extension.exten(ob_m_2,num_edge_p,L)
    ob_m_3 = Extension.exten(ob_m_3,num_edge_p,L)
    ob_m_4 = Extension.exten(ob_m_4,num_edge_p,L)
    ob_m_5 = Extension.exten(ob_m_5,num_edge_p,L)
    ob_m_6 = Extension.exten(ob_m_6,num_edge_p,L)
    ob_m_7 = Extension.exten(ob_m_7,num_edge_p,L)
    ob_m_8 = Extension.exten(ob_m_8,num_edge_p,L)
    ob_m_9 = Extension.exten(ob_m_9,num_edge_p,L)
    ob_m_10 = Extension.exten(ob_m_10,num_edge_p,L)
    
    
    obstacles_s = Extension.exten(obstacle_s,num_edge_s,L)

    obstacles_m = obstacle_m
    #obstacles_m = np.vstack((ob_m_1, ob_m_3, ob_m_5, ob_m_7, ob_m_9, ob_m_12, ob_m_14, ob_m_16, ob_m_18, ob_m_20))

    num_edge_m  = np.int32([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]).astype(np.int32)

    num_edge_all_m_out = cuda.to_device(num_edge_m)
    
    obstacles = np.concatenate((obstacles_s,obstacles_m))
    num_edge = np.concatenate((num_edge_s,num_edge_m))

    L = -0.001
    obstacles_o = Extension.exten(obstacles,num_edge,L)

    points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))

    init_dir_out = cuda.to_device(init_dir)

    obstacles_m_all_out = cuda.to_device(obstacles_m)
    obstacles_out = cuda.to_device(obstacles)
    ob_m_1_out = cuda.to_device(ob_m_1)
    ob_m_2_out = cuda.to_device(ob_m_2)
    ob_m_3_out = cuda.to_device(ob_m_3)
    ob_m_4_out = cuda.to_device(ob_m_4)
    ob_m_5_out = cuda.to_device(ob_m_5)
    
    ob_m_6_out = cuda.to_device(ob_m_6)
    ob_m_7_out = cuda.to_device(ob_m_7)
    ob_m_8_out = cuda.to_device(ob_m_8)
    ob_m_9_out = cuda.to_device(ob_m_9)
    ob_m_10_out = cuda.to_device(ob_m_10)
    
    start_point_out = cuda.to_device(start_point)
    obstacles_s_out = cuda.to_device(obstacles_s)
    
    points_out = cuda.to_device(points)

    N = points.shape[0]
    threads_per_block = (8, 8)#even number
    blocks_per_grid = (math.ceil(N/threads_per_block[0]), math.ceil(N/threads_per_block[1]))
    intersect_value = np.ones((N,N)).astype(np.int32)
    intersect_value_out = cuda.to_device(intersect_value)
    change_value = np.ones((N,N)).astype(np.int32)
    change_value_out = cuda.to_device(change_value)
    dist = np.zeros((N,N)).astype(np.float32)
    dist_out = cuda.to_device(dist)

    num_edge_out = cuda.to_device(num_edge)
    num_edge_s_out = cuda.to_device(num_edge_s)
    num_edge_m_out = cuda.to_device(num_edge_m)

    inters.intersection[blocks_per_grid,threads_per_block](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
    dist = dist_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    length = np.inf*np.ones((N,N)).astype(np.float32)
    smoothness = np.inf*np.ones((N,N)).astype(np.float32)
    # fitness = np.ones((N,N)).astype(np.float32)
    length_out = cuda.to_device(length)
    length_update_out = cuda.to_device(length)
    smoothness_out = cuda.to_device(smoothness)
    smoothness_update_out = cuda.to_device(smoothness)
    parent_node = -1*np.ones((N,N)).astype(np.int32)
    parent_node_out = cuda.to_device(parent_node)
    # fitness_out = cuda.to_device(fitness)    
    condition = 1
    
    while(condition>0):
        # inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
        updation.fitness_update[blocks_per_grid,threads_per_block](points_out, intersect_value_out, dist_out, parent_node_out, smoothness_out, smoothness_update_out, length_out, length_update_out, init_dir_out)
        # updation.parameter_update[(N,1),(1,N)](path_out, path_update_out, smoothness_out, smoothness_update_out, length_out, length_update_out)
        updation.parameter_check[blocks_per_grid,threads_per_block](smoothness_out, smoothness_update_out, length_out, length_update_out, change_value_out)
        change_value= change_value_out.copy_to_host()
        condition = np.sum(change_value[:,0:N-1])    

    length = length_out.copy_to_host()
    smoothness = smoothness_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    parent_node = parent_node_out.copy_to_host()

    # fitness =  length/0.26 + smoothness/1.82
    # fitness_min=np.min(fitness[:,0])
    # best_match_idx = np.where(fitness[:,0] == np.min(fitness[:,0]))
    # path_optimal = path[best_match_idx[0][0]][0]
    # path_index = np.zeros(1).astype(np.int64)
    
    # obtain all the potential paths
    path_all = updation.path_all(parent_node.copy(), points)

    paths_p, label_path, length_p, smoothness_p, dir_final = updation.path_part(parent_node, points, length, smoothness)
    length_p_out = cuda.to_device(length_p)
    smoothness_p_out = cuda.to_device(smoothness_p)
    dir_final_out = cuda.to_device(dir_final)
    # intialization for GA
    
    number_population = paths_p.shape[0]
    number_of_genes = 8
    number_candidate = 64
    num_generations = 16
    blocks_per_grid = (math.ceil(number_population/threads_per_block[0]), math.ceil(number_candidate/threads_per_block[1]))
    new_population = GA.pop_init(paths_p, number_candidate, number_of_genes)
    
    new_population_out = cuda.to_device(new_population)
    pop_out = cuda.to_device(paths_p)
    rng_states = create_xoroshiro128p_states(number_of_genes*number_population * number_candidate, seed=1)
    # initialize the population
    GA.population_path_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, rng_states, pop_out)
    new_population = new_population_out.copy_to_host()
    
    # matrix initialization for GA
    intersection_value = np.ones((number_population, number_candidate)).astype(np.int32)
    intersection_value_out = cuda.to_device(intersection_value)
    #popul = np.empty((number, number_of_genes))
    fitness = np.zeros((number_population, number_candidate)).astype(np.float32)
    fitness_value_out = cuda.to_device(fitness)
    fitness_out = cuda.device_array_like(fitness_value_out)
    parents = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    parents_out = cuda.to_device(parents)
    offspring_out = cuda.device_array_like(parents_out)

    trend = np.zeros(num_generations+1).astype(np.float32)
    trend_out = cuda.to_device(trend)
    order = np.zeros(1).astype(np.int32)
    order_out = cuda.to_device(order)
                        
    length_out = cuda.device_array_like(fitness_value_out)
    smoothness_out = cuda.device_array_like(fitness_value_out)

    for generation in range(num_generations):
        # checking if the path intersect the obstacles and caculating the fitness values
        GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out,  obstacles_m_all_out, num_edge_all_m_out, ob_m_2_out, ob_m_3_out, ob_m_4_out,ob_m_5_out, num_edge_out, ob_m_1_out, ob_m_6_out, ob_m_7_out, ob_m_8_out, ob_m_9_out, ob_m_10_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
        # Selecting the best parents in the population for mating.
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
        # selecting the best individual
        GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
        # crossover
        GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
        # Creating the new population based on the parents and offspring.
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        # Adding some variations to the offsrping using mutation.
        GA.mutation_free[blocks_per_grid, threads_per_block](rng_states, new_population_out, obstacles_out, num_edge_out)


    generation += 1

    # checking if the path intersect the obstacles and caculating the fitness values
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out,  obstacles_m_all_out, num_edge_all_m_out, ob_m_2_out, ob_m_3_out, ob_m_4_out,ob_m_5_out, num_edge_out, ob_m_1_out, ob_m_6_out, ob_m_7_out, ob_m_8_out, ob_m_9_out, ob_m_10_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
    # Selecting the best parents in each population for mating.
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    # select the best individual in all populations
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    trend = trend_out.copy_to_host()
    order = order_out.copy_to_host()
    parents = parents_out.copy_to_host()
    tem = order[0]
    best_individual = parents[tem][0]
    path_o = updation.path_one(label_path[tem], parent_node, points, best_individual)

    return path_o, obstacles, num_edge, obstacles_s, obstacles_m, length_p[tem],trend[20]
