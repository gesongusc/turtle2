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

def dp_ga_planner(obstacle_s, num_edge_s, obstacle_m, num_edge_m, start_point, goal_point, init_dir):
    # expansion of obstacles
    L = -0.22
    
    obstacles_s = Extension.exten(obstacle_s,num_edge_s,L)
    obstacles_m = Extension.exten(obstacle_m,num_edge_m,L)
    obstacles = np.concatenate((obstacles_s,obstacles_m))
    num_edge = np.concatenate((num_edge_s,num_edge_m))
    
    # expansion of obstacles
    L = -0.001
    
    obstacles_o = Extension.exten(obstacles,num_edge,L)
    # all the points
    points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))
    # matrix initialization
    # initial direction of robot (unit: rad)
    init_dir_out = cuda.to_device(init_dir)
    obstacles_out = cuda.to_device(obstacles)
    obstacles_s_out = cuda.to_device(obstacles_s)
    obstacles_m_out = cuda.to_device(obstacles_m)
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
    inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
    dist = dist_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    length = np.inf*np.ones((N,N)).astype(np.float32)
    smoothness = np.inf*np.ones((N,N)).astype(np.float32)
    # fitness = np.ones((N,N)).astype(np.float32)
    length_out = cuda.to_device(length)
    length_update_out = cuda.to_device(length)
    smoothness_out = cuda.to_device(smoothness)
    smoothness_update_out = cuda.to_device(smoothness)
    num_generations = 20
    trend = np.zeros(num_generations+1).astype(np.float32)
    trend_out = cuda.to_device(trend)
    parent_node = -1*np.ones((N,N)).astype(np.int32)
    parent_node_out = cuda.to_device(parent_node)
    condition = 1
    while(condition>0):
        # inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
        updation.fitness_update[blocks_per_grid,threads_per_block](points_out, intersect_value_out, dist_out, parent_node_out, smoothness_out, smoothness_update_out, length_out, length_update_out, init_dir_out)
        updation.parameter_check[blocks_per_grid,threads_per_block](smoothness_out, smoothness_update_out, length_out, length_update_out, change_value_out)
        change_value= change_value_out.copy_to_host()
        condition = np.sum(change_value[:,0:N-1])
    length = length_out.copy_to_host()
    smoothness = smoothness_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    parent_node = parent_node_out.copy_to_host()
    
    # obtain all the potential paths
    # path_all = updation.path_all(path.copy(), points)
    paths_p, label_path, length_p, smoothness_p, dir_final = updation.path_part(parent_node, points, length, smoothness)
    dir_final_out = cuda.to_device(dir_final)
    length_p_out = cuda.to_device(length_p)
    smoothness_p_out = cuda.to_device(smoothness_p)

    # intialization for GA
    number_population = paths_p.shape[0]
    print("number",number_population)
    if number_population == 0:
        trend[20]=20000
        return obstacles_s[:,0], obstacles, num_edge, obstacles_s, obstacles_m, length_p,trend[20]

    number_of_genes = 8
    number_candidate = 64
    blocks_per_grid = (math.ceil(number_population/threads_per_block[0]), math.ceil(number_candidate/threads_per_block[1]))
    new_population = GA.pop_init(paths_p, number_candidate, number_of_genes)
    new_population_out = cuda.to_device(new_population)
    pop_out = cuda.to_device(paths_p)
    rng_states = create_xoroshiro128p_states(10 * number_of_genes * number_population * number_candidate, seed=1)
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

            

    order = np.zeros(1).astype(np.int32)
    order_out = cuda.to_device(order)
                        
    length_out = cuda.device_array_like(fitness_value_out)
    smoothness_out = cuda.device_array_like(fitness_value_out)


    for generation in range(num_generations):
        # checking if the path intersect the obstacles.
        GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
        # Selecting the best parents in the population for mating.
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
        # selecting the best individual
        #GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
        #crossover
        GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
        # Creating the new population based on the parents and offspring.
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        # Adding some variations to the offsrping using mutation.
        GA.mutation_free[blocks_per_grid, threads_per_block](rng_states, new_population_out, obstacles_out, num_edge_out)

    generation += 1
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
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
