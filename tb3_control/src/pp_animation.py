#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:14:29 2022

@author: junlinou
"""


#%%load module
import os
#use the specified gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import Extension
from matplotlib import pyplot as plt
import math
import random
import INTERSECTION as inters
import GA
# import Dij
import scipy.io as sio
import updation
from numba.cuda.random import create_xoroshiro128p_states
import obstacle_updation as OU
#%%obstacles configuration

##fig. representation
# # moving obstacles
# obstacle_m = np.float32([[3.7,3], [3.7, 2.9], [3.8, 2.9], [3.8, 3]])
# # obstacle_m = obstacle_m - [0.15,0.15]
# num_edge_m = np.int32([4]).astype(np.int32)
# # static obstacles
# obstacle_s = np.float32([[1.5, 2.7], [1.1, 1.9], [1.4, 1.8], [2.5,2.1]])
# # obstacle_s = obstacle_s - [0.15,0.15]
# num_edge_s = np.int32([4]).astype(np.int32)


# x_1 = 3.7
# y_1 = 0.4
# x_n = 2.9
# y_n = 4

## fig. 1
# # moving obstacles
# obstacle_m = np.float32([[0.3,4], [0.3, 3.9], [0.4, 3.9], [0.4, 4], [2.2,2.9], [2.2, 2.8], [2.3, 2.8], [2.3, 2.9], [2.2,1.8], [2.2, 1.7], [2.3, 1.7], [2.3, 1.8]])
# # obstacle_m = obstacle_m - [0.15,0.15]
# num_edge_m = np.int32([4, 4, 4]).astype(np.int32)
# # static obstacles
# obstacle_s = np.float32([[0.8,2.8], [0.8, 2.1], [1.2, 2.1], [1.2, 2.3], [1.0, 2.3], [1.0, 2.8],
#                         [1.2,1.0], [1.2, 0.5], [3, 0.5], [3, 0.7], [1.4, 0.7], [1.4, 1.0],
#                         [3.2,2.6], [3.2, 2.4], [3.4, 2.4], [3.4, 1.6], [3.6, 1.6], [3.6, 2.6],
#                         [1.2,4.0], [1.2, 3.8], [3, 3.8], [3, 3.6], [3.2, 3.6], [3.2, 4.0]])
# # obstacle_s = obstacle_s - [0.15,0.15]
# num_edge_s = np.int32([6, 6, 6, 6]).astype(np.int32)

# x_1 = 0.3
# y_1 = 1.465
# x_n = 4
# y_n = 4


## fig. 2
# # moving obstacles
# obstacle_m = np.float32([[1.5,3.1], [1.5, 3.0], [1.6, 3.0], [1.6, 3.1], [3.3,2.1], [3.3, 2.0], [3.4, 2.0], [3.4, 2.1]])
# # obstacle_m = obstacle_m - [0.15,0.15]
# num_edge_m = np.int32([4, 4]).astype(np.int32)
# # static obstacles
# obstacle_s = np.float32([[1.2,3.9], [1.2, 3.6], [1.4, 3.8], [1.4, 3.9],
#                         [0.8,1.1], [0.8, 0.9], [1.0, 0.9], [1.0, 1.3],
#                         [2.4,3.9], [2.4, 3.2], [2.6, 3.4], [2.6, 3.9],
#                         [2.4,1.8], [2.4, 1.6], [2.2, 1.6], [2.2, 1.4], [2.8, 1.4], [2.8, 1.6], [2.6, 1.6], [2.6, 2.0],
#                         [3.6,3.3], [3.6, 3.0], [3.8, 3.2], [3.8, 3.3], 
#                         [3.6,0.9], [3.6, 0.7], [3.8, 0.7], [3.8, 1.1]])
# # obstacle_s = obstacle_s - [0.15,0.15]
# num_edge_s = np.int32([4, 4, 4, 8, 4, 4]).astype(np.int32)


# x_1 = 0.2
# y_1 = 1.2
# x_n = 4.0
# y_n = 2.5


## fig. 3
# # moving obstacles
# obstacle_m = np.float32([[1.7,2.7], [1.7, 2.6], [1.8, 2.6], [1.8, 2.7], [2.7,1.8], [2.7, 1.7], [2.8, 1.7], [2.8, 1.8]])
# num_edge_m = np.int32([4, 4]).astype(np.int32)
# # static obstacles
# obstacle_s = np.float32([[0.6,2.2], [0.6, 0.4], [2.8, 0.4], [2.8, 0.6], [0.8, 0.6], [0.8, 2.2],
#                         [1.5,1.9], [1.5, 1.5], [2.0, 1.5], [2.0, 1.7], [1.7, 1.7], [1.7, 1.9],
#                         [2.5,2.9], [2.5, 2.7], [2.7, 2.7], [2.7, 2.5], [2.9, 2.5], [2.9, 2.9],
#                         [1.4,4.0], [1.4, 3.8], [3.6, 3.8], [3.6, 2.6], [3.8, 2.6], [3.8, 4.0]])
# num_edge_s = np.int32([6, 6, 6, 6]).astype(np.int32)

# x_1 = 0.1
# y_1 = 3.5
# x_n = 4.2
# y_n = 1.2


## fig. 4
# moving obstacles
obstacle_m = np.float32([[1.7,2.7], [1.7, 2.6], [1.8, 2.6], [1.8, 2.7], [2.7,1.8], [2.7, 1.7], [2.8, 1.7], [2.8, 1.8]])
num_edge_m = np.int32([4, 4]).astype(np.int32)
# static obstacles
obstacle_s = np.float32([[2.2,4.0], [1.8, 3.6], [2.2, 3.2], [2.6, 3.6],
                        [1.0,2.6], [0.6, 2.2], [1.0, 1.8], [1.4, 2.2],
                        [3.4,2.6], [3.0, 2.2], [3.4, 1.8], [3.8, 2.2],
                        [2.2,1.2], [1.8, 0.8], [2.2, 0.4], [2.6, 0.8]])
num_edge_s = np.int32([4, 4, 4, 4]).astype(np.int32)

x_1 = 0.1
y_1 = 2.0
x_n = 4.2
y_n = 1.2



fitness_p = np.inf
path_p = []
# set up the parameters for the robot
start_point = np.float32([[x_1,y_1]])
goal_point = np.float32([[x_n,y_n]])
orientation = -np.random.uniform(-1, 1)*np.pi
# max velocity
velocity_max = 0.20
# set the intial value for time_nav
time_nav = 0
# set the time_limit for velocity change (it means the velocity would change within 5s)
time_limit = np.zeros(num_edge_m.shape[0]) * 5
# assign the initial value for velocity and theta
velocity = velocity_max*np.zeros(num_edge_m.shape[0])
theta = 2*math.pi*np.zeros(num_edge_m.shape[0])
length_o = np.float32(0)
# initial direction of robot (unit: rad)
init_dir = np.zeros(1).astype(np.float32)


for m in range(1):
    # theta =np.pi/45
    # R = np.float32([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)],])
    # temp=np.dot(obstacle-[50,50],R)
    # obstacle = temp + [50,50]
    
    obstacle_m, time_limit, velocity, theta = OU.dynamic_obstacle(obstacle_s, num_edge_s, obstacle_m, num_edge_m, velocity_max, time_limit, time_nav, velocity, theta, goal_point)
    start = timer()
    
    # expansion of static and dynamic obstacles
    L = -0.22
    obstacle = np.concatenate((obstacle_s,obstacle_m))
    obstacles_s = Extension.exten(obstacle_s,num_edge_s,L)
    obstacles_m = Extension.exten(obstacle_m,num_edge_m,L)
    # all obstacles
    obstacles = np.concatenate((obstacles_s,obstacles_m))
    num_edge = np.concatenate((num_edge_s,num_edge_m))
    # expansion of obstacles
    L = -0.001
    obstacles_o = Extension.exten(obstacles,num_edge,L)
    # all the points
    points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))
    # matrix initialization for dynamic programming
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
    # edge number
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
        # print(9)
        # print(condition)
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
    num_generations = 20
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
        GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
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
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, obstacles_m_out, num_edge_m_out, obstacles_s_out, num_edge_s_out, intersection_value_out, length_out, smoothness_out, fitness_value_out, init_dir_out, dir_final_out, length_p_out, smoothness_p_out)
    # Selecting the best parents in each population for mating.
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    # select the best individual in all populations
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    trend = trend_out.copy_to_host()
    order = order_out.copy_to_host()
    parents = parents_out.copy_to_host()
    length = length_out.copy_to_host()
    smoothness = smoothness_out.copy_to_host()
    tem = order[0]
    best_individual = parents[tem][0]
    print(trend[num_generations])
    # if length[tem,0] > 60:
    #     break
    # print(length[tem,0])
    # print(smoothness[tem,0])
    path_o = updation.path_one(label_path[tem], parent_node, points, best_individual)
    
    print(length_o)
    if len(path_p)>0:
        fitness_p = updation.fitness_one(path_p[0], obstacles, num_edge, obstacles_m, num_edge_m, obstacles_s, num_edge_s, init_dir, length_o)
        if trend[20]<round(fitness_p,6):
            path_p = []
            path_p.append(path_o)
            best_individual = path_o
            length_o = length_p[tem]
            print(2)
        
        else:
            best_individual = path_p[0]
            print(1)
    else:
        path_p.append(path_o)
        best_individual = path_o
        length_o = length_p[tem]
        
    
    # j = 0
    # while path_optimal[j] > 0:
    #     j += 1
    # for i in range(j-1,-1,-1):
    #     while path_optimal[i] > 0:
    #         a=int(path_optimal[i]%1000)
    #         path_optimal[i] = (path_optimal[i]-path_optimal[i]%1000)/1000
    #         path_index = np.append(path_index,a)
    time_nav = timer() - start
    print(time_nav)

    plt.plot(best_individual[range(0,len(best_individual),2)],best_individual[range(1,len(best_individual),2)])
    # plt.plot(path_o[range(0,len(path_o),2)],path_o[range(1,len(path_o),2)])
    # for i in range(parents.shape[0]):
    #     plt.plot(parents[i,0,range(0,8,2)],parents[i,0,range(1,8,2)])
    # for i in range(len(path_all)):
    #     plt.plot(path_all[i][range(0,path_all[i].shape[0],2)],path_all[i][range(1,path_all[i].shape[0],2)])
    
    # for i in range(parents.shape[0]):
    #     plt.plot(parents[i,0,range(0,8,2)],parents[i,0,range(1,8,2)])
    # for i in range(parents.shape[0]):
    #     for j in range(parents.shape[1]):
    #         plt.plot(parents[i,j,range(0,8,2)],parents[i,j,range(1,8,2)])
    # for i in range(len(paths_p)):
    #     plt.plot(paths_p[i][range(0,paths_p[i].shape[0],2)],paths_p[i][range(1,paths_p[i].shape[0],2)])
    # j = 0
    # while path_optimal[j]>=N-1:
    #     a=int(path_optimal[j]%1000)
    #     path_optimal[j] = (path_optimal[j]-path_optimal[j]%1000)/1000
    #     if path_optimal[j] == 0:
    #         j+=1
    #     path_index = np.append(path_index,a)
    
    filename = 'fig_1.mat'
    # sio.savemat(filename, {'points':points,'intersection':intersect_value,'path_all_1':path_all[0],'path_all_2':path_all[1],'path_all_3':path_all[2],'path_all_4':path_all[3],'path_all_5':path_all[4],'path_all_6':path_all[5],'path_part':paths_p,'path_o_all':parents,'best':best_individual})
    # sio.savemat(filename, {'best':best_individual})
    # sio.savemat(filename, {'path_optimized':parents[:,0,:],'path_all':path_all,'path_part':paths_p, 'best_individual':best_individual})
    # plt.figure("optimal path")
    n = 0
    for i in range(num_edge.shape[0]):
        plt.plot([obstacles[n+num_edge[i]-1,0],obstacles[n,0]], [obstacles[n+num_edge[i]-1,1],obstacles[n,1]],c="k")
        plt.plot(obstacles[n:n+num_edge[i],0], obstacles[n:n+num_edge[i],1],c="k")
        plt.plot([obstacles_o[n+num_edge[i]-1,0],obstacles_o[n,0]], [obstacles_o[n+num_edge[i]-1,1],obstacles_o[n,1]],c="k")
        plt.plot(obstacles_o[n:n+num_edge[i],0], obstacles_o[n:n+num_edge[i],1],c="k")
        n += num_edge[i]
    n = 0
    for i in range(num_edge.shape[0]):
        plt.plot([obstacle[n+num_edge[i]-1,0],obstacle[n,0]], [obstacle[n+num_edge[i]-1,1],obstacle[n,1]],c="k")
        plt.plot(obstacle[n:n+num_edge[i],0], obstacle[n:n+num_edge[i],1],c="k")
        n += num_edge[i]
    plt.xlim(0, 4.4)
    plt.ylim(0, 4.4)
    # plt.xlim(-20, 120)
    # plt.ylim(-20, 120)
    # plt.plot(points[path_index,0], points[path_index,1],c="k")
    plt.pause(0.01)
plt.show()
