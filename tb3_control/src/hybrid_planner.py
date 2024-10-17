#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:14:48 2023

@author: ouj
"""

#%%load module
# import os
#use the specified gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from numba import cuda
import Extension
import INTERSECTION as inters
import Astar
from numba.cuda.random import create_xoroshiro128p_states
import GA

def hybrid_planner(obstacle, num_edge, start_point, goal_point):
    number_population = 64
    number_candidate = 32
    num_generations = 200
    threads_per_block = (8, 8)#even number
    blocks_per_grid = (int(number_population/threads_per_block[0]), int(number_candidate/threads_per_block[1]))

    #generate population
    L = -1.8
    obstacles = Extension.exten(obstacle,num_edge,L)

    L = -1.801
    obstacles_o = Extension.exten(obstacle,num_edge,L)
    points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))
    
    obstacles_out = cuda.to_device(obstacles)
    points_out = cuda.to_device(points)
    N = points.shape[0]
    intersect_value = np.ones((N,N)).astype(np.int32)
    intersect_value_out = cuda.to_device(intersect_value)
    dist = np.zeros((N,N)).astype(np.float32)
    dist_out = cuda.to_device(dist)
    h = np.zeros(N).astype(np.float32)
    h_out = cuda.to_device(h)
    num_edge_out = cuda.to_device(num_edge)
    inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out, h_out)
    dist = dist_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    h = h_out.copy_to_host()

    num_path = 8 * number_population
    path, num_p = Astar.med_fitness_paths(dist, h, 0, N-1, num_path, points)
    if path[0,0] == 0:
        return path[0]
    way_points = int(path.shape[1]/2)
    number_of_genes = 2 * way_points

    
    pop = path[0:number_population][:]
    pop_out = cuda.to_device(pop)
    #pop_new = GA_tenc.population_path_free(pop, obstacles, number_population, number_candidate, number_of_genes)
    new_population = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    new_population[:, :, 0] = start_point[0,0]
    new_population[:, :, 1] = start_point[0,1]
    new_population[:, :, number_of_genes - 2] = goal_point[0,0]
    new_population[:, :,number_of_genes - 1] = goal_point[0,1]
    rng_states = create_xoroshiro128p_states(number_population * number_candidate, seed=1)

    new_population_out = cuda.to_device(new_population)

    GA.population_path_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, rng_states, pop_out)
    cuda.synchronize()
    #matrix initialization
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
    best_individual = np.zeros((number_of_genes)).astype(np.float32)
    order = np.zeros(1).astype(np.int32)
    order_out = cuda.to_device(order)
                
    length_out = cuda.device_array_like(fitness_value_out)
    smoothness_out = cuda.device_array_like(fitness_value_out)
    safety_out = cuda.device_array_like(fitness_value_out)
    

    
    #new_population = new_population_out.copy_to_host()
    #

    for generation in range(num_generations):
        # checking if the path intersect the obstacles.
        #start = timer()
        GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, safety_out, fitness_value_out)
        #time_ga = timer() - start
        # Selecting the best parents in the population for mating.
        #start = timer()
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
        #trend_out[generation] = fitness_value_out[0]
        #time_ga = timer() - start
        #print('Time taken for 5 is %f seconds.' % time_ga)
        #start = timer()
        GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
        #time_ga = timer() - start
        #print('Time taken for 6 is %f seconds.' % time_ga)
        #start = timer()
        GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
        #time_ga = timer() - start
        #print('Time taken for 7 is %f seconds.' % time_ga)
        # Creating the new population based on the parents and offspring.
        #start = timer()
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        #time_ga = timer() - start
        #print('Time taken for 8 is %f seconds.' % time_ga)
        # Adding some variations to the offsrping using mutation.
        #start = timer()
        GA.mutation_free[blocks_per_grid, threads_per_block](rng_states, new_population_out, obstacles_out, num_edge_out, generation)
        #GA_tenc.mutation[blocks_per_grid, threads_per_block](random_normal_out, random_int_out, new_population_out, generation)
        #time_ga = timer() - start
        #print('Time taken for 9 is %f seconds.' % time_ga)
        #start = timer()
        GA.migration[blocks_per_grid, threads_per_block](new_population_out, parents_out, generation)
        #time_ga = timer() - start
        #print('Time taken for 10 is %f seconds.' % time_ga)
        # Getting the best solution after iterating finishing all generations.
        #At first, the fitness is calculated for each solution in the final generation.


    #
    generation += 1
    GA.fitness[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, smoothness_out, safety_out, fitness_value_out)
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    trend = trend_out.copy_to_host()
    order = order_out.copy_to_host()
    parents = parents_out.copy_to_host()
    tem = order[0]
    best_individual = parents[tem][0]
    return best_individual*4.4/100
