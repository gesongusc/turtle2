#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:09:51 2021

@author: junlinou
"""


#%%load module
import numpy as np
import GA
from numba import cuda
import Extension
import INTERSECTION as inters
import Dijk


def GA_planner(obstacle,num_edge,L,start_point,goal_point):
    obstacles = Extension.exten(obstacle,num_edge,L)
    num = 98-obstacles.shape[0]
    free_points = GA.points_free(obstacles, num_edge, num)
    L = L - 0.001
    obstacles_o = Extension.exten(obstacle,num_edge,L)
    points = np.concatenate((np.concatenate((start_point,np.concatenate((obstacles_o,free_points)))),goal_point))
    obstacles_out = cuda.to_device(obstacles)
    points_out = cuda.to_device(points)
    N = points.shape[0]
    intersect_value = np.ones((N,N)).astype(np.int32)
    intersect_value_out = cuda.to_device(intersect_value)
    dist = np.zeros((N,N)).astype(np.float32)
    dist_out = cuda.to_device(dist)
    num_edge_out = cuda.to_device(num_edge)
    inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
    dist = dist_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    num_path = 256
    path, num_p = Dijk.med_fitness_paths(dist, 0, N-1, num_path, points)
    num_generations = 100
    way_points = int(path.shape[1]/2)
    number_of_genes = 2 * way_points
    number_population = 64
    number_candidate = 32
    threads_per_block = (8, 8)#even number
    blocks_per_grid = (int(number_population/threads_per_block[0]), int(number_candidate/threads_per_block[1]))
    pop = path[0:number_population,:]
    pop_out = cuda.to_device(pop)
    new_population = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    new_population[:, :, 0] = start_point[0,0]
    new_population[:, :, 1] = start_point[0,1]
    new_population[:, :, number_of_genes - 2] = goal_point[0,0]
    new_population[:, :,number_of_genes - 1] = goal_point[0,1]
    random = np.float32(np.random.normal(0, 3, (number_population, number_candidate, 2)))
    random_out = cuda.to_device(random)
    new_population_out = cuda.to_device(new_population)
    GA.population_path_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, random_out, pop_out)
    random = np.float32(np.random.normal(0, 3, (number_population, number_candidate, 2)))
    random_out = cuda.to_device(random)
    new_population_out = cuda.to_device(new_population)
    GA.population_path_free_G[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, random_out, pop_out)
    #matrix initialization
    intersection_value = np.ones((number_population, number_candidate)).astype(np.int32)
    intersection_value_out = cuda.to_device(intersection_value)

    fitness = np.zeros((number_population, number_candidate)).astype(np.float32)
    fitness_value_out = cuda.to_device(fitness)
    fitness_out = cuda.device_array_like(fitness_value_out)
    parents = np.zeros((number_population, number_candidate, number_of_genes)).astype(np.float32)
    parents_out = cuda.to_device(parents)
    offspring_out = cuda.device_array_like(parents_out)
    random_normal = np.float32(np.random.normal(0, 1, (number_population, number_candidate)))
    random_normal_out = cuda.to_device(random_normal)
    random_int = np.int32(np.random.randint(2, number_of_genes - 2, (number_population, number_candidate)))
    random_int_out = cuda.to_device(random_int)

    trend = np.zeros(num_generations).astype(np.float32)
    trend_out = cuda.to_device(trend)
    best_individual = np.zeros((number_of_genes)).astype(np.float32)
    order = np.zeros(1).astype(np.int32)
    order_out = cuda.to_device(order)
            
    length_out = cuda.device_array_like(fitness_value_out)
    smoothness_out = cuda.device_array_like(fitness_value_out)
    safety_out = cuda.device_array_like(fitness_value_out)
    for generation in range(num_generations):
        # checking if the path intersect the obstacles.
        inters.intersect[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, safety_out, smoothness_out, fitness_value_out)
        # Selecting the best parents in the population for mating.
        GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)

        GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
        #crossover
        GA.crossover[blocks_per_grid, (threads_per_block[0], int(threads_per_block[1]/2))](parents_out, offspring_out)
        # Creating the new population based on the parents and offspring.
        GA.new_popul[blocks_per_grid, threads_per_block](parents_out, offspring_out,new_population_out)
        # Adding some variations to the offsrping using mutation.
        GA.mutation_free[blocks_per_grid, threads_per_block](random_normal_out, random_int_out, new_population_out, obstacles_out, num_edge_out, generation)
        #GA.mutation[blocks_per_grid, threads_per_block](random_normal_out, random_int_out, new_population_out, generation)
        GA.migration[blocks_per_grid, threads_per_block](new_population_out, parents_out, generation)
        # caculating the fitness of each chromosome in the final population.
        inters.intersect[blocks_per_grid, threads_per_block](new_population_out, obstacles_out, num_edge_out, intersection_value_out, length_out, safety_out, smoothness_out, fitness_value_out)
    GA.selection[blocks_per_grid, threads_per_block](new_population_out, fitness_value_out, fitness_out, parents_out)
    GA.selection2[blocks_per_grid[0], threads_per_block[0]](fitness_out, generation, trend_out,order_out)
    order = order_out.copy_to_host()

    tem = order[0]
    best_individual = parents_out[tem][0].copy_to_host()
    return best_individual
