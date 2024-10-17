#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:03:40 2022

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
import updation

def dp_planner(obstacle, num_edge, start_point, goal_point):
    # expansion of obstacles
    L = -0.25
    obstacles = Extension.exten(obstacle,num_edge,L)
    
    # expansion of obstacles
    L = -0.251
    obstacles_o = Extension.exten(obstacle,num_edge,L)
    # all the points
    points = np.concatenate((np.concatenate((start_point,obstacles_o)),goal_point))
    # matrix initialization
    obstacles_out = cuda.to_device(obstacles)
    points_out = cuda.to_device(points)
    N = points.shape[0]
    intersect_value = np.ones((N,N)).astype(np.int32)
    intersect_value_out = cuda.to_device(intersect_value)
    change_value = np.ones((N,N)).astype(np.int32)
    change_value_out = cuda.to_device(change_value)
    dist = np.zeros((N,N)).astype(np.float32)
    dist_out = cuda.to_device(dist)
    num_edge_out = cuda.to_device(num_edge)
    inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
    dist = dist_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    length = np.inf*np.ones((N,N)).astype(np.float32)
    smoothness = np.inf*np.ones((N,N)).astype(np.float32)
    # fitness = np.ones((N,N)).astype(np.float32)
    path = np.zeros((N,N,int(np.ceil(N/6)))).astype(np.int64)
    length_out = cuda.to_device(length)
    length_update_out = cuda.to_device(length)
    smoothness_out = cuda.to_device(smoothness)
    smoothness_update_out = cuda.to_device(smoothness)
    # fitness_out = cuda.to_device(fitness)
    path_out = cuda.to_device(path)
    path_update_out = cuda.to_device(path)
    exceptional = np.zeros((N,N)).astype(np.float64)
    exceptional_out = cuda.to_device(exceptional)
    condition = 1
    while(condition>0):
        # inters.intersection[(N,1),(1,N)](points_out, obstacles_out, num_edge_out, intersect_value_out, dist_out)
        updation.fitness_update[(N,1),(1,N)](points_out, intersect_value_out, dist_out, path_out, path_update_out, smoothness_out, smoothness_update_out, length_out, length_update_out, exceptional_out)
        # updation.parameter_update[(N,1),(1,N)](path_out, path_update_out, smoothness_out, smoothness_update_out, length_out, length_update_out)
        updation.parameter_check[(N,1),(1,N)](path_out, path_update_out, smoothness_out, smoothness_update_out, length_out, length_update_out, change_value_out)
        change_value= change_value_out.copy_to_host()
        condition = np.sum(change_value[:,0:N-1])
    length = length_out.copy_to_host()
    smoothness = smoothness_out.copy_to_host()
    intersect_value = intersect_value_out.copy_to_host()
    path = path_out.copy_to_host()
    exceptional = exceptional_out.copy_to_host()
    fitness =  0.9*length + 0.1*smoothness
    best_match_idx = np.where(fitness[:,0] == np.min(fitness[:,0]))
    path_optimal = path[best_match_idx[0][0]][0]
    path_index = np.zeros(1).astype(np.int64)
    j = 0
    
    while path_optimal[j] > 0:
        j += 1
    for i in range(j-1,-1,-1):
        while path_optimal[i] > 0:
            a=int(path_optimal[i]%1000)
            path_optimal[i] = (path_optimal[i]-path_optimal[i]%1000)/1000
            path_index = np.append(path_index,a)

    path = []
    for k in range(path_index.shape[0]):
        path = np.append(path,points[path_index[k]])
    return path

