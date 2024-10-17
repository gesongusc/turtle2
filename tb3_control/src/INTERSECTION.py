#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:13:14 2022

@author: junlinou
"""

from numba import cuda
import math
from numba import jit
import numpy as np

@cuda.jit('(float32[:, :], float32[:, :], int32[:], int32[:, :], float32[:, :])')
def intersection(points_out, obstacles_out, num_edge_out, intersection_value_out, dist_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    if x < points_out.shape[0] and y < points_out.shape[0]:
        summ = 0
        n = 0
        
        intersection_value_out[x, y] = 1
        A_x = points_out[x, 0]*10
        A_y = points_out[x, 1]*10
        B_x = points_out[y, 0]*10
        B_y = points_out[y, 1]*10
        distAB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
        for i in range(num_edge_out.shape[0]):
            for j in range(num_edge_out[i]):
                n1 = (j+1)%num_edge_out[i]
                
                C_x = obstacles_out[n+j, 0]*10
                C_y = obstacles_out[n+j, 1]*10
                D_x = obstacles_out[n+n1, 0]*10
                D_y = obstacles_out[n+n1, 1]*10
                j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)
    
                if j1*j2<=0 and j3*j4<=0:
                    summ += 1
                
            n += num_edge_out[i]
        if summ == 0 and x!= y:
            intersection_value_out[x, y] = 0
            dist_out[x,y] = distAB
        
#@jit('(float32[:, :], float32[:, :], int32[:])')
def intersection_cpu(points, obstacles, num_edge):
    
    N = points.shape[0]
    intersection_value = np.ones((N,N)).astype(np.int32)
    dist = np.zeros((N,N)).astype(np.float32)

    for x in range(N):
        for y in range(N):
            summ = 0
            n = 0
            intersection_value[x, y] = 1
            A_x = points[x, 0]*10
            A_y = points[x, 1]*10
            B_x = points[y, 0]*10
            B_y = points[y, 1]*10
            distAB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
            for i in range(num_edge.shape[0]):
                for j in range(num_edge[i]):
                    n1 = (j+1)%num_edge[i]
                    
                    C_x = obstacles[n+j, 0]*10
                    C_y = obstacles[n+j, 1]*10
                    D_x = obstacles[n+n1, 0]*10
                    D_y = obstacles[n+n1, 1]*10
                    j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                    j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                    j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                    j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)
        
                    if j1*j2<=0 and j3*j4<=0:
                        summ += 1
                    
                n += num_edge[i]
            if summ == 0 and x!= y:
                intersection_value[x, y] = 0
                dist[x,y] = distAB
    return intersection_value, dist

@cuda.jit('(float32[:, :, :], float32[:, :], int32[:], int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def intersect(pop, obstacles_out, num_edge_out, intersection_value_out, length_out, safety_out, smoothness_out, fitness_value_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    #lamda = 1.5
    summ = 0
    length = 0
    safety = 0
    smoothness = 0
    #length_out[x,y] = 0
    #safety_out[x,y] = 0
    #smoothness_out[x,y] = 0
    intersection_value_out[x, y] = 1
    dx_1 = pop[x,y,2]-pop[x,y,0]
    dy_1 = pop[x,y,3]-pop[x,y,1]
    dir_1 = math.atan2(dy_1,dx_1)
    for z in range(int(pop.shape[2]/2) - 1):
        if z < int(pop.shape[2]/2) - 2:
            dx_2 = pop[x,y,2*z+4]-pop[x,y,2*z+2]
            dy_2 = pop[x,y,2*z+5]-pop[x,y,2*z+3]
            dir_2 = math.atan2(dy_2,dx_2)
            direction = 180 * abs(dir_2-dir_1)/math.pi
            temp = dir_1
            dir_1 =dir_2
            if direction>180:
                direction = 360 - direction
            if dx_2==0 and dy_2==0:
                direction = 0
                dir_1 = temp
            smoothness += direction
        A_x = pop[x,y,2*z]*10
        A_y = pop[x,y,2*z+1]*10
        B_x = pop[x,y,2*z+2]*10
        B_y = pop[x,y,2*z+3]*10
        AB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
        #min_value = 10000
        n = 0
        
        for i in range(num_edge_out.shape[0]):
            for j in range(num_edge_out[i]):
                n1 = (j+1)%num_edge_out[i]
            
                C_x = obstacles_out[n+j, 0]*10
                C_y = obstacles_out[n+j, 1]*10
                D_x = obstacles_out[n+n1, 0]*10
                D_y = obstacles_out[n+n1, 1]*10
                
                j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)

                #length of CD,......
                #CD = math.sqrt((C_x - D_x) * (C_x - D_x) + (C_y - D_y) * (C_y - D_y))
                #AC = math.sqrt((A_x - C_x) * (A_x - C_x) + (A_y - C_y) * (A_y - C_y))
                #AD = math.sqrt((A_x - D_x) * (A_x - D_x) + (A_y - D_y) * (A_y - D_y))
                #BC = math.sqrt((B_x - C_x) * (B_x - C_x) + (B_y - C_y) * (B_y - C_y))
                #BD = math.sqrt((B_x - D_x) * (B_x - D_x) + (B_y - D_y) * (B_y - D_y))

                    
                if j1*j2<=0 and j3*j4<=0 and (B_x-A_x)**2+(B_y-A_y)**2!=0:
                    summ += 1
            n += num_edge_out[i]
        length += AB
    if summ == 0:
        intersection_value_out[x, y] = 0
    safety_out[x,y] = safety
    smoothness_out[x,y] = smoothness
    length_out[x,y] = length
    
    fitness_value_out[x][y] = 0.1 * length + 0.9 * smoothness + 0.4 * safety + 1000 * intersection_value_out[x, y]

@cuda.jit('(float32[:, :, :], float32[:, :], int32[:], int32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def preintersect(pop, obstacles_out, num_edge_out, intersection_value_out, length_out, safety_out, smoothness_out, fitness_value_out, fitness_out):
    # 2 D coordinates
    x, y = cuda.grid(2)
    #lamda = 1.5
    summ = 0
    length = 0
    safety = 0
    smoothness = 0
    #length_out[x,y] = 0
    #safety_out[x,y] = 0
    #smoothness_out[x,y] = 0
    intersection_value_out[x, y] = 1
    dx_1 = pop[x,y,2]-pop[x,y,0]
    dy_1 = pop[x,y,3]-pop[x,y,1]
    dir_1 = math.atan2(dy_1,dx_1)
    for z in range(int(pop.shape[2]/2) - 1):
        if z < int(pop.shape[2]/2) - 2:
            dx_2 = pop[x,y,2*z+4]-pop[x,y,2*z+2]
            dy_2 = pop[x,y,2*z+5]-pop[x,y,2*z+3]
            dir_2 = math.atan2(dy_2,dx_2)
            direction = 180 * abs(dir_2-dir_1)/math.pi
            temp = dir_1
            dir_1 =dir_2
            if direction>180:
                direction = 360 - direction
            if dx_2==0 and dy_2==0:
                direction = 0
                dir_1 = temp
            smoothness += direction
        A_x = pop[x,y,2*z]*10
        A_y = pop[x,y,2*z+1]*10
        B_x = pop[x,y,2*z+2]*10
        B_y = pop[x,y,2*z+3]*10
        AB = math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))/10
        #min_value = 10000
        n = 0
        
        for i in range(num_edge_out.shape[0]):
            for j in range(num_edge_out[i]):
                n1 = (j+1)%num_edge_out[i]
            
                C_x = obstacles_out[n+j, 0]*10
                C_y = obstacles_out[n+j, 1]*10
                D_x = obstacles_out[n+n1, 0]*10
                D_y = obstacles_out[n+n1, 1]*10

                j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)

                #length of CD,......
                #CD = math.sqrt((C_x - D_x) * (C_x - D_x) + (C_y - D_y) * (C_y - D_y))
                #AC = math.sqrt((A_x - C_x) * (A_x - C_x) + (A_y - C_y) * (A_y - C_y))
                #AD = math.sqrt((A_x - D_x) * (A_x - D_x) + (A_y - D_y) * (A_y - D_y))
                #BC = math.sqrt((B_x - C_x) * (B_x - C_x) + (B_y - C_y) * (B_y - C_y))
                #BD = math.sqrt((B_x - D_x) * (B_x - D_x) + (B_y - D_y) * (B_y - D_y))

                if j1*j2<=0 and j3*j4<=0:
                    summ += 1
                
            n += num_edge_out[i]

        length += AB
    if summ == 0:
        intersection_value_out[x, y] = 0
    safety_out[x,y] = safety
    smoothness_out[x,y] = smoothness
    length_out[x,y] = length
    
    fitness_value_out[x][y] = 0.5 * length + 0.5 * smoothness + 0.4 * safety + 1000 * intersection_value_out[x, y]
    fitness_out[x][y] = fitness_value_out[x][y]


