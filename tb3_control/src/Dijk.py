#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 09:43:26 2021

@author: junlinou
"""
import numpy as np
from numba import jit
import math
import random

#from the website: https://stackoverflow.com/questions/54930852/python-numba-value-in-array
@jit(nopython=True)
def isin(val, arr):
    for i in range(len(arr)):
        if arr[i] == val:
            return True
    return False

#from statistics import median
@jit(nopython=True)
def minDistance(dist,queue):
    # Initialize min value and min_index as -1
    minimum = math.inf
    min_index = -1
          
    # from the dist array,pick one which
    # has min value and is till in queue
    for i in range(len(dist)):
        if dist[i] < minimum and isin(i,queue):
            minimum = dist[i]
            min_index = i
    return min_index

# Function to print shortest path
# from source to j
# using parent array
#@jit(nopython=True)
#def getPath(path, parent, j):
#    pa = path
#    while j != 0:
#        path = np.append(pa,j)
#        j = parent[j]
#        pa = path
#    path = np.append(path,0)

#    return path
@jit(nopython=True)
def dijkstra(graph, src, tar):
  
    row = graph.shape[0]
    col = row
    path = np.ones(0).astype(np.int64)
    # The output array. dist[i] will hold
    # the shortest distance from src to i
    # Initialize all distances as INFINITE 
    dist = math.inf * np.ones(row)
  
    #Parent array to store 
    # shortest path tree
    parent = -1 * np.ones(row).astype(np.int64)
  
    # Distance of source vertex 
    # from itself is always 0
    dist[src] = 0
    # Add all vertices in queue
    queue = np.arange(row).astype(np.int64)
    
    u = -1
    #Find shortest path for all vertices
    while u != tar:
  
        # Pick the minimum dist vertex
        # from the set of vertices
        # still in queue
        u = minDistance(dist,queue)
        #print(u)
        # remove min element
        if u == tar:
            break
        if u < 0:
            tar = 0
            break
        queue = queue[queue != u]
  
        # Update dist value and parent 
        # index of the adjacent vertices of
        # the picked vertex. Consider only 
        # those vertices which are still in
        # queue
        for i in range(col):
            '''Update dist[i] only if it is in queue, there is
            an edge from u to i, and total weight of path from
            src to i through u is smaller than current value of
            dist[i]'''
            if graph[u][i]!=0 and isin(i,queue):
                if dist[u] + graph[u][i] < dist[i]:
                    dist[i] = dist[u] + graph[u][i]
                    parent[i] = u
    j = tar
    pa = path
    while j != 0:
        path = np.append(pa,j)
        j = parent[j]
        pa = path
    path = np.append(path,0)
    #path = getPath(path, parent, tar)
    return path

@jit(nopython=True)
def paths(dist, src, tar, num):
    path = []
    num_p = np.zeros(num).astype(np.int64)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1 
            path.append(pa)
            num_p[i] = len(pa)
            i += 1
    return path, num_p


@jit(nopython=True)
def med_paths(dist, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1
            path_feasible.append(pa)
            num_p[i] = len(pa)
            i += 1
    num_median = np.median(num_p)
    #num_median = 11
    for i in range(len(path_feasible)-1,-1,-1):
        if len(path_feasible[i])>num_median:
            path_feasible.pop(i)

    path = np.zeros((len(path_feasible),int(2*num_median))).astype(np.float32)
    for i in range(len(path_feasible)):
        r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_median-len(path_feasible[i])))])
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0;
        n = 0;
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==s[k]):
                    path[i,2*n] = (points[path_feasible[i][len(path_feasible[i])-j-1],0]+points[path_feasible[i][len(path_feasible[i])-j-2],0])/2
                    path[i,2*n+1] = (points[path_feasible[i][len(path_feasible[i])-j-1],1]+points[path_feasible[i][len(path_feasible[i])-j-2],1])/2
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, num_p

@jit(nopython=True)
def gauss_paths(dist, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1
            path_feasible.append(pa)
            num_p[i] = len(pa)
            i += 1
    num_mean = np.mean(num_p)
    num_std = np.std(num_p)
    num_d = max(num_mean + num_std,len(path_feasible[0]))
    for i in range(len(path_feasible)-1,-1,-1):
        if len(path_feasible[i])>num_d:
            path_feasible.pop(i)

    path = np.zeros((len(path_feasible),2*int(num_d))).astype(np.float32)
    for i in range(len(path_feasible)):
        r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_d-len(path_feasible[i])))])
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0;
        n = 0;
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==s[k]):
                    path[i,2*n] = (points[path_feasible[i][len(path_feasible[i])-j-1],0]+points[path_feasible[i][len(path_feasible[i])-j-2],0])/2
                    path[i,2*n+1] = (points[path_feasible[i][len(path_feasible[i])-j-1],1]+points[path_feasible[i][len(path_feasible[i])-j-2],1])/2
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, num_p

@jit(nopython=True)
def gauss_paths_r(dist, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1
            path_feasible.append(pa)
            num_p[i] = len(pa)
            i += 1
    num_mean = np.mean(num_p)
    num_std = np.std(num_p)
    num_d = num_mean + num_std
    for i in range(len(path_feasible)-1,-1,-1):
        if len(path_feasible[i])>num_d:
            path_feasible.pop(i)

    path = np.zeros((len(path_feasible),2*int(num_d))).astype(np.float32)
    for i in range(len(path_feasible)):
        #r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_d-len(path_feasible[i])))])
        r = np.random.rand(int(num_d-len(path_feasible[i]))).astype(np.float32)*(len(path_feasible[i])-2)/10000000000000000000*10000000000000000000
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0;
        n = 0;
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==int(s[k])):
                    path[i,2*n] = (100*(s[k]-j)*(points[path_feasible[i][len(path_feasible[i])-j-1],0]-points[path_feasible[i][len(path_feasible[i])-j-2],0])+100*points[path_feasible[i][len(path_feasible[i])-j-2],0])/100
                    path[i,2*n+1] = (100*(s[k]-j)*(points[path_feasible[i][len(path_feasible[i])-j-1],1]-points[path_feasible[i][len(path_feasible[i])-j-2],1])+100*points[path_feasible[i][len(path_feasible[i])-j-2],1])/100
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, num_p

@jit(nopython=True)
def smo_paths(dist, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1 
            path_feasible.append(pa)
            num_p[i] = len(pa)
            i += 1
    num_median = np.median(num_p)
    
    for i in range(len(path_feasible)-1,-1,-1):
        if len(path_feasible[i])>num_median:
            path_feasible.pop(i)

    path = np.zeros((len(path_feasible),int(2*num_median))).astype(np.float32)
    for i in range(len(path_feasible)):
        r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_median-len(path_feasible[i])))])
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0;
        n = 0;
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==s[k]):
                    path[i,2*n] = (points[path_feasible[i][len(path_feasible[i])-j-1],0]+points[path_feasible[i][len(path_feasible[i])-j-2],0])/2
                    path[i,2*n+1] = (points[path_feasible[i][len(path_feasible[i])-j-1],1]+points[path_feasible[i][len(path_feasible[i])-j-2],1])/2
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, num_p

@jit(nopython=True)
def med_fitness_paths(dist, src, tar, num, points):
    path_feasible = []
    num_p = np.zeros(num).astype(np.int64)
    fitness_p = np.zeros(num).astype(np.float32)
    p = 0
    i = 0
    while(i<num):
        #if i == 1:
        #    p = 0.8
        dist1 = dist.copy()
        for m in range(len(dist)):
            for n in range(m,len(dist)):
                if random.random() < p:
                    dist1[m,n] = 0
                    dist1[n,m] = 0
        pa = dijkstra(dist1,src,tar)
        if pa[0] == 0:
            p -= 0.1
        else:
            p += 0.1
            path_feasible.append(pa)
            num_p[i] = len(pa)
            fitness_p[i] = fitness(pa,points)
            i += 1
    fitness_median = np.median(fitness_p)
    #num_median = 11
    for i in range(len(path_feasible)-1,-1,-1):
        if fitness_p[i]>fitness_median:
            path_feasible.pop(i)
            num_p = np.delete(num_p, i)
            fitness_p = np.delete(fitness_p, i)
    fitness_median = np.median(fitness_p)
    #num_median = 11
    for i in range(len(path_feasible)-1,-1,-1):
        if fitness_p[i]>fitness_median:
            path_feasible.pop(i)
            num_p = np.delete(num_p, i)
            fitness_p = np.delete(fitness_p, i)
    num_point = max(num_p)
    path = np.zeros((len(path_feasible),int(2*num_point))).astype(np.float32)
    for i in range(len(path_feasible)):
        r = np.array([random.randint(0, len(path_feasible[i])-2) for k in range(int(num_point-len(path_feasible[i])))])
        #r = np.random.randint(len(path_feasible[i])-1, size=int(num_median-len(path_feasible[i])))
        s =  np.sort(r)
        k = 0;
        n = 0;
        for j in range(len(path_feasible[i])):
            path[i,2*n] = points[path_feasible[i][len(path_feasible[i])-j-1],0]
            path[i,2*n+1] = points[path_feasible[i][len(path_feasible[i])-j-1],1]
            n += 1
            if k < len(s):
                while(j==s[k]):
                    path[i,2*n] = (points[path_feasible[i][len(path_feasible[i])-j-1],0]+points[path_feasible[i][len(path_feasible[i])-j-2],0])/2
                    path[i,2*n+1] = (points[path_feasible[i][len(path_feasible[i])-j-1],1]+points[path_feasible[i][len(path_feasible[i])-j-2],1])/2
                    n += 1
                    k += 1
                    if k >= len(s):
                        break
    return path, fitness_p

@jit(nopython=True)
def fitness(path,points):
    smoothness = 0
    length = 0
    for i in range(path.shape[0] - 1):
        if i < path.shape[0] - 2:
            dir_1 = math.atan2(points[path[i+1],1]-points[path[i],1],points[path[i+1],0]-points[path[i],0])
            dir_2 = math.atan2(points[path[i+2],1]-points[path[i+1],1],points[path[i+2],0]-points[path[i+1],0])
            direction = 180 * abs(dir_2-dir_1)/math.pi
            if direction>180:
                direction = 360 - direction
            smoothness += direction
        A_x = points[path[i],0]
        A_y = points[path[i],1]
        B_x = points[path[i+1],0]
        B_y = points[path[i+1],1]
        length += math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))
    return (0.5*length + 0.5*smoothness)

@jit(nopython=True)
def fitness_2(path):
    smoothness = 0
    length = 0
    for i in range(int(path.shape[0]/2) - 1):
        if i < int(path.shape[0]/2) - 2:
            dir_1 = math.atan2(path[2*i+3]-path[2*i+1],path[2*i+2]-path[2*i])
            dir_2 = math.atan2(path[2*i+5]-path[2*i+3],path[2*i+4]-path[2*i+2])
            direction = 180 * abs(dir_2-dir_1)/math.pi
            if direction>180:
                direction = 360 - direction
            smoothness += direction
        A_x = path[2*i]
        A_y = path[2*i+1]
        B_x = path[2*i+2]
        B_y = path[2*i+3]
        length += math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))
    return (0.5*length + 0.1*smoothness)

@jit(nopython=True)
def fitness_1(path):
    smoothness = 0
    length = 0
    dx_1 = path[2]-path[0]
    dy_1 = path[3]-path[1]
    dir_1 = math.atan2(dy_1,dx_1)
    for i in range(int(path.shape[0]/2) - 1):
        if i < int(path.shape[0]/2) - 2:
            dx_2 = path[2*i+4]-path[2*i+2]
            dy_2 = path[2*i+5]-path[2*i+3]
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
        A_x = path[2*i]
        A_y = path[2*i+1]
        B_x = path[2*i+2]
        B_y = path[2*i+3]
        length += math.sqrt((B_x-A_x) * (B_x-A_x) + (B_y-A_y) * (B_y-A_y))
    return (0.5*length + 0.1*smoothness)