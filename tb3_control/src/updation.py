#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:49:48 2022

@author: junlinou
"""
from numba import cuda, jit, njit
import numpy as np
import math

@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def fitness_update(points_out, intersect_value_out, dist_out, parent_node_out, smoothness_out, smoothness_update_out, length_out, length_update_out, init_dir_out):
    x, y = cuda.grid(2)
    N = intersect_value_out.shape[0]
    if x < N and y < N:
        
        l = np.inf
        s = np.inf
        fitness = np.inf
        index = 0
        if intersect_value_out[x][y] ==0:
            for i in range(N):
                if intersect_value_out[y][i] == 0:
                    #calculate the angle
                    dx_1 = points_out[y][0]-points_out[x][0]
                    dy_1 = points_out[y][1]-points_out[x][1]
                    dx_2 = points_out[i][0]-points_out[y][0]
                    dy_2 = points_out[i][1]-points_out[y][1]
                    direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
                    
                    # calculate a segment length
                    AB = math.sqrt(dx_1**2 + dy_1**2)

                    if y == N-1:
                        direction = 0
                        smoothness_out[x][y] = 0
                        length_out[x][y] = AB
                        smoothness_update_out[x][y] = 0
                        length_update_out[x][y] = AB
                        smoothness = direction
                        length = AB

                    else:
                        smoothness = smoothness_update_out[y][i]+direction
                        length = length_update_out[y][i]+AB
                    temp = length/0.26 + smoothness/1.82
                    if temp < fitness:
                        fitness = temp
                        l = length
                        s = smoothness
                        parent = i
                        index = 1
    
        length_out[x][y] = l
        smoothness_out[x][y] = s
        if index == 1:
            parent_node_out[x][y] = parent



@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def parameter_update(path_out, path_update_out, smoothness_out, smoothness_update_out, length_out, length_update_out):
    x, y = cuda.grid(2)
    for j in range(path_update_out.shape[2]):
        path_update_out[x][y][j] = path_out[x][y][j]
    length_update_out[x][y] = length_out[x][y]
    smoothness_update_out[x][y] = smoothness_out[x][y]

@cuda.jit#('(float32[:, :, :], float32[:, :], int32[:], float32[:, :, :], float32[:, :])')
def parameter_check(smoothness_out, smoothness_update_out, length_out, length_update_out, change_value_out):
    x, y = cuda.grid(2)
    if x < length_update_out.shape[0] and y < length_update_out.shape[0]:
        if (length_update_out[x][y] == length_out[x][y] )and (smoothness_update_out[x][y] == smoothness_out[x][y]):
            change_value_out[x][y] = 0
        else:
            change_value_out[x][y] = 1
            length_update_out[x][y] = length_out[x][y]
            smoothness_update_out[x][y] = smoothness_out[x][y]
@njit
def fitness_one(path, obstacles, num_edge, obstacles_m, num_edge_m, obstacles_s, num_edge_s, init_dir, length_p):
    smoothness = 0
    length = 0
    safety = 0
    summ = 0
    intersection_value = 1
    dx_1 = math.cos(init_dir[0])
    dy_1 = math.sin(init_dir[0])
    
    
    for z in range(int(path.shape[0]/2) - 1):
        dx_2 = path[2*z+2]-path[2*z+0]
        dy_2 = path[2*z+3]-path[2*z+1]
        direction = abs(math.atan2(dy_2*dx_1-dx_2*dy_1,dx_1*dx_2+dy_1*dy_2))
        if dx_2!=0 or dy_2!=0:
            dx_1 = dx_2
            dy_1 = dy_2
        # if direction > math.pi/2:
        #     direction += 20
        smoothness += direction
        
        A_x = path[2*z]
        A_y = path[2*z+1]
        B_x = path[2*z+2]
        B_y = path[2*z+3]
        AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
        #min_value = 10000
        n = 0
        
        for i in range(num_edge.shape[0]):
            for j in range(num_edge[i]):
                n1 = (j+1)%num_edge[i]
            
                C_x = obstacles[n+j, 0]
                C_y = obstacles[n+j, 1]
                D_x = obstacles[n+n1, 0]
                D_y = obstacles[n+n1, 1]
                
                j1 = (B_x-A_x)*(C_y-A_y)-(B_y-A_y)*(C_x-A_x)
                j2 = (B_x-A_x)*(D_y-A_y)-(B_y-A_y)*(D_x-A_x)
                j3 = (D_x-C_x)*(A_y-C_y)-(D_y-C_y)*(A_x-C_x)
                j4 = (D_x-C_x)*(B_y-C_y)-(D_y-C_y)*(B_x-C_x)

                    
                if j1*j2<=0 and j3*j4<=0 and (B_x-A_x)**2+(B_y-A_y)**2!=0:
                    summ += 1
            n += num_edge[i]
        length += AB
    if summ == 0:
        intersection_value = 0

    # if the path does not intersect with the obstacles, we need to calculate the shortest distance between segments of path and obtacles
    if intersection_value == 0:
        for z in range(3):
            # set a large value for sub
            sub = np.inf
            # set initil value for counting the times (close to obstacle)
            time_o = 0
            # set initial value for n
            n = 0
            #coodinates of points A, B
            A_x = path[2*z]
            A_y = path[2*z+1]
            B_x = path[2*z+2]
            B_y = path[2*z+3]
            # length of AB
            AB = math.sqrt((B_x-A_x)**2 + (B_y-A_y)**2)
            for i in range(num_edge_m.shape[0]):
                sub = np.inf
                for j in range(num_edge_m[i]):
                    n1 = (j+1)%num_edge_m[i]
                    #coodinates of points C, D
                    C_x = obstacles_m[n+j, 0]
                    C_y = obstacles_m[n+j, 1]
                    D_x = obstacles_m[n+n1, 0]
                    D_y = obstacles_m[n+n1, 1]
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
                    if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p!=0:
                        time_o += 1
                    #assign the minimum to distance_out[x][i][j][0]
                
                    # dist_out[x][y][i][j][k][0] = temp1
                    # dist_out[x][y][i][j][k][1] = temp2
                    # dist_out[x][y][i][j][k][2] = temp3
                    # dist_out[x][y][i][j][k][3] = temp4
                    # obtain the shortest the distance
                    if sub > temp:
                        sub = temp
                # for the next moving obstacle
                n += num_edge_m[i]
                if z==2 and length_p!=0 and time_o==2:
                    sub = 0.05
                    time_o += 1
                if sub<0.5:
                    safety += 4/sub
            
            # calculate the safety value for the moving obstacles
            # safety += 2/sub

            # set a large value for sub
            sub = np.inf
            # set initial value for n
            n = 0
            for i in range(num_edge_s.shape[0]):
                sub = np.inf
                for j in range(num_edge_s[i]):
                    n1 = (j+1)%num_edge_s[i]
                    #dist_out[x][y][i][j][k] = 0
                    #coodinates of points C, D
                    C_x = obstacles_s[n+j, 0]
                    C_y = obstacles_s[n+j, 1]
                    D_x = obstacles_s[n+n1, 0]
                    D_y = obstacles_s[n+n1, 1]
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
                    if (BC<10**(-2) or BD<10**(-2)) and z==2 and length_p!=0:
                        time_o += 1

                    # obtain the shortest the distance
                    if sub > temp:
                        sub = temp
                # for the next static obstacle
                n += num_edge_s[i]
                if z==2 and length_p!=0 and time_o==2:
                    sub = 0.025
                    time_o += 1
                if sub<0.2:
                    safety += 1/sub
            
            # calculate the safety value for the static obstacles
            # safety += 1/sub
    fitness_value = length/0.26 + smoothness/1.82 + safety + 10000 * intersection_value
    
    return fitness_value


# @jit
def path_all(parent_node, points):
    path_all =[]
    N = parent_node.shape[0]
    for k in range(N):
        j = 0
        if parent_node[j,k] >= 0:
            path_index = np.zeros(1).astype(np.int32)
            while k != N-1:
                path_index = np.append(path_index,k)
                temp = j
                j = k
                k = parent_node[temp,k]

            path_index = np.append(path_index,k)
            path_v = []
            for m in range(path_index.shape[0]):
                path_v = np.append(path_v,points[path_index[m]])
            path_all.append(path_v)
    return path_all

@njit
def path_part(parent_node, points, length, smoothness):
    paths_p = np.zeros((0,8)).astype(np.float32)
    length_p = np.zeros(0).astype(np.float32)
    smoothness_p = np.zeros(0).astype(np.float32)
    label_path = np.zeros(0).astype(np.int64)
    dir_final = np.zeros(0).astype(np.float32)
    N = parent_node.shape[0]
    for k in range(N):
        j = 0
        if parent_node[j,k] >= 0:
            label_path = np.append(label_path,k)
            path_index = np.zeros(1).astype(np.int64)
            while k != N-1:
                path_index = np.append(path_index,k)
                temp = j
                j = k
                k = parent_node[temp,k]

            path_index = np.append(path_index,k)
            path_one = np.zeros((1,8)).astype(np.float32)
            path_one[0,0:2] = points[0]
            if path_index.shape[0] == 2:
                point_m = (points[0]*3/4 + points[-1]*1/4)
                #assign the value to each waypoint
                path_one[0,2:4] = point_m
                path_one[0,4:6] = (points[0] + points[-1])/2
                path_one[0,6:8] = points[-1]
                length_p = np.append(length_p,np.float32(0))
                smoothness_p = np.append(smoothness_p,np.float32(0))
                # 1000 represent no orientation requirement
                dir_final = np.append(dir_final,np.float32(1000))
            if path_index.shape[0] == 3:
                a = path_index[1]
                # obtain first waypoint
                point_m = (points[0] + points[a])/2
                #assign the value to each waypoint
                path_one[0,2:4] = point_m
                path_one[0,4:6] = points[a]
                path_one[0,6:8] = points[-1]
                length_p = np.append(length_p,np.float32(0))
                smoothness_p = np.append(smoothness_p,np.float32(0))
                # 1000 represent no orientation requirement
                dir_final = np.append(dir_final,np.float32(1000))
            if path_index.shape[0] == 4:
                for m in range(4):
                    path_one[0,2*m:2*m+2] = points[path_index[m]]
                length_p = np.append(length_p,np.float32(0))
                smoothness_p = np.append(smoothness_p,np.float32(0))
                # 1000 represent no orientation requirement
                dir_final = np.append(dir_final,np.float32(1000))
            if path_index.shape[0] > 4:
                for m in range(4):
                    path_one[0,2*m:2*m+2] = points[path_index[m]]
                length_p = np.append(length_p,length[path_index[3],path_index[4]])
                smoothness_p = np.append(smoothness_p,smoothness[path_index[3],path_index[4]])
                dx = points[path_index[4],0]-points[path_index[3],0]
                dy = points[path_index[4],1]-points[path_index[3],1]
                dir_final = np.append(dir_final,math.atan2(dy,dx)) 

            paths_p=np.append(paths_p,path_one,axis=0)
    return paths_p, label_path, length_p, smoothness_p, dir_final


@njit
def path_one(path_select, parent_node, points, best_individual):
    j = 0
    k = path_select
    N = parent_node.shape[0]
    path_index = np.zeros(1).astype(np.int64)
    while k != N-1:
        path_index = np.append(path_index,k)
        temp = j
        j = k
        k = parent_node[temp,k]

    path_index = np.append(path_index,k)
    if path_index.shape[0] > 4:
        for i in range(int(path_index.shape[0]-4)):
            best_individual = np.append(best_individual,points[path_index[i+4]]) 
    return best_individual

@njit
def paths_part_new_old(path, points, length, smoothness):
    # counting the number of available paths and the order of availble paths
    
    label_path = np.zeros(1).astype(np.int64)
    for i in range(path.shape[0]):
        if path[i,0,0] > 0:
            label_path = np.append(label_path,i)
    label_path = np.delete(label_path, 0)
    M = label_path.shape[0]
    paths_p = np.zeros((M,8)).astype(np.float32)
    length_p = np.zeros(M).astype(np.float32)
    smoothness_p = np.zeros(M).astype(np.float32)
    dir_final = np.zeros(M).astype(np.float32)
    paths_p[:,0:2] = points[0]
    for k in range(M):
        if path[label_path[k],0,0] < 1000:
            # obtain first waypoint
            point_m = (points[0]*3/4 + points[-1]*1/4).astype(np.float32)
            #assign the value to each waypoint
            paths_p[k,2:4] = point_m
            paths_p[k,4:6] = (points[0] + points[-1])/2
            paths_p[k,6:8] = points[-1]
            length_p[k] = 0
            smoothness_p[k] = 0
            # final_dir means the final direction. 1000 means that there is no requirement for the final direction
            dir_final[k] = 1000
        elif path[label_path[k],0,0] < 1000000 and path[label_path[k],0,0] > 1000:
            a = path[label_path[k],0,0]%1000
            # obtain first waypoint
            point_m = (points[0] + points[a])/2
            #assign the value to each waypoint
            paths_p[k,2:4] = point_m
            paths_p[k,4:6] = points[a]
            paths_p[k,6:8] = points[-1]
            length_p[k] = 0
            smoothness_p[k] = 0
            # final_dir means the final direction. 1000 means that there is no requirement for the final direction
            dir_final[k] = 1000
        else:
            j = 0
            n = 0
            while path[label_path[k],0,j] > 0:
                j += 1
            path_index = np.zeros(1).astype(np.int64)
            for i in range(j-1,-1,-1):
                while path[label_path[k],0,i] > 0:
                    a = path[label_path[k],0,i]%1000
                    path[label_path[k],0,i] = (path[label_path[k],0,i]-path[label_path[k],0,i]%1000)/1000
                    path_index = np.append(path_index,a)
                    n += 1
                    if n > 2:
                        break
            paths_p[k,2:4] = points[path_index[1]]
            paths_p[k,4:6] = points[path_index[2]]
            paths_p[k,6:8] = points[path_index[3]]
            if path_index[3] == points.shape[0]-1:
                length_p[k] = 0
                smoothness_p[k] = 0
                dir_final[k] = 1000
            else:
                length_p[k] = length[path_index[3],path_index[2]]
                smoothness_p[k] = smoothness[path_index[3],path_index[2]]
                dx = points[path_index[3],0]-points[path_index[2],0]
                dy = points[path_index[3],1]-points[path_index[2],1]
                dir_final[k] = math.atan2(dy,dx)
    return paths_p, label_path, dir_final, length_p, smoothness_p


@njit
def paths_part_old(path, points, length, smoothness, L_limit):
    # counting the number of available paths and the order of availble paths
    
    label_path = np.zeros(1).astype(np.int64)
    for i in range(path.shape[0]):
        if path[i,0,0] > 0:
            label_path = np.append(label_path,i)
    label_path = np.delete(label_path, 0)
    M = label_path.shape[0]
    paths_p = np.zeros((M,8)).astype(np.float32)
    length_p = np.zeros(M).astype(np.float32)
    smoothness_p = np.zeros(M).astype(np.float32)
    dir_final = np.zeros(M).astype(np.float32)
    paths_p[:,0:2] = points[0]
    for k in range(M):
        if path[label_path[k],0,0] < 1000:
            # obtain the length between the start point and first waypoint
            point_m = (points[0]*3/4 + points[-1]*1/4).astype(np.float32)
            L = math.sqrt(np.sum((point_m - points[0])**2))
            if L > L_limit:
                L_total = math.sqrt(np.sum((points[-1] - points[0])**2))
                point_m = (L_limit/L_total*(points[-1] - points[0]) + points[0]).astype(np.float32)
            paths_p[k,2:4] = point_m
            paths_p[k,4:6] = (points[0] + points[-1])/2
            paths_p[k,6:8] = points[-1]
            length_p[k] = 0
            smoothness_p[k] = 0
            # final_dir means the final direction. 1000 means that there is no requirement for the final direction
            dir_final[k] = 1000
        else:
            j = 0
            n = 0
            while path[label_path[k],0,j] > 0:
                j += 1
            path_index = np.zeros(1).astype(np.int64)
            for i in range(j-1,-1,-1):
                while path[label_path[k],0,i] > 0:
                    a = path[label_path[k],0,i]%1000
                    path[label_path[k],0,i] = (path[label_path[k],0,i]-path[label_path[k],0,i]%1000)/1000
                    path_index = np.append(path_index,a)
                    n += 1
                    if n > 2:
                        break
            paths_p[k,2:4] = (points[0] + points[path_index[1]])/2
            paths_p[k,4:6] = points[path_index[1]]
            paths_p[k,6:8] = points[path_index[2]]
            L = math.sqrt(np.sum((paths_p[k,2:4] - points[0])**2))
            if L > L_limit:
                L_total = math.sqrt(np.sum((points[path_index[1]] - points[0])**2))
                paths_p[k,2:4] = (L_limit/L_total*(points[path_index[1]] - points[0]) + points[0]).astype(np.float32)
            if path_index[2] == points.shape[0]-1:
                length_p[k] = 0
                smoothness_p[k] = 0
                dir_final[k] = 1000
            else:
                length_p[k] = length[path_index[3],path_index[2]]
                smoothness_p[k] = smoothness[path_index[3],path_index[2]]
                dx = points[path_index[3],0]-points[path_index[2],0]
                dy = points[path_index[3],1]-points[path_index[2],1]
                dir_final[k] = math.atan2(dy,dx)
    return paths_p, label_path, dir_final, length_p, smoothness_p
