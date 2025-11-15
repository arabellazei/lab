#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from inverse_geometry import computeqgrasppose
from setup_meshcat import updatevisuals

from config import LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import time

def sample_random_SE3(robot, cube, q):
    """Return a random SE3 pose (uniform position & orientation)."""
    # --- settings
    TABLE_Z = CUBE_PLACEMENT.translation[2]
    RADIUS = 0.3
    MAX_REACH = 1
    MIN_REACH = 0.2
    MAX_TRIES = 100

    # midpoint between start and goal
    mid_pos = 0.5 * (CUBE_PLACEMENT.translation + CUBE_PLACEMENT_TARGET.translation)

    # robot chest position
    CHEST_FRAME = "CHEST_JOINT0_Link"
    CHEST_ID = robot.model.getFrameId(CHEST_FRAME)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    chest_pos = robot.data.oMf[CHEST_ID].translation

    for _ in range(MAX_TRIES):
        # random direction inside unit sphere
        v = np.random.normal(0, 1, 3)
        v /= np.linalg.norm(v)
        r = np.random.rand() ** (1/3) * RADIUS  # uniform in volume
        pos = mid_pos + r * v                   # sampled position

        dist = np.linalg.norm(pos - chest_pos)
        # table condition
        if pos[2] < TABLE_Z:
            continue

        # reachability condition
        if not (MIN_REACH < dist < MAX_REACH):
            continue

        # identity rotation for now
        R = np.eye(3)
        oMcube = pin.SE3(R, pos)

        # temporarily assign cube placement for collision check
        setcubeplacement(robot, cube, oMcube)
        if not pin.computeCollisions(cube.collision_model, cube.collision_data, False):
            return oMcube

    # if no valid pose found
    return None

def RAND_CONF(robot, q, cube, viz = None):
    """ Generate a random Sample cube position and compute the corresponding grasping pose """
    for i in range(50):     # retry attempts until one is valid
        oMcube_rand = sample_random_SE3(robot, cube, q)
        q_rand, success = computeqgrasppose(robot, q, cube, oMcube_rand, viz)
        if success: 
            return q_rand, oMcube_rand
    
def NEAREST_VERTEX(G, oMcube):
    """ Return the graph index of the node that has the nearest cube position """
    min_dist = 10e4
    idx = -1
    for (i, node) in enumerate(G):
        dist = np.linalg.norm(oMcube.translation - node[2].translation) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx 

def lerp(q0,q1,t):    
    """ Linear interpolation """
    return q0 * (1 - t) + q1 * t

def NEW_CONF(robot, cube, cube_near, cube_rand, q_near, discretisationsteps, delta_q = None):
    """ Return the path section to the closest cube position cube_end such that the path cube_near => cube_end is the longest
    along the linear interpolation (cube_near,cube_rand) that is collision free and of length <  delta_q """
    cube_end = cube_rand.copy()
    R = cube_end.rotation
    dist = np.linalg.norm(cube_rand.translation - cube_near.translation)
    if delta_q is not None and dist > delta_q:
        #compute the configuration that corresponds to a path of length delta_q
        cube_end_pos = lerp(cube_near.translation, cube_rand.translation, delta_q/dist)
        cube_end = pin.SE3(R, cube_end_pos)
        # now dist == delta_q
    dt = 1 / discretisationsteps

    path = []
    flag = True

    for i in range(1, discretisationsteps):
        R = cube_near.rotation
        cube_pos = lerp(cube_near.translation, cube_end.translation, dt*i)
        cube_i = pin.SE3(R, cube_pos)
        setcubeplacement(robot, cube, cube_i)
        q, success = computeqgrasppose(robot, robot.q0, cube, cube_i)
        cube_ok = not pin.computeCollisions(cube.collision_model, cube.collision_data, False)
        
        if not (success and cube_ok):
            flag = False
            break
        # collision free
        path.append((q, cube_i))  
    return path, flag
 

def ADD_PATH_SECTION(G, parent, section):
    current_parent = parent
    for (q, cube_pos) in section:
        new_node_index = len(G)
        G.append((current_parent, q, cube_pos))
        current_parent = new_node_index

def VALID_EDGE(robot, cube, cube_new, cube_goal, q, discretisationsteps):
    path, flag = NEW_CONF(robot, cube, cube_new, cube_goal, q, discretisationsteps)
    return path, flag

def rrt(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    """ This is the RRT algorithm engine """
    G = [(None, qinit, cubeplacementq0)]    # each node of graph stores (Parent, configuration, cube position)
    k = 1000    # number of nodes. Can be adjusted
    delta_q = 0.2
    discretisationsteps = 80

    for _ in range(k):
        q_rand, cube_rand = RAND_CONF(robot, qinit, cube)
        cube_near_index = NEAREST_VERTEX(G, cube_rand)
        _, q_near, cube_near = G[cube_near_index]   
        new_G_section, _ = NEW_CONF(robot, cube, cube_near, cube_rand, q_near, discretisationsteps, delta_q)    
        if new_G_section == []: continue
        ADD_PATH_SECTION(G, cube_near_index, new_G_section)
        q_new, cube_new = new_G_section[-1][0], new_G_section[-1][1], 
        to_goal_section, flag = VALID_EDGE(robot, cube, cube_new, cubeplacementqgoal, q_new, discretisationsteps)
        if flag:    
            print ("Path found!")
            ADD_PATH_SECTION(G, len(G)-1, to_goal_section)
            return G, True
    print("path not found")
    return G, False    


def interpolate_SE3(T1: pin.SE3, T2: pin.SE3, alpha: float) -> pin.SE3:
    """Interpolate between two SE3 poses: 
       linear position + quaternion slerp for rotation."""
    # Linear interpolation for position
    p1 = T1.translation
    p2 = T2.translation
    p = (1 - alpha) * p1 + alpha * p2

    # Quaternion slerp
    q1 = pin.Quaternion(T1.rotation)
    q2 = pin.Quaternion(T2.rotation)
    q_interp = q1.slerp(alpha, q2)

    return pin.SE3(q_interp.toRotationMatrix(), p)

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    G, pathfound = rrt(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    
    path = []
    cube_path = []
    node = G[-1]
    while node[0] is not None:
        path = [node[1]] + path
        cube_path = [node[2]] + cube_path
        node = G[node[0]]
    path = [G[0][1]] + path
    cube_path = [G[0][2]] + cube_path
    return path, cube_path


def displaypath(robot, cube, path, cube_path, dt, viz):
    for i, q in enumerate(path):
        setcubeplacement(robot, cube, cube_path[i])
        viz.display(q)

def save_path(filename, path, cube_path):
    # Convert cube_path SE3 → 4×4 arrays
    cube_mats = [pose.homogeneous for pose in cube_path]

    np.savez(filename,
             path=np.array(path),          # shape (N, nq)
             cube=np.array(cube_mats))     # shape (N, 4, 4)

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat("tcp://127.0.0.1:6000")
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path, cube_path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot, cube, path, cube_path, dt=0.5, viz=viz) #you ll probably want to lower dt
    save_path("saved_rrt_path.npz", path, cube_path)
