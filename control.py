#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin

from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND

Kp = 300.               # proportional gain
Kv = 2 * np.sqrt(Kp)   # derivative gain

# Grasping force parameters
GRASP_FORCE = 150.0
LIFT_FORCE = 30.0

def controlLaw(sim, robot, trajs, tcurrent, cube):
    """
    Control equation: τ = M(q)(q̈_d + Kp*e + Kv*ė) + h(q,q̇) + J^T*F_grasp
    """
    q, vq = sim.getpybulletstate()

    # Get desired state from trajectory
    q_of_t, vq_of_t, vvq_of_t = trajs
    q_desired = q_of_t(tcurrent)
    vq_desired = vq_of_t(tcurrent)
    vvq_desired = vvq_of_t(tcurrent)

    # Compute tracking errors
    e = q_desired - q 
    de = vq_desired - vq 

    # Update pinocchio data for current state
    pin.computeAllTerms(robot.model, robot.data, q, vq)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    # Get dynamics terms
    M = robot.data.M            
    h = robot.data.nle 

    # Use higher gains to more accurately follow the collision-free path
    Kp_effective = Kp * 1.5 
    Kv_effective = Kv * 1.2 

    # Compute feedforward + feedback control
    desired_acceleration = vvq_desired + Kp_effective * e + Kv_effective * de
    torques_tracking = M @ desired_acceleration + h

    # PART 2: Add grasping force control

    # Get hand frame IDs
    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)

    # Get hand positions and orientations
    oMleft = robot.data.oMf[left_hand_id]
    oMright = robot.data.oMf[right_hand_id]

    # Compute direction vector from left hand to right hand
    left_pos = oMleft.translation
    right_pos = oMright.translation
    grasp_direction = right_pos - left_pos
    grasp_direction_norm = np.linalg.norm(grasp_direction)

    if grasp_direction_norm > 1e-6:
        grasp_direction = grasp_direction / grasp_direction_norm

        # Apply inward forces: left hand pushes toward right, right hand pushes toward left also add upward force component to help lift the cube
        F_left = np.zeros(6)
        F_left[:3] = GRASP_FORCE * grasp_direction 
        F_left[2] += LIFT_FORCE 

        F_right = np.zeros(6)
        F_right[:3] = -GRASP_FORCE * grasp_direction 
        F_right[2] += LIFT_FORCE 

        # Compute Jacobians for both hands
        J_left = pin.computeFrameJacobian(robot.model, robot.data, q, left_hand_id, pin.LOCAL_WORLD_ALIGNED)
        J_right = pin.computeFrameJacobian(robot.model, robot.data, q, right_hand_id, pin.LOCAL_WORLD_ALIGNED)

        # Convert forces to joint torques
        torques_grasp = J_left.T @ F_left + J_right.T @ F_right
    else:
        torques_grasp = np.zeros(len(q))

    # Combine tracking and grasping torques
    torques = torques_tracking + torques_grasp

    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    # PART 2: Trajectory generation from path
    def makeTrajectory(path, T):
        """""
        Generate trajectory using piecewise linear interpolation
        Arguments:
            path: List of configurations along the geometric path
            T: Total time duration
            
        Returns:
            position, velocity, acceleration
        """
        if path is None or len(path) == 0:
            # Fallback: simple trajectory between q0 and qe
            print("Warning: No valid path provided, using simple trajectory")
            q_of_t = Bezier([q0, q0, q0, qe, qe, qe], t_max=T)
            vq_of_t = q_of_t.derivative(1)
            vvq_of_t = vq_of_t.derivative(1)
            return q_of_t, vq_of_t, vvq_of_t

        print(f"Generating trajectory with {len(path)} waypoints over {T} seconds")

        # Convert path to numpy array for easier manipulation
        path_array = np.array(path)
        n_waypoints = len(path)

        # Allocate time uniformly across segments
        segment_times = np.linspace(0, T, n_waypoints)

        def q_of_t(t):
            # Piecewise linear position interpolation
            t = np.clip(t, 0, T)

            # Find which segment we're in
            idx = np.searchsorted(segment_times, t) - 1
            idx = np.clip(idx, 0, n_waypoints - 2)

            # Linear interpolation between waypoints
            t0 = segment_times[idx]
            t1 = segment_times[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0

            return (1 - alpha) * path_array[idx] + alpha * path_array[idx + 1]

        def vq_of_t(t):
            # Piecewise constant velocity
            t = np.clip(t, 0, T)

            # Find which segment we're in
            idx = np.searchsorted(segment_times, t) - 1
            idx = np.clip(idx, 0, n_waypoints - 2)

            # Constant velocity in each segment
            t0 = segment_times[idx]
            t1 = segment_times[idx + 1]
            dt = t1 - t0

            if dt > 0:
                return (path_array[idx + 1] - path_array[idx]) / dt
            else:
                return np.zeros_like(path_array[0])

        def vvq_of_t(t):
            # Zero acceleration (piecewise linear has constant velocity per segment)
            return np.zeros_like(path_array[0])

        return q_of_t, vq_of_t, vvq_of_t


    # Generate trajectory from the computed path
    # Time allocation: Longer times allow smoother, more stable motions
    total_time = 15.0
    trajs = makeTrajectory(path, total_time)   
    
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controlLaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT

