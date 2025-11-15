#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from config import LEFT_HOOK, LEFT_HAND, RIGHT_HAND, RIGHT_HOOK, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import getcubeplacement

from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    # Current state from PyBullet
    q, vq = sim.getpybulletstate()          # both are length 15
    q = np.array(q)
    vq = np.array(vq)

    # Desired state from trajectory
    q_des  = trajs[0](tcurrent)
    v_des  = trajs[1](tcurrent)
    a_des  = trajs[2](tcurrent)

    # PD tracking in acceleration space
    e     = q_des - q
    e_dot = v_des - vq
    a_cmd = a_des + Kp * e + Kv * e_dot

    model = robot.model
    data  = robot.data

    # Compute dynamics terms
    pin.computeAllTerms(model, data, q, vq)
    M = pin.crba(model, data, q)            # mass matrix
    h = pin.nle(model, data, q, vq)         # nonlinear effects (gravity + Coriolis + centrifugal)

    M = 0.5 * (M + M.T)

    # Computed torque
    tau_track = M.dot(a_cmd) + h
    
    # Send to simulator
    sim.step(tau_track.tolist())


def load_path(filename):
    data = np.load(filename, allow_pickle=True)
    path = [q for q in data["path"]]
    cube_path = [pin.SE3(mat) for mat in data["cube"]]
    return path, cube_path

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, setupwithmeshcat, rununtil
    from config import DT

    
    robot, sim, cube = setupwithpybullet()
    #robot, sim, cube, viz = setupwithpybulletandmeshcat("tcp://127.0.0.1:6000")
    #robot, cube, viz = setupwithmeshcat("tcp://127.0.0.1:6000")
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    # path, cube_path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    def fix_path(path, cube_path, max_points):
        qs = np.array(path)
        print("path length: ", len(qs))
        N = len(qs)
        if N <= max_points:
            return path, cube_path

        stride = int(np.ceil(N / max_points))
        path_sparse = path[::stride]
        cube_sparse = cube_path[::stride]

        # ensure last point is included
        if not np.allclose(path_sparse[-1], path[-1]):
            path_sparse.append(path[-1])
            cube_sparse.append(cube_path[-1])
        
        q0_block = [path[0]] * 50   # 50 timesteps staying still
        new_path = q0_block + path

        return new_path, cube_sparse

    path, cube_path = load_path("saved_rrt_path.npz")
    path, cube_path = fix_path(path, cube_path, max_points=300)


    #setting initial configuration
    sim.setqsim(q0)
    
    
    def maketraj(path, T):
        """
        Build a smooth trajectory q(t), qdot(t), qddot(t) that follows
        the full RRT path using cubic Hermite interpolation with
        zero velocities at each waypoint.
        """

        qs = path
        N = len(qs) - 1                # number of segments
        dt = T / N                     # time per segment

        # Precompute times
        ts = [i * dt for i in range(N + 1)]

        def q_of_t(t):
            """Return desired configuration q(t)."""
            if t <= 0:
                return qs[0]
            if t >= T:
                return qs[-1]

            # Determine which segment t is in
            i = int(t // dt)
            t0 = ts[i]
            t1 = ts[i + 1]
            s = (t - t0) / (t1 - t0)

            q0 = qs[i]
            q1 = qs[i + 1]

            # Hermite basis (zero-velocity endpoints)
            h1 = 2*s**3 - 3*s**2 + 1
            h2 = -2*s**3 + 3*s**2

            return h1 * q0 + h2 * q1

        def vq_of_t(t):
            """Return desired joint velocity qdot(t)."""
            if t <= 0 or t >= T:
                return np.zeros_like(qs[0])

            i = int(t // dt)
            t0 = ts[i]
            t1 = ts[i + 1]
            s = (t - t0) / (t1 - t0)

            q0 = qs[i]
            q1 = qs[i + 1]

            # Derivative of Hermite basis (chain rule)
            h1_p = (6*s**2 - 6*s) / dt
            h2_p = (-6*s**2 + 6*s) / dt

            return h1_p * q0 + h2_p * q1

        def vvq_of_t(t):
            """Return desired joint acceleration qddot(t)."""
            if t <= 0 or t >= T:
                return np.zeros_like(qs[0])

            i = int(t // dt)
            t0 = ts[i]
            t1 = ts[i + 1]
            s = (t - t0) / (t1 - t0)

            q0 = qs[i]
            q1 = qs[i + 1]

            # Second derivative of Hermite basis
            h1_pp = (12*s - 6) / (dt**2)
            h2_pp = (-12*s + 6) / (dt**2)

            return h1_pp * q0 + h2_pp * q1

        return q_of_t, vq_of_t, vvq_of_t


    total_time=3.
    trajs = maketraj(path, total_time)   
    
    tcur = 0.
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    