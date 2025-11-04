#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):

    setcubeplacement(robot, cube, cubetarget)

    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)
    pin.updateGeometryPlacements(cube.model, cube.data,
                                  cube.collision_model, cube.collision_data, cube.q0)

    left_hook_id = cube.model.getFrameId(LEFT_HOOK)
    right_hook_id = cube.model.getFrameId(RIGHT_HOOK)

    oMlhook = cubetarget * cube.data.oMf[left_hook_id] # ?
    oMrhook = cubetarget * cube.data.oMf[right_hook_id]

    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)

    q = qcurrent.copy()

    max_iterations = 1000
    epsilon = 1e-4
    step_size = 0.5

    for i in range(max_iterations):
        pin.framesForwardKinematics(robot.model, robot.data, q)

        oMlhand = robot.data.oMf[left_hand_id]
        oMrhand = robot.data.oMf[right_hand_id]

        left_error = pin.log(oMlhand.inverse() * oMlhook).vector # inverse of matrix
        right_error = pin.log(oMrhand.inverse() * oMrhook).vector

        if norm(left_error) < epsilon and norm(right_error) < epsilon:
            if not collision(robot, q): # collion with anything in the env
                if viz:
                    viz.display(q)
                return q, True
            else:
                break

        pin.computeJointJacobians(robot.model, robot.data, q)
        J_left = pin.getFrameJacobian(robot.model, robot.data, left_hand_id, pin.ReferenceFrame.LOCAL)
        J_right = pin.getFrameJacobian(robot.model, robot.data, right_hand_id, pin.ReferenceFrame.LOCAL)

        error = np.concatenate([left_error, right_error])
        J = np.vstack([J_left, J_right])

        damping = 1e-6
        J_pinv = J.T @ inv(J @ J.T + damping * np.eye(J.shape[0])) # damped psuedo inv

        dq_task = J_pinv @ error

        postural_weight = 0.1
        P = np.eye(len(q)) - J_pinv @ J
        dq_posture = postural_weight * (qcurrent - q)

        dq = dq_task + P @ dq_posture

        q = q + step_size * dq

        q = projecttojointlimits(robot, q)

    max_escape_iter = 200
    for i in range(max_escape_iter):
        if not collision(robot, q):
            if viz:
                viz.display(q)
            return q, True

        gradient = np.zeros(len(q))
        delta = 0.01

        for j in range(len(q)):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[j] += delta
            q_minus[j] -= delta

            score_plus = 1.0 if collision(robot, q_plus) else 0.0
            score_minus = 1.0 if collision(robot, q_minus) else 0.0

            gradient[j] = (score_plus - score_minus) / (2 * delta)

        if norm(gradient) > 1e-6:
            q = q - 0.05 * gradient / norm(gradient)
        else:
            q = q + np.random.randn(len(q)) * 0.02

        q = projecttojointlimits(robot, q)

    if viz:
        viz.display(q)
    return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    