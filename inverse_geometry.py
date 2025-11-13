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
import time
from setup_meshcat import updatevisuals

from tools import setcubeplacement

import quadprog

def solve_qp(H, f, lb, ub):
    # quadprog solves: min 1/2 xᵀ H x - bᵀ x
    # so we pass b = -f
    n = H.shape[0]

    # inequality: lb <= x <= ub   →   Gx ≤ h
    G = np.vstack(( np.eye(n), -np.eye(n) ))
    h = np.hstack(( ub, -lb ))

    sol = quadprog.solve_qp(H, -f, G.T, h)[0]
    return sol


def damped_pinv(J, lam=1e-3):
    # 6xnv Jacobian -> nvx6 pseudo-inverse with Tikhonov damping
    # J# = J^T ( J J^T + lam^2 I )^{-1}
    JJt = J @ J.T
    return J.T @ inv(JJt + (lam**2) * np.eye(J.shape[0]))


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    # controller parameters
    DT = 1/50      # smaller steps for stability
    KP = 4.0        # SE(3) twist gain
    LAMBDA = 1e-4   # Damping for pseudoinverse
    VMAX = 0.8

    # convergence thresholds
    TOL_ROT = 2e-2
    TOL_LIN = 2e-3
    MAX_IT = 200
    
    oMcubeL = getcubeplacement(cube, LEFT_HOOK) #placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK) #placement of the right hand hook

    IDX_RARM = robot.model.getFrameId(RIGHT_HAND)
    IDX_LARM = robot.model.getFrameId(LEFT_HAND)

    q = robot.q0.copy()
    herr_r = [] # Log the value of the error between right hand and right target.
    herr_l = [] # Log the value of the error between left hand and left target.
    
    updatevisuals(viz, robot, cube, q)

    for it in range(MAX_IT):  # Integrate over 3 second of robot life

        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)
        pin.updateFramePlacements(robot.model, robot.data)

        # Current EE poses
        oMleft = robot.data.oMf[IDX_LARM]
        oMright = robot.data.oMf[IDX_RARM]

        # 6D pose errors expressed in local frame of each EE
        lhandMhook = oMleft.inverse() * oMcubeL
        left_nu = pin.log(lhandMhook).vector

        rhandMhook = oMright.inverse() * oMcubeR
        right_nu = pin.log(rhandMhook).vector

        # ---- convergence test ----
        if (np.linalg.norm(left_nu[:3])  < TOL_ROT and 
            np.linalg.norm(left_nu[3:])  < TOL_LIN and
            np.linalg.norm(right_nu[:3]) < TOL_ROT and 
            np.linalg.norm(right_nu[3:]) < TOL_LIN):

            if not collision(robot, q):
                print(f"[IK] converged after {it} iterations.")
                if viz is not None:
                    viz.display(q)
                return q, True
            
            else:
                return q, False

        # Desired local twists
        vstar_L = KP * left_nu
        vstar_R = KP * right_nu

        # 6D Jacobians in local frames
        left_Jleft = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_LARM, pin.ReferenceFrame.LOCAL)
        right_Jright = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_RARM, pin.ReferenceFrame.LOCAL)
        
        #-------
        # Primary task (right hand)
        JR = damped_pinv(right_Jright, LAMBDA)
        vq = JR @ vstar_R

        # Null-space projector for right-hand task
        Pright = np.eye(robot.nv) - JR @ right_Jright

        # Secondary task (left hand)
        left_Jleft_Pright = left_Jleft @ Pright
        JL = damped_pinv(left_Jleft_Pright, LAMBDA)
        #vq += - pinv(JL @ Pright) @ (left_nu + JL @ vq)
        vq += Pright @ (JL @ (vstar_L - left_Jleft @ vq))
        #-------

        # Build QP terms -----------------

        # JR = right_Jright
        # JL = left_Jleft

        # alpha = 0.2         # weight for left hand
        # beta = 0.01          # postural weight
        # Kposture = 0.3

        # # Cost matrices
        # H = (JR.T @ JR) + alpha*(JL.T @ JL) + beta*np.eye(robot.nv) + 1e-6 * np.eye(robot.nv)
        # f = -(JR.T @ vstar_R) - alpha*(JL.T @ vstar_L) - beta*(Kposture*(robot.q0 - q))

        # # Joint limit constraints in velocity space
        # qmin = robot.model.lowerPositionLimit
        # qmax = robot.model.upperPositionLimit

        # lb = (qmin - q) / DT
        # ub = (qmax - q) / DT

        # # Solve QP
    
        # vq = solve_qp(H, f, lb, ub)
        #     #vq = solve_qp_slsqp(H, f, lb, ub)
        # ----- end of qp

        #vq = np.clip(vq, , VMAX)

        q = pin.integrate(robot.model, q, vq * DT)
        q = projecttojointlimits(robot, q)

        viz.display(q)
        #time.sleep(1e-3)

        herr_r.append(right_nu)
        herr_l.append(left_nu) 

    # failed to converge
    print("IK did not converge within iteration limit")
    return q, False








    # print ("TODO: implement me")
    # return robot.q0, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
