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
    KP = 10.0        # SE(3) twist gain
    LAMBDA = 1e-4   # Damping for pseudoinverse

    # convergence thresholds
    TOL_ROT = 1e-3
    TOL_LIN = 1e-3
    MAX_IT = 60
    
    oMcubeL = getcubeplacement(cube, LEFT_HOOK) # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK) # placement of the right hand hook

    IDX_RARM = robot.model.getFrameId(RIGHT_HAND)
    IDX_LARM = robot.model.getFrameId(LEFT_HAND)

    q = qcurrent.copy()
    herr_r = [] # Log the value of the error between right hand and right target.
    herr_l = [] # Log the value of the error between left hand and left target.
    
    if viz is not None:
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
                #print(f"[IK] converged after {it} iterations.")
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
        
        # Primary task (right hand)
        JR = damped_pinv(right_Jright, LAMBDA)
        vq = JR @ vstar_R

        # Null-space projector for right-hand task
        Pright = np.eye(robot.nv) - JR @ right_Jright

        # Secondary task (left hand)
        left_Jleft_Pright = left_Jleft @ Pright
        JL = damped_pinv(left_Jleft_Pright, LAMBDA)
        vq += Pright @ (JL @ (vstar_L - left_Jleft @ vq))

        q = pin.integrate(robot.model, q, vq * DT)
        q = projecttojointlimits(robot, q)

        if viz is not None:
            viz.display(q)

        herr_r.append(right_nu)
        herr_l.append(left_nu) 

    # failed to converge
    print("IK did not converge within iteration limit")
    return q, False

            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
