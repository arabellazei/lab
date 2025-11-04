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

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
  
    DT = 1/48
    
    oMcubeL = getcubeplacement(cube, LEFT_HOOK) #placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK) #placement of the right hand hook

    IDX_RARM = robot.model.getFrameId(RIGHT_HAND)
    IDX_LARM = robot.model.getFrameId(LEFT_HAND)

    q = q0.copy()
    herr_r = [] # Log the value of the error between right hand and right target.
    herr_l = [] # Log the value of the error between left hand and left target.
    
    for i in range(300):  # Integrate over 3 second of robot life

        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        # Current EE poses
        oMleft = robot.data.oMf[IDX_LARM]
        oMright = robot.data.oMf[IDX_RARM]

        # 6D pose errors expressed in local frame of each EE
        lhandMhook = oMleft.inverse() * oMcubeL
        left_nu = pin.log(lhandMhook).vector

        rhandMhook = oMright.inverse() * oMcubeR
        right_nu = pin.log(rhandMhook).vector

        # 6D Jacobians in local frames
        left_Jleft = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_LARM)
        right_Jright = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_RARM)
        
        # Primary task (right hand)
        vq = - pinv(right_Jright) @ right_nu

        # Null-space projector for right-hand task
        Pright = np.eye(robot.nv) - pinv(right_Jright) @ right_Jright

        # Secondary task (left hand)
        vq += - pinv(left_Jleft @ Pright) @ (left_nu + left_Jleft @ vq)

        # # Control law by least square - FIX
        # vq = pinv(right_Jright) @ right_nu
        # Pright = np.eye(robot.nv)-pinv(right_Jright) @ right_Jright
        # vq += pinv(left_Jleft @ Pright) @ (left_nu @ vq)

        q = pin.integrate(robot.model, q, vq * DT)

        viz.display(q)
        time.sleep(1e-3)

        herr_r.append(right_nu)
        herr_l.append(left_nu) 

    return q







    # print ("TODO: implement me")
    # return robot.q0, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
