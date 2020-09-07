#############################################################
# Title: Custom gym environment for the four legged robot
# Author: Kez Smithson Whitehead
# Last updated: 2nd September 2018
#############################################################

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
import math
import os
import xml.etree.ElementTree as ET
import pickle
import time
import random
import shutil
rad2deg = 180./(np.pi)
deg2rad = (np.pi)/180.




# ---------------------------------------------------------
# Converts rotation matrix to euler angles
# output [x y z]
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0] * R[0] + R[3] * R[3])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[7], R[8])
        y = np.arctan2(-R[6], sy)
        z = np.arctan2(R[3], R[0])
    else:
        x = np.arctan2(-R[5], R[4])
        y = np.arctan2(-R[6], sy)
        z = 0
    
    return np.array([x, y, z]) * 180 / 3.142
# ----------------------------------------------------------

# ----------------------------------------------------------
# Motor Direction Function
# ==========================================================
# Determines the direction from the velocity
def motorDirect(velocity):
    tol = 0.05
    if velocity < -tol:
        direction = -1
    elif velocity > tol:
        direction = 1
    else:
        direction = 0
    
    return direction
# ----------------------------------------------------------








class QuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    #def __init__(self, index=0, legLengths=[0.6,0.6,0.6,0.6], legSigma=0.02 ,OUsigma=0.03, OUtau=0.3):
    def __init__(self, index=0, legLengths=[0.6,0.6,0.6,0.6], legSigma=0.02 ,OUsigma=0.03, OUtau=0.3, seed=None):
        """Create an quad environment

        Args:
            legLengths float[4]: list of leg lengths (in order : Front left, Front right, Back left, Back right)
            OUsigma float: standard deviation of the OU process applied on the action vectors
            OUtau float: time constant of the OU process applied on the action vectors  CAN NOT BE 0
        """

        if seed!=None:
            random.seed(a=seed)
            np.random.seed(seed)

        #mujoco_env.MujocoEnv.__init__(self, '/home/brl/.mujoco/mujoco200/model/real.xml', 5)
        dirpath = os.path.dirname(os.path.realpath(__file__))
        #fullpath = os.path.join(dirpath, "assets/real.xml")
        xmlstr= 'assets/real'+str(index)+'.xml'
        #xmlstr='/home/gorgsss/Desktop/Dissertation/Main/gym_quad/envs/assets/real'+str(index)+'.xml'
        fullpath = os.path.join(dirpath, xmlstr)
        self.xml_path = fullpath
        self.maxEpisodeSteps = 5000

        #OU parameters
        self.OUsigma = OUsigma  # Standard deviation.
        self.OUmu = 0  # Mean.
        self.OUtau = OUtau  # Time constant.
        self.OUdt = .002  # Time step.
        self.OUsigma_bis = OUsigma * np.sqrt(2. / OUtau)
        self.OUsqrtdt = np.sqrt(self.OUdt)
        self.OUold_noise = np.zeros((12))
        self.OUnoise = np.zeros((12))

        self.legSigma = legSigma
        #self.FL_Length = FL_Length
        #self.FR_Length = FR_Length
        #self.BL_Length = BL_Length
        #self.BR_Length = BR_Length
        self.legLengths = legLengths
        self.defineLegs(self.legLengths)

        self.episode_steps = 0
        xmlImported = False
        while not xmlImported:
            try :
                mujoco_env.MujocoEnv.__init__(self, self.xml_path, 5)
                xmlImported = True
            except :
                continue
        utils.EzPickle.__init__(self)

        

        # self.obsevation_space = spaces.Box(low=np.array([-90.*deg2rad, 0.*deg2rad]),
        #                                high=np.array([90.*deg2rad, 90.*deg2rad]))

        self.action_space = spaces.Box(low=np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
                                       high=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
        #print("INiT DONE", legLengths)
    

    def defineLegs(self, legLengths):
        """Modifies leg lengths in xml file.

        Args:
            legLengths float[4]: list of leg lengths (in order : Front left, Front right, Back left, Back right)
        """
        access = False
        while access ==False:
            try :
                tree = ET.parse(self.xml_path)
                access = True
            except :
           
                try :
                    os.remove(self.xml_path)
                except :
                    a=0
                original = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets/real.xml')
                shutil.copyfile(original, self.xml_path)

        #tree = ET.parse(self.xml_path)
        root = tree.getroot()
        #for geom in root.findall("worldbody/body/body/body/body/[@name='footFR']/geom"):
        #    geom.set("fromto", "0 0 0 0 0 " + str(self.legLengths[0]*0.6))
        #for geom in root.findall("worldbody/body/body/body/body/[@name='footFL']/geom"):
        #    geom.set("fromto", "0 0 0 0 0 " + str(self.legLengths[1]*0.6))
        #for geom in root.findall("worldbody/body/body/body/body/[@name='footBR']/geom"):
        #    geom.set("fromto", "0 0 0 0 0 " + str(self.legLengths[2]*0.6))
        #for geom in root.findall("worldbody/body/body/body/body/[@name='footBL']/geom"):
        #    geom.set("fromto", "0 0 0 0 0 " + str(self.legLengths[3]*0.6))
        
        #for pos in root.findall("worldbody/body/[@name='torso']"): 
        #    pos.set("pos", "0 0 " + str(abs(np.average(legLengths)+0.7)))



        for geom in root.findall("asset/mesh/[@name='kneeFR']"):
            geom.set("scale", "0.01 0.01 " + str(legLengths[0]*0.01))
            #print(legLengths[0]*0.01)
        for geom in root.findall("worldbody/body/body/body/body/[@name='footFR']"):
            geom.set("pos", "0 0 " + str(legLengths[0]*-0.92))
        for geom in root.findall("asset/mesh/[@name='kneeFL']"):
            geom.set("scale", "0.01 0.01 " + str(legLengths[1]*0.01))
        for geom in root.findall("worldbody/body/body/body/body/[@name='footFL']"):
            geom.set("pos", "0 0 " + str(legLengths[1]*-0.92))
        for geom in root.findall("asset/mesh/[@name='kneeBR']"):
            geom.set("scale", "0.01 0.01 " + str(legLengths[2]*0.01))
        for geom in root.findall("worldbody/body/body/body/body/[@name='footBR']"):
            geom.set("pos", "0 0 " + str(legLengths[2]*-0.92))
        for geom in root.findall("asset/mesh/[@name='kneeBL']"):
            geom.set("scale", "0.01 0.01 " + str(legLengths[3]*0.01))
        for geom in root.findall("worldbody/body/body/body/body/[@name='footBL']"):
            geom.set("pos", "0 0 " + str(legLengths[3]*-0.92))

        for pos in root.findall("worldbody/body/[@name='torso']"): 
            pos.set("pos", "0 0 " + str(abs(np.average(legLengths)*0.92)))



       
        tree.write(self.xml_path)

  
    def step(self, a):
        self.episode_steps = self.episode_steps+1
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        quat_Matrix_Before = self.data.sensordata
        # euler_Matrix_Before = rotationMatrixToEulerAngles(quat_Matrix_Before)

        #Action Noise
        self.OUnoise = self.OUold_noise + self.OUdt * (-self.OUold_noise - self.OUmu*np.ones((12))) / self.OUtau*np.ones((12)) + self.OUsigma_bis*np.ones((12)) * self.OUsqrtdt*np.ones((12)) * np.random.randn((12))
        relevantNoise = self.OUnoise
        #for i in range[12]:
        #    if i%3==0 or i%3==1 :
        #        relevantNoise[i] = 0
        a = a+relevantNoise
        #print(relevantNoise)
        #print(a)

        self.do_simulation(a, self.frame_skip)
        
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        quat_Matrix_After = self.data.sensordata
        #euler_Matrix_After = rotationMatrixToEulerAngles(quat_Matrix_After)
        
        forwardReward = (xposafter - xposbefore)/self.dt
        sidePunishment = 0.5*abs(yposafter - yposbefore)/self.dt
        # pitchReward = (euler_Matrix_After[0] - euler_Matrix_Before[0])/self.dt
        # rollReward = (euler_Matrix_After[1] - euler_Matrix_Before[1])/self.dt
        # yawReward = (euler_Matrix_After[2] - euler_Matrix_Before[2])/self.dt
        reward = forwardReward #- sidePunishment#- pitchReward - rollReward - yawReward
        
 
        # notdone = np.isfinite(ob).all()
        # done = not notdone
        # qpos = self.sim.data.qpos
        # bob, fred, ang = self.sim.data.qpos[0:3]
        # print('position x: ', xposafter, '  y: ', yposafter, '  reward: ', reward)
        # print ("Euler orietation torso: ", quat_Matrix_After)
        
        # termination criteria
        orientation = self.data.sensordata
        orientation = np.round(orientation,2)
        done = bool( (orientation[1] < -0.3) or (orientation[1] > 0.3) or (self.episode_steps==self.maxEpisodeSteps) )
        if (orientation[1] < -0.3) or (orientation[1] > 0.3):
            reward=reward #-20


        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forwardReward)
        # return ob, reward, done, done
    
    # Everything the robot observes-------------------------------------------------------------------------------------
    def _get_obs(self):
        
        # Joint Angles speed up by removing rad to deg
        a0 = self.sim.data.qpos[7] * rad2deg  # right hip out
        a1 = self.sim.data.qpos[8] * rad2deg
        a2 = self.sim.data.qpos[9] * rad2deg  # left hip out self.get_body_com("torso")
        a3 = self.sim.data.qpos[10] * rad2deg
        a4 = self.sim.data.qpos[11] * rad2deg  # right hip out
        a5 = self.sim.data.qpos[12] * rad2deg
        a6 = self.sim.data.qpos[13] * rad2deg  # left hip out self.get_body_com("torso")
        a7 = self.sim.data.qpos[14] * rad2deg
        a8 = self.sim.data.qpos[15] * rad2deg  # right hip out
        a9 = self.sim.data.qpos[16] * rad2deg
        a10 = self.sim.data.qpos[17] * rad2deg  # left hip out self.get_body_com("torso")
        a11 = self.sim.data.qpos[18] * rad2deg

        # Orientation sensor
        orientation = self.data.sensordata
        orientation = np.round(orientation,2)
        quatr = orientation[0]
        quat1 = orientation[1]
        quat2 = orientation[2]
        quat3 = orientation[3]
        OBS = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, quatr, quat1, quat2, quat3]
        return np.array(OBS)
    
    # Reset by setting velocity = 0 and position to xml model values----------------------------------------------------
    def reset_model(self):
        #update leg lengths with noise
        self.episode_steps = 0
        newLegLengths = self.legLengths + self.legSigma**0.5 * np.random.randn(4)*self.legLengths
        newLegLengths = np.clip(newLegLengths,0.,10.)
        self.defineLegs(newLegLengths)
        #print(newLegLengths)
        
        qpos = self.init_qpos   # Initial + randomness
        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)  # add randomness
        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()
    
    # Camera point--------------------------------------------------------------------------------------------------------
    def viewer_setup(self):
        self.viewer.cam.lookat[1] += 3


#if __name__ == '__main__':
#    test = QuadEnv([1,1,1,1])
#    test.defineLegs([0.7,0.8,0.8,0.7])
#    for i in range(15):    
#        ob, reward, done1, done2 = test.step([0,0,0.2,0,0,0.2,0,0,0.2,0,0,0.2])

        #print(reward)
    #print(ob)