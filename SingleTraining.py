# Train an agent from scratch with PPO2 and save package and learning graphs
# from OpenGL import GLU
import os
import glob
import time
import subprocess
import shutil
import gym
from gym_quad.envs import quad_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import imageio
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_vec_env


class PPOtraining():
    def __init__(self, envParameters, initModelLoc=None) :
        self.step_stage = 0
        self.step_total = 1000000

        self.env = None
        self.n_cpu = 10
        self.modelLoc = initModelLoc
        self.model = None
        self.stage = -1
        self.envParameters = envParameters
        self.createDirectories()
 
        #Attributes to store results across stages
        self.all_x = []
        self.all_y = []
        self.vert_x = []
        
        
        for i in self.envParameters[:,0]:
            self.changeStage()
            self.train()


    def createDirectories(self):
        self.models_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "models/")
        
        self.models_tmp_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "models_tmp/")
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        self.gif_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp_gif/")
        self.plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.models_tmp_dir, exist_ok=True)
        os.makedirs(self.plt_dir, exist_ok=True) 
      
   

    def updateEnv(self):
        if self.env != None :
            self.env.close()
        #env = quad_env.QuadEnv(legLengths=self.envParameters[self.stage,:4],legSigma=self.envParameters[self.stage,4],OUsigma=self.envParameters[self.stage,5],OUtau=self.envParameters[self.stage,6])
        
        #env.close() #just to update the parameters
        #self.env = [Monitor(env, self.log_dir+"/"+str(i), allow_early_resets=True) for i in range(self.n_cpu)]
        #env = gym.make("Quad-v0", legLengths=self.envParameters[self.stage,:4], legSigma=self.envParameters[self.stage,4], OUsigma=self.envParameters[self.stage,5], OUtau=self.envParameters[self.stage,6])
        #print(str(type(env)))
        env_kwargs = {'legLengths':self.envParameters[self.stage,:4],'legSigma':self.envParameters[self.stage,4],'OUsigma':self.envParameters[self.stage,5],'OUtau':self.envParameters[self.stage,6]}
        #print(env_kwargs)
        self.env=make_vec_env(quad_env.QuadEnv, n_envs=self.n_cpu, seed=None, start_index=0, monitor_dir=self.log_dir, wrapper_class=None, env_kwargs=env_kwargs, vec_env_cls=None, vec_env_kwargs=None)
        
        #self.env = gym.make("Quad-v0", legLengths = self.envParameters[self.stage,:4], legSigma = self.envParameters[self.stage,4], OUsigma = self.envParameters[self.stage,5], OUtau = self.envParameters[self.stage,6])
        #self.env = Monitor(self.env, self.log_dir, allow_early_resets=True)
        #self.env = DummyVecEnv( [lambda: self.env for i in range(self.n_cpu)] )
        #self.env = SubprocVecEnv([lambda: self.env for i in range(self.n_cpu)])
        #self.env = SubprocVecEnv(lambda : self.env)
        #env =self.custom_quad_env()
        #self.env = VecFrameStack(env, n_stack=self.n_cpu)

    def updateModel(self):
        if self.modelLoc==None:
            self.model = PPO2(MlpPolicy, self.env, verbose=1)
            self.modelLoc = self.models_dir
        else :
            del self.model
            print(self.stage)
            self.model = PPO2.load(self.modelLoc+"stage_"+str(self.stage-1)+"final.plk", env=self.env)
            print("loaded "+str((self.modelLoc+"stage_"+str(self.stage-1)+"final.plk")))

    def changeStage(self):
        self.stage = self.stage + 1
        self.updateEnv()
        
        self.updateModel()
        
    

    def train(self):
        start = time.time()
        self.model.learn(total_timesteps=self.step_total)


        self.model.save(self.modelLoc+"stage_"+str(self.stage)+"final.plk")
        print("saved "+str((self.modelLoc+"stage_"+str(self.stage)+"final.plk")))
        end = time.time()
        training_time = end - start 
        print("Training of stage "+ str(self.stage) +" completed in "+ str(training_time) +" s !")
        self.step_stage = self.step_stage + self.step_total      
        self.performance()
        self.createGif()

    def performance(self):

        #log_files = glob.glob(self.log_dir+"/*.csv") 
        #df = pd.concat((pd.read_csv(f, header = 0) for f in log_files))
        #df.to_csv(self.log_dir+"/monitor.csv")
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        #print(x)
        #print(y)
        y = self.moving_average(y, window=50)
        x = x[len(x) - len(y):]
        x = x + self.step_stage*np.ones(np.size(x))
        for i in x:
            try :
                self.all_x.append(i + vert_x[-1])
                appended_val = x[-1] + vert_x[-1]
            except :
                self.all_x.append(i)
                appended_val = x[-1]

        self.vert_x.append(appended_val)
        for i in y:
            self.all_y.append(i)
        #os.remove(os.path.join(self.log_dir, "/monitor.csv"))

        save_name = os.path.join(self.plt_dir, 'Curriculum_stage'+str(self.stage))
        print(self.all_x)
        print(self.all_y)
        fig = plt.figure('Curriculum' + str(self.step_stage)+"_stage"+str(self.stage))
        plt.plot(self.all_x, self.all_y)
        for i in self.vert_x:
            plt.axvline(x=i, linestyle='--', color='#ccc5c6', label='leg increment')
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title('Curriculum learning' + " Smoothed")
        plt.savefig(save_name + ".png")
        plt.savefig(save_name + ".eps")
        print("plots saved...")
        if self.stage==3:
            plt.show()

    def moving_average(self,values, window):
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')

    def createGif(self):
        gif_name = "PPO2_" + "stage_"+ str(self.stage) 
        save_str = self.gif_dir + gif_name + '.gif'
        self.env.close()
        #env_gif = quad_env.QuadEnv(self.envParameters[self.stage,:4],self.envParameters[self.stage,4],self.envParameters[self.stage,5],self.envParameters[self.stage,6])
        env_gif = quad_env.QuadEnv(legLengths=self.envParameters[self.stage,:4],legSigma=self.envParameters[self.stage,4],OUsigma=self.envParameters[self.stage,5],OUtau=self.envParameters[self.stage,6])
        #env_gif = gym.make("Quad-v0", legLengths=self.envParameters[self.stage,:4], legSigma=self.envParameters[self.stage,4], OUsigma=self.envParameters[self.stage,5], OUtau=self.envParameters[self.stage,6])
        #del self.model
       
        #self.model = PPO2.load(self.modelLoc+"stage_"+str(self.stage)+"final.plk", env=self.env)
        images = []
        obs = env_gif.reset()
        img = env_gif.sim.render(
            width=400, height=400, camera_name="isometric_view")
        for _ in range(1000):
            action, _ = self.model.predict(obs)
            obs, _, _, _ = env_gif.step(action)
            img = env_gif.sim.render(
                width=400, height=400, camera_name="isometric_view")
            images.append(np.flipud(img))

        print("creating gif...")
        imageio.mimsave(save_str, [np.array(img)
                                   for i, img in enumerate(images) if i % 2 == 0], fps=29)
        print("gif created...")
        env_gif.close()



if __name__ == '__main__':
    Curriculum = np.array([[0.8,0.8,0.8,0.8,0.02,0.03,0.1],[0.9,0.9,0.9,0.9,0.02,0.03,0.2],[0.95,0.95,0.95,0.95,0.02,0.03,0.2],[1.0,1.0,1.0,1.0,0.02,0.03,0.2]])
    agent = PPOtraining(Curriculum)