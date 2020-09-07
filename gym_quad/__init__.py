from gym.envs.registration import register

register(
         id='Quad-v0',
         entry_point='gym_quad.envs:QuadEnv',
         kwargs = {'legLengths':[0.6,0.6,0.6,0.6], 'legSigma':0.02 ,'OUsigma':0.03, 'OUtau':0.3},
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )