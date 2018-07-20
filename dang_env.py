"""
OpenAI gym environment wrapper
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import math
from target import list_sceneID, list_sceneLocation

from minos.lib.Simulator import Simulator
from minos.lib import common


class DangEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self._last_state = None
        self._sim = None
        self.viewer = None
        self.actions = []
        action_name = ['idle', 'moveTo', 'forwards', 'backwards', 'strafeLeft', 'strafeRight', 
                            'turnLeft', 'turnRight', 'lookUp', 'lookDown']
        
        for name in action_name:
            action = {'name': name, 'strength': 1, 'angle': math.radians(5)}
            self.actions.append(action)
        self.arg = None 
        self.id_scene = None
        self.id_object = None 
        self.goalObject = None
        #TODO: select id scene as well as set id object 
        # default=['00a76592d5cc7d92eef022393784a2de', '066e8a32fd15541e814a9eafa82abf5d',
                                # '11be354e8dbd5d7275c486a5037ea949'],

    def configure(self, sim_args):
        print("varrrr")
        print(vars(sim_args))
        self.arg = sim_args
        self.set_scene(number = '11be354e8dbd5d7275c486a5037ea949')
        self.set_goal()
        self._sim = Simulator(vars(sim_args))
        common.attach_exit_handler(self._sim)
        self._sim.start()
        # for i, x in enumerate(sim_args):
        #     print(x)
        #     print(sim_args[x])
        
        # self._sim_obs_space = self._sim.get_observation_space(sim_args['outputs'])
        # print("Enterrrrrrrrrrrrrrrrrrrrrrrrrrrrrrreeddddddddddddddddddddddddddddddddd configure")
        # #self.action_space = spaces.Discrete(self._sim.num_buttons)
        # self.action_space = spaces.MultiBinary(self._sim.num_buttons)
        # self.screen_height = self._sim_obs_space['color'].shape[1]
        # self.screen_width = self._sim_obs_space['color'].shape[0]
        # self.observation_space = spaces.Box(low=0, high=255, 
        #     shape=(self.screen_height, self.screen_width, 3))
        # # TODO: have more complex observation space with additional modalities and measurements
        # # obs_space = self._sim.get_observation_space
        # #self.observation_space = spaces.Dict({"images": ..., "depth": ...})
    def set_scene(self, number = None ):
        self.id_scene = number
        self.arg['scene_ids'][0] = number
        self.arg['fullId'] = self.arg.source + '.' + number
        self.arg['scene']['fullId'] = self.arg['fullId']
        return self.id_scene

    # select goal by collection in target.py
    # todo goal can be selected freely
    def set_goal(self, goal_number = None):
        if self.id_scene == None:
            print("selecting default goal for scene cause dont not exist scene ID")
            self.goalObject = None
            return 
        self.goalObject = list_sceneLocation[self.id_scene][0]
        try:
            self.goalObject = list_sceneLocation[self.id_scene][goal_number]
        except Exception as e:
            print("Warning error in set_goal ")
            
        return True

    def set_object(self, number = None):
        self.id_object = number
        return self.id_object

    @property
    def simulator(self):
        return self._sim

    

    def _seed(self, seed= None):
       
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        if self._last_state != None:
            res = self._sim.reset()
            observation = self._sim.get_last_observation()
            return observation
        else:
            return None

    def _step(self, action):
      
        response = self._sim.step(action, 1)

        if response is None:
            print("error")
            sys.exit()

        # state = self._sim.step(action)
        self._last_state = self._sim.get_last_observation()  # Last observed state
        observation = self._last_state
        print(observation)
        reward, done = self.check_reward(observation)
        return observation, reward, done

    def _render(self, mode='human', close=False):
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None      # If we don't None out this reference pyglet becomes unhappy
            return
        if self._last_state is not None:
            img = self._last_state['observation']['sensors']['color']['data']
            if len(img.shape) == 2:  # assume gray
                img = np.dstack([img, img, img])
            else:  # assume rgba
                img = img[:, :, :-1]
            img = img.reshape((img.shape[1], img.shape[0], img.shape[2]))
            if mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    if self.viewer is None:
                        self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
            elif mode == 'rgb_array':
                return img

    def _close(self):
        if self._sim is not None:
            self._sim.kill()
            del self._sim

    def action_list(self):
          # # Generic actions
          # { name: 'idle' }
          # { name: 'moveTo', position: <vector3>, angle: <radians> }
          
          # # Movement
          # { name: 'forwards', strength: <multiplier> }
          # { name: 'backwards', strength: <multiplier> }
          # { name: 'strafeLeft', strength: <multiplier> }
          # { name: 'strafeRight', strength: <multiplier> }
           
          # # Rotation
          # { name: 'turnLeft', strength: <multiplier> }
          # { name: 'turnRight', strength: <multiplier> }
          
          # # Look
          # { name: 'lookUp', angle: <radians> }
          # { name: 'lookDown', angle: <radians> }
        
        
        return self.actions

    


    def check_reward(self, observation):
        reward = -1
        done = False
        position = observation['info']['agent_state']['position']
        print(position)
        print(self.goalObject)
        similarity = distance(position, self.goalObject)
        print(similarity)
        if similarity < 0.1:
            reward = 1000
        if reward == 1000:
            done = True
        return reward, done

def distance(p1, p2, number = 2):
    value = 0
    for x in range(number):
        value += (p1[x]-p2[x])**2
    value = math.sqrt(value)
    return value 