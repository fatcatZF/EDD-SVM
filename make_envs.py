"""
create undrifted and the corresponding drifted environments
env1: undrifted production environment
env2: drifted production environment
"""
import numpy as np 
import math 

import gymnasium as gym 
from gymnasium import Wrapper
from gymnasium.envs.classic_control.cartpole import CartPoleEnv 
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from typing import  Optional





def make_acrobot():
    env1 = gym.make("Acrobot-v1") 
    env2 = gym.make("Acrobot-v1")
    env2.unwrapped.book_or_nips = "nips" 
    return env1, env2 


def make_cartpole():
    class RewardShapingWrapper(Wrapper):
      def __init__(self, env):
        super().__init__(env)

      def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Modify the reward based on observation
        reward = self.modify_reward(observation, reward)
        return observation, reward, terminated, truncated, info

      def modify_reward(self, observation, reward):
        # Define custom reward modification logic
        # For example, add a penalty if the pole angle is too high
        pole_angle = observation[2]
        reward = np.cos(pole_angle)
        return reward
    class CartPoleEnvDrifted(CartPoleEnv):
      def __init__(self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None):
        super(CartPoleEnvDrifted, self).__init__()
        self.force_mag = 11.5 # increase the force mag from 10.0 to 11.5
    gym.register("CartPoleDrifted-v1",
                 CartPoleEnvDrifted, max_episode_steps=500) 
    env2 = RewardShapingWrapper(gym.make("CartPoleDrifted-v1"))
    env1 = RewardShapingWrapper(gym.make("CartPole-v1"))
    return env1, env2    


def make_lunarlander():
    env1 = gym.make("LunarLander-v2") 
    env2 = gym.make("LunarLander-v2", 
                    enable_wind=True,
                    wind_power=5.) # add wind with power=5.
    return env1, env2 


def make_mountaincar():
    class MountainCarEnvDrifted(MountainCarEnv):
      def __init__(self):
        super().__init__()
        self.force = 0.0009  # The engine force of the car decreased from 0.001 to 0.0009
    gym.register("MountainCarDrifted-v0",
             MountainCarEnvDrifted,
             max_episode_steps=999)
    env2 = gym.make("MountainCarDrifted-v0")
    env1 = gym.make("MountainCar-v0")
    return env1, env2 
    

def make_mountaincar_continuous():
    class Continuous_MountainCarEnvWithWind(Continuous_MountainCarEnv):
      def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, wind_direction="left",
               windpower=0.35):
        super().__init__(render_mode, goal_velocity)
        self.wind_direction = wind_direction
        self.windpower = windpower

      def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        windforce = max(np.random.normal(self.windpower, 0.015),0.)

        #velocity += force * self.power - 0.0025 * math.cos(3 * position)

        if self.wind_direction == "left":
          # Left Wind
          velocity += (force-windforce) * self.power - 0.0025 * math.cos(3 * position)
        else:
          # Right Wind
          velocity += (force+windforce) * self.power - 0.0025 * math.cos(3 * position)


        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, terminated, False, {}


    gym.register("MountainCarContinuousLeftWind-v0",
             Continuous_MountainCarEnvWithWind,
             max_episode_steps=999)
    env2 = gym.make("MountainCarContinuousLeftWind-v0")
    env1 = gym.make("MountainCarContinuous-v0")

    return env1, env2 


def make_pendulum():
    env1 = gym.make("Pendulum-v1") 
    env2 = gym.make("Pendulum-v1", g=11.5) # the gravity increases from 10.0 to 11.5
    return env1, env2 





if __name__ == "__main__":
     env1, env2 = make_acrobot()
     env1, env2 = make_cartpole()
     env1, env2 = make_lunarlander()
     env1, env2 = make_mountaincar()
     env1, env2 = make_mountaincar_continuous()
     env1, env2 = make_pendulum()