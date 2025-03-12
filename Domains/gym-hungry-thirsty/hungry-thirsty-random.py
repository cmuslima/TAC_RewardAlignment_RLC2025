import gym
import time
import sys
import pickle
import os
from gym_hungry_thirsty.envs.hungry_thirsty_env import Available_Actions
from copy import deepcopy
size = (4,4)
env = gym.make('gym_hungry_thirsty:hungry-thirsty-v0', size=size)
reward = None
return_val = None

NUM_EPISODES = 10

def barto_singh_reward(obs):
    hungry = obs['hungry']
    thirsty = obs['thirsty']

    if hungry and thirsty:
        reward = -0.05
    if hungry and not thirsty:
        reward = -0.01
    if not hungry and thirsty:
        reward = 1
    if not hungry and not thirsty:
        reward = 0.5
    return reward
for _ in range(NUM_EPISODES):
    score = 0
    total_reward = 0
    obs = env.reset(food_loc=(0,0), water_loc=(size[0]-1,0))


    trajectory = []

    while True:
        if not obs["hungry"]:
            score += 1
        env.render(score=score)
        time.sleep(0.01)

        
        
        
        action = Available_Actions.random()

        #print(f'You took action {action}')
        trajectory.append((deepcopy(obs), action))

        # construct the trajectory for a return fn score
        new_obs, _, done, info = env.step(action)
        reward = barto_singh_reward(obs)

        obs = new_obs
        total_reward+=reward

        if done:
            rewards = {'Fitness': score, 'Shaped Reward': total_reward}
            trajectory.append(rewards)
            print(f'Fitness Score: {score}')
            print(f'Shaped reward score: {total_reward}')
            os.makedirs(f'./Assets', exist_ok=True)  # creates the dir if it doesn't exist
            with open("Assets/random_trajectories.txt", 'wb') as file:
                pickle.dump(trajectory, file)
            print("Resetting the environment")
            
            obs = env.reset()
            break
        
