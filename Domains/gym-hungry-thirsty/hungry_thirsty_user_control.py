import gym
import time
import sys
import pickle
import os
sys.path.append('./gym_hungry_thirsty/envs')

from hungry_thirsty_env import Available_Actions
from copy import deepcopy

def user_control():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    size = (4,4)
    env = gym.make('hungry-thirsty-v0', size=size, dir=parent_dir)

    episode_metadata = []
    quit_user_control = False
    episode_count = 0

    while not quit_user_control:
        state_counter = {
            "not_hungry_not_thirsty": 0,
            "not_hungry_thirsty": 0,
            "hungry_thirsty": 0,
            "hungry_not_thirsty": 0
        }
        episode_count += 1
        timestep = 0
        water_loc = (0, 0)
        food_loc = (size[0] - 1, 0)
        obs = env.reset(food_loc=food_loc, water_loc=water_loc)
        action = 'w'
        done = False

        while not quit_user_control and not done:
            env.render_with_episode(state_counter=state_counter, episode=episode_count, action=action, timestep=timestep)
            time.sleep(0.2)

            while True:
                print("What action should I take?\n"
                      "w,s,a,d: up, down, left, right\n"
                      "e: eat\n"
                      "r: drink\n", 
                      "quit: to exit game control")

                action = input()
                if action not in ['w', 'a', 's', 'd', 'e', 'r', 'quit']:
                    print("Not a valid action. Try again")
                else:
                    action = {'w': 0, 'a': 2, 's': 1, 'd': 3, 'e': 4, 'r': 5}.get(action, action)

                    if action == 'quit':
                        quit_user_control = True
                    break

            if action == 'quit':
                break

            new_obs, _, done, info = env.step(action)
            obs = new_obs

            if not obs.get("hungry") and not obs.get("thirsty"):
                state_counter['not_hungry_not_thirsty'] += 1
            elif obs.get("hungry") and not obs.get("thirsty"):
                state_counter['hungry_not_thirsty'] += 1
            elif obs.get("hungry") and obs.get("thirsty"):
                state_counter['hungry_thirsty'] += 1
            elif not obs.get("hungry") and obs.get("thirsty"):
                state_counter['not_hungry_thirsty'] += 1

            timestep += 1
            if done:
                print("Resetting the environment")
                obs = env.reset(food_loc=episode_metadata["food_loc"], water_loc=episode_metadata["water_loc"])

    # ✅ CLEAN UP TKINTER WHEN USER CONTROL EXITS
    if env.canvas_root:
        try:
            env.canvas_root.quit()
            env.canvas_root.destroy()
            env.canvas_root = None
        except Exception as e:
            print(f"Warning: Failed to destroy Tkinter window - {e}")
