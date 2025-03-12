from itertools import product
import eval_reward_fn
import numpy as np
import csv
import gym
import os
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append('../RL_algorithms')
from Utils import tsplot, find_filenames_with_extension
from default_parameters import default_param_lookup
import argparse
import utils
import json
import pandas as pd
from pathlib import Path

"""
This file is the backbone for computing true cumulative performance for each configuration.
"""

gym.logger.set_level(40)
BASE_DIR = "Experiments/"


def main(alg, hyperparameter_values, reward_fns, dict_versions_reward_functions, new_water_food_loc, photo_dir, seed, env_size=(4,4)):
    """


    :param alg: string
    :param hyperparameter_values: dict of hyperparameters
    :return:
    """



    selected_hyper_params, env_type = utils.get_hyperparams(alg, hyperparameter_values, new_water_food_loc)
    local_dir = utils.make_dir(selected_hyper_params, f'{BASE_DIR}/Env_{env_type}/UserStudyRewardFuncs/')
    #0th row is done
    #2nd row is done, 
    existing_files = find_filenames_with_extension(local_dir, ".csv")
    # print('got local dir', local_dir)

    def eval_params(reward_fn_params, dict_version_reward_params, env_size=env_size):
        """
        Train an RL agent using this specific reward function parameters

        :param reward_fn_params, dict_version_reward_params:
        :return:
        """


        """Setting up the directory to save our data in """
        file_extension = "reward_fn_perf_{}.csv".format(str(dict_version_reward_params))
        if file_extension in existing_files:
            return
        dir=f'{local_dir}/{str(dict_version_reward_params)}'

        os.makedirs(f'{dir}', exist_ok=True) 
        filename = dir + "/" + file_extension
        print("Reward fn: ", reward_fn_params, "\n\n")

        fitness_all_seeds = []
        return_all_seeds = []

        
        """Training the RL agent"""
        print(f'trial number {seed}')
        fitness_all_episodes, undiscounted_return_all_episodes, _, trajectories = eval_reward_fn.eval_reward_fn(seed=seed, alg=selected_hyper_params["alg"], env_size=env_size,
                                                                reward_fn_params=reward_fn_params,
                                                                hyper_params=selected_hyper_params,
                                                                num_trials=selected_hyper_params["num_environments"], new_water_food_loc=new_water_food_loc, dir=dir, photo_dir=photo_dir)
        
      
        """Saving relevant information"""
        utils.save(fitness_all_episodes, f'fitness_all_episodes_{dict_version_reward_params}_seed{seed}',dir)
        utils.save(undiscounted_return_all_episodes, f'undiscounted_return_all_episodes_{dict_version_reward_params}_seed{seed}',dir)
        utils.save(trajectories, f'trajectories_{dict_version_reward_params}_seed{seed}',dir)
        fitness_all_seeds.append(fitness_all_episodes)
        return_all_seeds.append(undiscounted_return_all_episodes)



        """Plotting"""
        utils.plot_performance(filename, selected_hyper_params, reward_fn_params, fitness_all_seeds, 'Fitness', dir + f"/reward_fn_fitness_learning_curve_{str(dict_version_reward_params)}_{seed}.png")
        utils.plot_performance(filename, selected_hyper_params, reward_fn_params, return_all_seeds, 'Undiscounted Return', dir + f"/reward_fn_undiscounted_return_learning_curve_{str(dict_version_reward_params)}_{seed}.png")

        print("Reward Fn Assessment Done")
        print("\n\n")




    for reward_fn in zip(reward_fns, dict_versions_reward_functions):
        print(f'using reward function = {reward_fn}')
        string_reward_fn = reward_fn[0]
        dict_version_reward_fn = reward_fn[1]
        print(string_reward_fn, dict_version_reward_fn)
        eval_params(string_reward_fn, dict_version_reward_fn)
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--specific_id', help='choice of: id >= 12 and <=30', type=int, required=False, default=30)
    parser.add_argument('--specific_id', help='choice of: id >= 0 and <=36', type=int, required=False, default=-1)

    parser.add_argument('--seed', help='any int', type=int, required=False, default=0)
    parser.add_argument('--participant_selected', type=int, help='choice of: 1 (True) 0 (False', required=False, default=0)
    parser.add_argument('--alg',help='choice of: "Q_Learn" | "Sarsa" | "Expected_Sarsa" | "A2C" | "DDQN" | "PPO"',required=False, default = 'Q_Learn')
    parser.add_argument('--new_water_food_loc', type=int,
                    help='choice of: 1(True), 0 (False)',
                    required=False, default = 0)
    args = parser.parse_args()


    #reward_funcs_considered = [18, 32, 34, 37, 29, 28, 12, 21, 24, 17, 10, 13, 14, 15]

    reward_funcs_considered = [args.specific_id]
    for reward_func in reward_funcs_considered:

        args.specific_id = reward_func
        print(f'reward_func id = {reward_func}')
        print(f'on seed = {args.seed}')
  
        # Get the current working directory
        current_dir = Path.cwd()
        # Get the parent directory
        photo_dir = current_dir.parent

        env_size = (4,4)

        reward_function_dict_version, reward_function_list_version  = utils.get_specific_user_reward_function(args.specific_id)
        print('reward_function full name', reward_function_dict_version)

        
        if args.participant_selected:
            alg, hyperparameter_values = utils.get_selected_agent(args.participant_selected, args.alg, args.specific_id)
            print('Using the hyperparameters the designer chose')
            main(alg, hyperparameter_values, reward_function_dict_version, reward_function_list_version, args.new_water_food_loc, photo_dir,args.seed, env_size=env_size)
        else:
            alg, hyperparameter_values = utils.get_selected_agent(args.participant_selected, args.alg)
            main(alg, hyperparameter_values, reward_function_dict_version, reward_function_list_version, args.new_water_food_loc, photo_dir, args.seed, env_size=env_size)


