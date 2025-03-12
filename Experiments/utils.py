import pandas as pd
import json
import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../RL_algorithms')
from default_parameters import default_param_lookup

def get_min_keys(dictionary):
    """
    Returns a list of all keys with the minimim value in a dictionary.

    Args:
    dictionary (dict): A dictionary where keys are identifiers and
                            values are numerical or comparable values.

    Returns:
    list: A list of all keys with the minimim value.
    """
    min_value = min(dictionary.values())
    min_keys = [key for key, value in dictionary.items() if value == min_value]
    return min_keys, min_value

def get_max_keys(dictionary):
    """
    Returns a list of all keys with the maximum value in a dictionary.

    Args:
    dictionary (dict): A dictionary where keys are identifiers and
                            values are numerical or comparable values.

    Returns:
    list: A list of all keys with the maximum value.
    """
    max_value = max(dictionary.values())
    max_keys = [key for key, value in dictionary.items() if value == max_value]
    return max_keys, max_value
def combine_dicts(list_of_dicts):
  """
  Combines a list of dictionaries with the same keys into a single dictionary.

  The values in the resulting dictionary are lists containing all the values 
  from the corresponding keys in the input dictionaries.

  Args:
    list_of_dicts: A list of dictionaries with the same keys.

  Returns:
    A single dictionary with the combined values.
  """
  combined_dict = {}
  for key in list_of_dicts[0]:  # Assuming all dicts have the same keys
    combined_dict[key] = []
    for d in list_of_dicts:
      combined_dict[key].extend(d[key])
  return combined_dict


def multiply_dict_by_scalar_comp(my_dict, scalar):
  """Multiplies all values in a dictionary by a scalar using comprehension.

  Args:
      my_dict: The dictionary to modify.
      scalar: The scalar to multiply by.

  Returns:
      A new dictionary with the multiplied values.
  """
  return {key: value * scalar for key, value in my_dict.items()}
def is_list_in_list_exact(outer_list, inner_list):
  """Checks if a list is present within another list as an exact sublist.

  Args:
      outer_list: The outer list to search.
      inner_list: The inner list to find.

  Returns:
      True if the inner list is found as an exact sublist, False otherwise.
  """
  return inner_list in [sublist for sublist in outer_list if sublist == inner_list]
def plot_performance(filename, selected_hyper_params, reward_fn_params, data, y_label, filename_pdf):

    

        f, ax = plt.subplots()
        data = np.array(data) #before it looks like we were plotting the final return for each seed?
        print(np.shape(data), 'np.shape(data)')

        average_data_over_time = np.mean(data, axis=0)
        ci = 1.96 * np.std(data, axis=0) / np.sqrt(len(data))

        plt.plot(np.arange(0, selected_hyper_params["num_episodes"]), average_data_over_time)
        plt.fill_between(np.arange(0, selected_hyper_params["num_episodes"]), average_data_over_time - ci, average_data_over_time + ci, alpha=0.25)
        plt.ylabel(f"{y_label}")
        plt.xlabel("Episode")
        plt.savefig(filename_pdf)
        plt.close()



def inputted_reward_function(reward_func, obs):

    reward_func = reward_func.replace("\'", "\"")
    reward_func = json.loads(reward_func)


    hungry = obs['hungry']
    thirsty = obs['thirsty']
    if hungry and thirsty:
        reward = reward_func['hungry and thirsty'] 
    if hungry and not thirsty:
        reward = reward_func['hungry and not thirsty']
    if not hungry and thirsty:
        reward = reward_func['not hungry and thirsty']
    if not hungry and not thirsty:
        reward = reward_func['not hungry and not thirsty']
    return reward


def turn_str_to_dict(string):
    string = string.replace("\'", "\"")
    dictionary = json.loads(string)
    return dictionary


def get_all_reward_functions_per_user(specific_id):
    user_reward_functions_considered = []
    dict_versions_reward_functions = []
    user_reward_data = pd.read_csv(f'{dir}/Old_User_Studies/Expert-User-Study/user_tests/{specific_id}/{specific_id}.csv')
    reward_funcs = user_reward_data['reward_fn'].values
    reward_scaling = user_reward_data['hyper_params'].values

    for reward_func, reward_scale in zip(reward_funcs, reward_scaling):
        
        reward_func_updated = np.array(list(turn_str_to_dict(reward_func).values()))
        reward_scale_updated = turn_str_to_dict(reward_scale)['reward_scaling_factor']


        reward_func_updated*=reward_scale_updated
        


        reward_func_dict = multiply_dict_by_scalar_comp(turn_str_to_dict(reward_func).copy(), reward_scale_updated)

        if is_list_in_list_exact(user_reward_functions_considered, list(reward_func_updated)):
            print(f'{list(reward_func_updated)} already in {user_reward_functions_considered}')
            print('\n')
            pass
        else:
            print(f'adding {list(reward_func_updated)} to {user_reward_functions_considered}')
            user_reward_functions_considered.append(list(reward_func_updated))
            print('\n')
            dict_versions_reward_functions.append(reward_func_dict)

    print(user_reward_functions_considered)
    print(dict_versions_reward_functions, len(dict_versions_reward_functions))
    return user_reward_functions_considered, dict_versions_reward_functions

def get_user_reward_functions(use_all=False):
   
    final_user_reward_data = pd.read_csv(f"{os.getcwd()}/User_Study_Data/human_reward_fns.csv")
    final_user_reward_funcs = final_user_reward_data['Reward Fn'].values
    user_ids = final_user_reward_data['User'].values


    user_reward_functions_considered = []
    dict_versions_reward_functions = []
    if use_all:
        for id in user_ids:
            user_reward_functions_considered.append(final_user_reward_funcs[id])

            string_format_reward = turn_str_to_dict(final_user_reward_funcs[id])
            dict_versions_reward_functions.append(tuple(list(string_format_reward.values())))   
    else:
        for id in user_ids:
            if id >=12 and id <=29:
                user_reward_functions_considered.append(final_user_reward_funcs[id])

                string_format_reward = turn_str_to_dict(final_user_reward_funcs[id])
                dict_versions_reward_functions.append(tuple(list(string_format_reward.values())))
                #print(dict_versions_reward_functions)



    fitness =  str({'hungry and thirsty': 0, 'hungry and not thirsty': 0, 'not hungry and thirsty': 1.0, 'not hungry and not thirsty': 1.0})
    user_reward_functions_considered.append(fitness)
    dict_versions_reward_functions.append((0, 0, 1.0, 1.0))


    
    return user_reward_functions_considered, dict_versions_reward_functions
def get_selected_agent(designer_selected=False, alg='Q_Learn', specific_id=None):

    dir = os.getcwd()
    dir = dir.replace('Experiments', '')
    if designer_selected:
        print(f'{dir}/Old_User_Studies/Expert-User-Study/user_tests/{specific_id}/{specific_id}.txt')
        with open(f'{dir}/Old_User_Studies/Expert-User-Study/user_tests/{specific_id}/{specific_id}.txt','r') as input:
            lines = input.readlines()
        selected_agent = int(lines[0].replace("Selected agent:", ""))
        
        hyperparameters_details = pd.read_csv(f'{dir}/Old_User_Studies/Expert-User-Study/user_tests/{specific_id}/{specific_id}.csv')

        alg = hyperparameters_details['alg'].values[selected_agent]
        hyper_params = hyperparameters_details['hyper_params'].values[selected_agent]
        hyper_params = hyper_params.replace("\'", "\"")
        hyperparameter_values = json.loads(hyper_params)
        print('Using the hyperparameters the designer chose')
    else:
        alg = alg
        hyperparameter_values = None
    

    return alg, hyperparameter_values
def get_specific_user_reward_function(specific_id):

   
    # final_user_reward_data = pd.read_csv(f"../Old_User_Studies/Expert-User-Study/user_tests/final_reward_fns.csv")
    final_user_reward_data = pd.read_csv(f"{os.getcwd()}/User_Study_Data/human_reward_fns.csv")

    final_user_reward_funcs = final_user_reward_data['Reward Fn'].values
    user_ids = final_user_reward_data['User'].values



    user_reward_functions_considered = []
    dict_versions_reward_functions = []
    for id in user_ids:
        if str(id) == str(specific_id):
            user_reward_functions_considered.append(final_user_reward_funcs[id])

            string_format_reward = turn_str_to_dict(final_user_reward_funcs[id])
            dict_versions_reward_functions.append(tuple(list(string_format_reward.values())))
        
        


    if len(user_reward_functions_considered) == 0:   
        if specific_id == -1:
            fitness =  str({'hungry and thirsty': 0, 'hungry and not thirsty': 0, 'not hungry and thirsty': 1.0, 'not hungry and not thirsty': 1.0})
            user_reward_functions_considered = [(fitness)]
            dict_versions_reward_functions =[(0, 0, 1.0, 1.0)]

    return user_reward_functions_considered, dict_versions_reward_functions

def save(data, file_name, dir):
    import pickle
    with open(f"{dir}/{file_name}.pkl", 'wb') as file:
        pickle.dump(data, file)



def get_pickle_data(file_name):
    with open(file_name,'rb') as input:
        data = pickle.load(input)
    return data

def get_hyperparams(alg, hyperparameter_values=None, new_water_food_loc=False):
    selected_hyper_params = default_param_lookup[alg].copy()

    
    if hyperparameter_values == None:
        pass
    else:
        for hyperparameter in list(hyperparameter_values.keys()):
            if hyperparameter == 'num_episodes':
                continue
            selected_hyper_params[hyperparameter] = hyperparameter_values[hyperparameter]



    if new_water_food_loc:
        env_type = "Hard"
    else:
        env_type = 'Easy'
    return selected_hyper_params, env_type
def make_dir(selected_hyper_params, BASEDIR):

    local_dir = BASEDIR
    os.makedirs(f'{local_dir}', exist_ok=True) 
    
    for h in list(selected_hyper_params.keys()):
        local_dir +=  f'{h}_{selected_hyper_params[h]}/'
 
    local_dir = local_dir.replace("neural_net_hidden_size", "nn_size")
    local_dir = local_dir.replace("plotting_steps", "plt")
    local_dir = local_dir.replace("num_environments", "envs")
    local_dir = local_dir.replace("epsilon", "eps")
    local_dir = local_dir.replace("exp_replay_size", "rply_sz")
    local_dir = local_dir.replace("sync_frequency", "sync_frq")
    os.makedirs(f'{local_dir}', exist_ok=True) 
    return local_dir

def get_files_by_prefix(directory, prefix):
  """
  Gets all files from a directory that start with a specific prefix.

  Args:
      directory: The path to the directory.
      prefix: The prefix to match in filenames.

  Returns:
      A list of filenames starting with the prefix.
  """
  files = []
  for filename in os.listdir(directory):
    if filename.startswith(prefix):
      files.append(filename)
  return files


def create_timestamped_dir(parent_dir):
  """
  Creates a directory with the current date and time as its name.

  Returns:
      The path to the created directory.
  """
  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Get current date and time
  directory_name = f"timestamped_dir_{now}"  # Create directory name with prefix
  directory_path = os.path.join(parent_dir, directory_name)  # Get full path

  # Create directory if it doesn't exist, avoiding potential errors
  os.makedirs(directory_path, exist_ok=True)

  return directory_path

def get_id(id):
    """This has to be done b/c the data set of human designed reward functions I am using has a mixture of 
    reward functions from the hungry thirsty env of grid size 6x6 and grid size 4 by 4.
    
    The first 12 reward functions are from the latter (which the authors mentioned weren't very good b/c users had trouble 
    writing down the reward function in the larger grid).
    
    In my experiments, I only consider the reward functions for the user study where participants designed the reward func
    for the 4x4 version of the env"""
    return id - 12
# dir = os.getcwd()
# dir = dir.replace('Experiments', '')
# get_all_reward_functions_per_user(12)

def get_dir(alg, reward_function, dir=None):
    if dir == None:
        alg, hyperparameter_values = get_selected_agent(alg=alg)
        selected_hyper_params, env_type = get_hyperparams(alg=alg, hyperparameter_values=hyperparameter_values)
        local_dir = make_dir(selected_hyper_params, f'{os.getcwd()}/Experiments/Env_{env_type}/Fixed_Env/')
        local_dir += f'{reward_function}'
    else:
        return dir 
    return local_dir

def save_as_csv(filename, data):
    print(type(data))
    if str(type(data)) == "<class 'numpy.ndarray'>":
        print('inside here')
        np.savetxt(filename, data, delimiter=",")
    else:
        # Open the file in write mode ('w')
        with open(filename, 'w', newline='') as csvfile:
            # Create a csv writer object
            writer = csv.writer(csvfile)

            # Write the data to the csv file
            writer.writerows(data)

def get_trajectories(dict_version_reward_params = '(0, 0, 1.0, 1.0)', num_runs=None, dir=None):
    """
    This function retrieves trajectories for a specific reward function, hyperparameter configuration,
    and environment type.

    Args:
        dict_version_reward_params (str): Dictionary version of the reward function parameters.
        alg (str): Reinforcement learning algorithm (e.g., DDQN, A2C).
        num_runs (int): Number of simulation runs.

    Returns:
         a list of all trajectories (list), and the directory where the trajectories were loaded from (str).
    """


    all_trajectories = []
    for seed in np.arange(0, num_runs):
        with open(f'{dir}/trajectories_{dict_version_reward_params}_seed{seed}.pkl','rb') as input:
            trajectories = pickle.load(input)
        try:
            trajectories = np.array(trajectories)
        except:
            trajectories[-1] = trajectories[-1][0]
            trajectories = np.array(trajectories)
           
        all_trajectories.append(trajectories)

    #print('shape of all_trajectories', np.shape(all_trajectories))
     
    return all_trajectories, dir
