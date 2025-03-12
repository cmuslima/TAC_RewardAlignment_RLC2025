import gym
import sys
import os 

sys.path.append('../Domains/gym-hungry-thirsty/gym_hungry_thirsty/envs')
from hungry_thirsty_env import compute_reward
from utils import inputted_reward_function
import hungry_thirsty_env
sys.path.append('../RL_algorithms')
import A2C
import PPO
import DDQN
import Q_learning
import Sarsa
import Expected_Sarsa
import utils
"""
Given an env, hyperparameters, and reward_fn, train an agent and return the cumulative fitness over training
"""


def a2c(env, hyper_params, reward_fn, new_water_food_loc):
    """
    Construct an A2C agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return A2C.create_a2c_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False, new_water_food_loc=new_water_food_loc)


def ppo(env, hyper_params, reward_fn, new_water_food_loc):
    """
    Construct a PPO agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return PPO.create_ppo_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False, new_water_food_loc=new_water_food_loc)


def ddqn(env, hyper_params, reward_fn, new_water_food_loc):
    """
    Construct a DQN agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return DDQN.create_ddqn_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, plot_results=False, new_water_food_loc=new_water_food_loc)


def q_learn(env, hyper_params, reward_fn,dir, new_water_food_loc=False, train=True):
    """
    Construct a Q-learning agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return Q_learning.create_q_learning_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, dir=dir, new_water_food_loc=new_water_food_loc, train=train)
def sarsa(env, hyper_params, reward_fn,dir, new_water_food_loc=False, train=True):
    """
    Construct a Q-learning agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return Sarsa.create_sarsa_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, dir=dir, new_water_food_loc=new_water_food_loc, train=train)
def expected_sarsa(env, hyper_params, reward_fn,dir, new_water_food_loc=False, train=True):
    """
    Construct a Q-learning agent.
    :param env: the OpenAI gym environment
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn: a function of (s, a, s')
    :return:
    """
    return Expected_Sarsa.create_expected_sarsa_agent(env=env, hyper_params=hyper_params, user_reward_fn=reward_fn, dir=dir, new_water_food_loc=new_water_food_loc, train=train)

def make_env(env_name, env_size, seed):
    env = gym.make(env_name, size=env_size, seed=seed)
    return env
def eval_reward_fn(seed, alg, hyper_params, reward_fn_params, env_name='hungry-thirsty-v0', num_trials=10, env_size=(4,4), new_water_food_loc=False, dir=None, photo_dir=None):
    """
    Evaluate a reward function
    :param alg: string, the choice of algorithm (e.g., "A2C", "Q_learn", "DDQN", or "PPO")
    :param hyper_params: a dict of hyperparameters for the learning alg
    :param reward_fn_params: either a dict or list of reward fn parameters
    :param env_name: string, the name of the openai gym environment
    :param num_trials: int, the number of trials to run
    :param env_size: tuple (int, int), the size of the environment
    :return:
    """


    def user_reward_fn(state, action, new_state):
        """
        wrapper for compute_reward function in r(s, a, s') form,
        which instantiates reward params
        :param state: dict, the state
        :param action: int, the action
        :param new_state: dict, the new state
        :return: float, the reward
        """
        try:
            #print('old place')
            #print('reward_fn_params', reward_fn_params, type(reward_fn_params))
            return compute_reward(reward_fn_params=reward_fn_params, state=state)

        except:
            #print('do I ever get here')
            #print('reward_fn_params', reward_fn_params, type(reward_fn_params))

            return inputted_reward_function(reward_fn_params, state)


    env = gym.make(env_name, size=env_size, dir=photo_dir)
    print('env size', env_size)
    env.set_seed(seed)
    print('seed', seed)
    #print('inside eval_reward_fn')


    if "env_timesteps" in hyper_params.keys():
        env.update_step_limit(hyper_params["env_timesteps"])


    env.reset(new_water_food_loc=new_water_food_loc) # the food and water locations only change per seed, not per episode 

    print(alg, 'alg')
    if alg == "Q_learn":
        a, results_over_time = q_learn(env=env,
                                        hyper_params=hyper_params,
                                        reward_fn=user_reward_fn, dir=dir, new_water_food_loc=new_water_food_loc)
        fitness_over_time = [x[1]['avg_fitness'] for x in results_over_time]
        undiscounted_return_over_time = [x[1]['avg_undiscounted_return'] for x in results_over_time]
        discounted_return_over_time = [x[1]['avg_discounted_return'] for x in results_over_time]
        final_fitness = fitness_over_time[-1]
        traj = [x[1]['all_trajectory'] for x in results_over_time]

        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)
    if alg == "Sarsa":
        a, results_over_time = sarsa(env=env,
                                        hyper_params=hyper_params,
                                        reward_fn=user_reward_fn, dir=dir, new_water_food_loc=new_water_food_loc)
        fitness_over_time = [x[1]['avg_fitness'] for x in results_over_time]
        undiscounted_return_over_time = [x[1]['avg_undiscounted_return'] for x in results_over_time]
        discounted_return_over_time = [x[1]['avg_discounted_return'] for x in results_over_time]
        final_fitness = fitness_over_time[-1]
        traj = [x[1]['all_trajectory'] for x in results_over_time]

        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)

    if alg == "Expected_Sarsa":
        a, results_over_time = expected_sarsa(env=env,
                                        hyper_params=hyper_params,
                                        reward_fn=user_reward_fn, dir=dir, new_water_food_loc=new_water_food_loc)
        fitness_over_time = [x[1]['avg_fitness'] for x in results_over_time]
        undiscounted_return_over_time = [x[1]['avg_undiscounted_return'] for x in results_over_time]
        discounted_return_over_time = [x[1]['avg_discounted_return'] for x in results_over_time]
        final_fitness = fitness_over_time[-1]
        traj = [x[1]['all_trajectory'] for x in results_over_time]

        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)

    elif alg == "A2C":
        _, undiscounted_return_over_time, fitness_over_time, discounted_return_over_time, traj, final_fitness = a2c(env=env,
                                        hyper_params=hyper_params,
                                        reward_fn=user_reward_fn, new_water_food_loc=new_water_food_loc)
        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)
    elif alg == "DDQN":
        _, _, undiscounted_return_over_time, fitness_over_time, discounted_return_over_time, traj, final_fitness = ddqn(env=env,
                                            hyper_params=hyper_params,
                                            reward_fn=user_reward_fn, new_water_food_loc=new_water_food_loc)
        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)

    elif alg == "PPO":
        print('alg', alg)
        _, undiscounted_return_over_time, fitness_over_time, discounted_return_over_time, traj, final_fitness = ppo(env=env,
                                        hyper_params=hyper_params,
                                        reward_fn=user_reward_fn, new_water_food_loc=new_water_food_loc)

        utils.save(final_fitness, f'average_final_fitness_{seed}',dir)

    return fitness_over_time, undiscounted_return_over_time, discounted_return_over_time, traj