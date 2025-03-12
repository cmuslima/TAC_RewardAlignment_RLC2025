## Install Virtual Env

* We use conda virtaul environments. Documentation here: https://docs.anaconda.com/miniconda/
* conda create --name reward_design python=3.10
* conda activate reward_design
* pip install -r requirements.txt 

## Study Domain: Hungry Thirsty 

<img src="https://user-images.githubusercontent.com/6353393/217291559-d9db4a5c-b1df-4f3f-a2c4-5d26d6c017a7.png" alt="Hungry Thirsty" width="30%">

* We use the Hungry Thirsty domain as a testbed for studying reward design. 
* In Hungry Thirsty, food is located in one random corner; water in another.
* The goal is for the agent to eat as much as possible, but the agent can only eat if it’s not thirsty.
* If the agent drinks, it becomes not thirsty. If the agent doesn’t drink, it becomes thirsty with 10% probability.
* The optimal policy is for the agent to navigate to the water and drink whenever it's thirsty, but to navigate to the food and eat whenever it's not thirsty. 

## Codebase Structure

This work is divided into the following directories:
* ```Domains```
  * This contains the Hungry-Thirsty Domain code, which is written as an OpenAI Gym Environment.
  * Files of interest: ```Domains/gym-hungry-thirsty/gym_hungry_thirsty/envs/hungry_thirsty_env.py``` is the main implementation.
  * ```Domains/hungry-thristy-user-control.py``` lets you control the agent directly, which gives a feel for the task. 
  
* ```User_Studies```

user_study_notebook_domain_expert_version.ipynb is the UI used for the user study, with supporting functions inside user_study_interface_backend_domain_expert.py

In Experiments/Experiments/human_reward_fns.csv contains a list of human reward functions some of which are used in the human subject study.

In RewardDesignUserStudy2025/pilot_version_2_21_anonymized, we have the data from all 11 participants in the user study. 

* ```RL_algorithms```
  * This directory contains all of the code for training the agents. In particular, there are implementations for Q-learning, SARSA, Expected SARSA, PPO, A2C, DDQN, and Value Iteration in this directory. 

  We trained agents using the evaluate_human_reward_functions.py and evaluate_human_reward_functions_hp_sweep.py files. 

  ```Plotting Code```

  In the Plotting dir contains all the plotting code files used to reproduce the plots and the statistical tests.
  