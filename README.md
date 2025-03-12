## Install Virtual Env

* We use conda virtaul environments. Documentation here: https://docs.anaconda.com/miniconda/
* conda create --name reward_design python=3.10
* conda activate reward_design
* pip install -r requirements.txt 

## Install Virual Env on CC
* module load StdEnv/2020
* module load python/3.8
* virtualenv --no-download reward_design
* source reward_design/bin/activate
* pip install -r requirements_cc.txt 

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
  In User_Studies/Expert-User-Study/user_tests/final_reward_fns.csv contains the reward functions the participants submitted. These are the reward functions we assess the alignment of.


* ```RL_algorithms```
  * This directory contains all of the code for training the agents. In particular, there are implementations for Q-learning, PPO, A2C, DDQN, and Value Iteration in this directory. 

  
* ```Measuring Alignment Experiments```

  * Step 1:
    * To run our alignment metric, we need to first select trajectories. To do so, we first train a Q Learning agent with the ground truth reward function (0,0,1,1). This is done in the evaluate_human_reward_functions.py file.  
    

  * Step 2:
    * Second, we have to sample trajectories from step 1, and evaluate them according to each of the human designed reward functions.
    * This is done in the rank_reward_functions.py file. 
    * I have a sample of 10 seeds of 10000 trajectories in the data file /Experiments/Ranked_Trajectories/Easy/Q_Learn/Fixed_Env/anked_data_935156_numseeds_2.pkl
.

  * Step 3:
    * Perform the kendall tau measure in using the data from Step 2. This is done in the get_alignment.py file. I also have some plotting code in the plot_alignment.py file

* ```Understanding how trajectories influence alignment metric experiments: work in progress```
 
  * Our proposed alignment metric involves selecting trajectories for the human to rank. However, we don't have a complete understanding of what trajectories we should select. 
  * One option is to greedily select trajectories that maximize the relationship between performance and alignment. 
  * Why would we do that? Generally, a good property of an alignment metric is that it is predictive of performance (i.e, if you trained an RL agent with that reward function). If the alignment is high, then the policy should be good, and vice verse. 
  * The greedy_trajectory_selection.py file begins to perform this analysis*

