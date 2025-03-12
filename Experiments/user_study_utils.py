import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import utils 
import random
import time
import gym
import sys
from termcolor import colored
import pyfiglet
import uuid
import pickle
import ast
from collections import OrderedDict
from itertools import combinations
from collections import defaultdict
from datetime import datetime
import tkinter as tk
import threading
from IPython.display import Image
from tkinter import Toplevel, Label, PhotoImage, Canvas, messagebox
from PIL import Image, ImageTk
from functools import partial
import inspect
from typing import List, Tuple, Optional


import user_study_utils
import get_alignment
sys.path.append('../Domains/gym-hungry-thirsty/gym_hungry_thirsty/envs')
import hungry_thirsty_env 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

from termcolor import colored

def random_swap_tuples(lst):
    """
    Randomly swaps the order of elements in each tuple within a given list.

    Args:
    lst (list of tuples): List containing tuples of two elements.

    Returns:
    list of tuples: List with elements randomly swapped in each tuple.
    """
    return [(b, a) if random.choice([True, False]) else (a, b) for a, b in lst]

def get_specific_reward_functions( index_pairs: Optional[List[Tuple[int, int]]] = None, random_seed=0, debug=False):
    """
    Selects specific reward functions based on provided index pairs.

    Args:
        index_pairs (Optional[List[Tuple[int, int]]]): Pairs of indices indicating which reward functions to select.

    Raises:
        ValueError: If no index pairs are provided or if `index_pairs` is invalid.
    """
    if index_pairs is None:
        raise ValueError("No index pairs provided. Please specify pairs of indices.")

    if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in index_pairs):
        raise ValueError("Index pairs must be a list of 2-tuples, e.g., [(0, 1), (2, 3)].")

    index_pairs = random_swap_tuples(index_pairs)

    # Shuffle the index pairs

    # only go back to this if the np.random.shuffle is not working
    # random_order_index_pairs = [
    # [(34, 37), (32, 29), (22, 24), (7, 21), (13, 18), (12, 21), (18, 37), (13, 14), (7, 29)],
    # [(7, 21), (22, 24), (7, 29), (13, 14), (13, 18), (32, 29), (12, 21), (18, 37), (34, 37)],
    # [(13, 14), (12, 21), (7, 29), (34, 37), (22, 24), (7, 21), (18, 37), (32, 29), (13, 18)],
    # [(22, 24), (7, 29), (32, 29), (13, 18), (12, 21), (13, 14), (7, 21), (34, 37), (18, 37)],
    # [(7, 21), (34, 37), (7, 29), (22, 24), (32, 29), (13, 14), (13, 18), (12, 21), (18, 37)],
    # [(7, 29), (22, 24), (32, 29), (13, 14), (34, 37), (12, 21), (7, 21), (18, 37), (13, 18)]
    # ]
    #index_pairs = random_order_index_pairs[random_seed]



    np.random.shuffle(index_pairs) #seems to be working on my macbook
    if debug:
        print(index_pairs)

    


    # Retrieve reward functions
    reward_functions = utils.get_user_reward_functions(use_all=True)[0]
    human_reward_functions = [
        ast.literal_eval(func) for func in reward_functions[:-1]
    ]  # Parse all but the last reward function

    # Select reward functions based on the provided index pairs
    selected_human_reward_functions = [
        [human_reward_functions[i], human_reward_functions[j]] for i, j in index_pairs
    ]


    # Set the ground truth to the last reward function
    ground_truth = tuple(ast.literal_eval(reward_functions[-1]).values())


    # Assign the selected reward functions
    return ground_truth, selected_human_reward_functions

    # def get_reward_functions(self):
    #     reward_functions = utils.get_user_reward_functions(use_all=True)[0]

    #     human_reward_functions = reward_functions[:-1]
    #     human_reward_functions = [ast.literal_eval(i) for i in human_reward_functions]


    #     self.ground_truth = tuple(ast.literal_eval(reward_functions[-1]).values())
    #     self.selected_reward_functions = random.sample(human_reward_functions, self.num_reward_functions_per_condtion*self.num_conditions)

def get_env_and_meta_data(env_seed, num_traj):
    env = gym.make('hungry-thirsty-v0', size=(4, 4), seed=env_seed,  dir=parent_dir)

    
    GRID_HEIGHT = 4
    GRID_WIDTH = 4
    if env_seed == 0:
        water_loc = (0, 0)
        food_loc = (GRID_HEIGHT - 1, 0 )
    elif env_seed == 1:
        water_loc = (0, 0)
        food_loc = (GRID_WIDTH - 1, GRID_WIDTH - 1)
    elif env_seed == 2:
        water_loc = (GRID_WIDTH - 1, 0)
        food_loc = (0, GRID_HEIGHT - 1)


    elif env_seed == 3: 
        water_loc = (0, 0)
        food_loc = (GRID_WIDTH - 1, GRID_WIDTH - 1)
    elif env_seed == 4:
        water_loc = (0, 0)
        food_loc = (GRID_WIDTH - 1, 0)


    elif env_seed  == 5:
        water_loc = (0, GRID_HEIGHT - 1)
        food_loc = (0, 0)

    elif env_seed  == 6: 
        water_loc = (GRID_WIDTH - 1, 0)
        food_loc = (GRID_WIDTH - 1, GRID_WIDTH - 1)

    elif env_seed == 7:
        water_loc = (0, GRID_HEIGHT - 1)
        food_loc = (GRID_WIDTH - 1, GRID_WIDTH - 1)


    elif env_seed  == 8:
        water_loc = (GRID_WIDTH - 1, 0)
        food_loc = (0, 0)

    elif env_seed == 9: 
        water_loc = (0, GRID_HEIGHT - 1)
        food_loc = (0, 0)


    metadata = [{"food_loc": food_loc, "water_loc": water_loc} for _ in range(num_traj)]
    return env, metadata


from collections import OrderedDict

def list_to_ranked_dict(data):
    """
    Converts a list of values into a dictionary with keys a, b, c... and ranks the values,
    returning an OrderedDict where the keys are in their original order.

    Parameters:
        data (list): A list of numeric values.

    Returns:
        OrderedDict: An OrderedDict with keys a, b, c... and ranks as values, preserving original order.
    """
    # Step 1: Create a dictionary with keys 'Traj. A', 'Traj. B', 'Traj. C', ... and values from the list
    keys = [f"Traj. {chr(65 + i)}" for i in range(len(data))]  # Generate keys
    data_dict = dict(zip(keys, data))

    # Step 2: Create a sorted list of (key, value) pairs based on values, descending
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Assign ranks, accounting for ties
    rank_dict = OrderedDict()
    current_rank = 1
    previous_value = None

    for i, (key, value) in enumerate(sorted_items):
        if i == 0 or value != previous_value:
            rank_dict[key] = current_rank  # Start new rank
        else:
            # Handle ties by assigning the same rank
            rank_dict[key] = rank_dict[sorted_items[i - 1][0]]
        
        if i < len(sorted_items) - 1 and sorted_items[i + 1][1] != value:
            current_rank += 1  # Only increment rank if the next item is not a tie
        previous_value = value  # Update the previous value to current for next iteration check

    # Step 4: Build an OrderedDict with the original key order
    ranked_dict = OrderedDict((key, rank_dict[key]) for key in keys)
    return dict(ranked_dict)




def pairwise_ranking_trajectories_interface(trajectories, current_rankings, root):
    """
    Create an interface to rank all trajectories using horizontal radio buttons with captions.

    :param trajectories: The list of trajectories.
    :param current_rankings: A dictionary of current rankings for each trajectory.
    :return: Updated rankings as a dictionary.
    """
    def submit():
        # Collect rankings from the radio buttons
        for i in range(len(trajectories)):
            updated_rankings[f"Traj. {chr(97 + i).upper()}"] = rankings[i].get()
        messagebox.showinfo("Rankings Submitted", "All rankings have been updated.")
        root.destroy()  # Close the interface to ensure it's properly closed

    num_trajectories = len(trajectories)
    #root = tk.Tk()
    root.title("Rank Trajectories")

    tk.Label(root, text="Rank Trajectories", font=("Arial", 16)).pack(pady=10)
    tk.Label(
        root,
        text=f"Select a rank for each trajectory (1 = Highest, {num_trajectories} = Lowest):",
    ).pack(pady=5)

    rankings = []
    updated_rankings = {}

    for i in range(num_trajectories):
        tk.Label(root, text=f"Trajectory {chr(97 + i).upper()}:").pack(anchor="w", padx=20)

        # Variable to hold the selected rank for this trajectory
        rank_var = tk.IntVar(value=current_rankings.get(i + 1, num_trajectories))
        rankings.append(rank_var)

        # Frame to hold the horizontal radio buttons and captions
        frame = tk.Frame(root)
        frame.pack(anchor="w", padx=40, pady=5)

        # Add the "Highest" caption
        tk.Label(frame, text="Highest", font=("Arial", 10)).pack(side="left", padx=5)

        # Create radio buttons for the ranking options
        for rank in range(1, num_trajectories + 1):  # 1 to num_trajectories
            tk.Radiobutton(
                frame,
                text=str(rank),
                variable=rank_var,
                value=rank,
                indicatoron=True,
            ).pack(side="left", padx=5)

        # Add the "Lowest" caption
        tk.Label(frame, text="Lowest", font=("Arial", 10)).pack(side="left", padx=5)

    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=10)

    # Run the Tkinter loop
    root.mainloop()
    return updated_rankings

def full_order_rank_trajectories_interface(trajectories, current_rankings, root):
    """
    Create an interface to rank all trajectories using horizontal radio buttons with captions.

    :param trajectories: The list of trajectories.
    :param current_rankings: A dictionary of current rankings for each trajectory.
    :return: Updated rankings as a dictionary.
    """
    def submit():
        # Collect rankings from the radio buttons
        for i in range(len(trajectories)):
            updated_rankings[f"Traj. {chr(97 + i).upper()}"] = rankings[i].get()
        messagebox.showinfo("Rankings Submitted", "All rankings have been updated.")
        root.destroy()  # Close the interface to ensure it's properly closed

    num_trajectories = len(trajectories)
    #root = tk.Tk()
    root.title("Rank Trajectories")

    tk.Label(root, text="Rank Trajectories", font=("Arial", 16)).pack(pady=10)
    tk.Label(
        root,
        text=f"Select a rank for each trajectory (1 = Highest, {num_trajectories} = Lowest):",
    ).pack(pady=5)

    rankings = []
    updated_rankings = {}

    for i in range(num_trajectories):
        tk.Label(root, text=f"Trajectory {chr(97 + i).upper()}:").pack(anchor="w", padx=20)

        # Variable to hold the selected rank for this trajectory
        rank_var = tk.IntVar(value=current_rankings.get(i + 1, num_trajectories))
        rankings.append(rank_var)

        # Frame to hold the horizontal radio buttons and captions
        frame = tk.Frame(root)
        frame.pack(anchor="w", padx=40, pady=5)

        # Add the "Highest" caption
        tk.Label(frame, text="Highest", font=("Arial", 10)).pack(side="left", padx=5)

        # Create radio buttons for the ranking options
        for rank in range(1, num_trajectories + 1):  # 1 to num_trajectories
            tk.Radiobutton(
                frame,
                text=str(rank),
                variable=rank_var,
                value=rank,
                indicatoron=True,
            ).pack(side="left", padx=5)

        # Add the "Lowest" caption
        tk.Label(frame, text="Lowest", font=("Arial", 10)).pack(side="left", padx=5)

    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=10)

    # Run the Tkinter loop
    root.mainloop()
    return updated_rankings

def compute_trajectory_rewards_and_rankings( trajectories, reward_funcs):
    """
    Computes the total reward and ranking for each trajectory with respect to each reward function.
    
    Parameters:
        trajectories (list): A list of trajectories where each trajectory contains state-action pairs.
        reward_funcs (list): A list of reward functions to evaluate.
    
    Returns:
        dict: A dictionary containing:
            - "returns": A list of dictionaries with total rewards for each reward function.
            - "rankings": A list of dictionaries with rankings for each reward function.
    """
    trajectory_returns = {}
    trajectory_rankings = {}


    for idx, specific_reward_function in enumerate(reward_funcs):
        # Compute total returns for all trajectories
        returns = []
        for traj_idx, traj in enumerate(trajectories):
            total_reward = 0
            for obs_a in traj:
                obs = obs_a[0]
                hungry = obs['hungry']
                thirsty = obs['thirsty']
                if hungry and thirsty:
                    reward = specific_reward_function['hungry and thirsty'] 
                if hungry and not thirsty:
                    reward = specific_reward_function['hungry and not thirsty']
                if not hungry and thirsty:
                    reward = specific_reward_function['not hungry and thirsty']
                if not hungry and not thirsty:
                    reward = specific_reward_function['not hungry and not thirsty']
                total_reward += reward
            returns.append(total_reward)

        # Store returns in dictionary
        trajectory_returns[str(specific_reward_function)] = returns
        rankings = list_to_ranked_dict(trajectory_returns[str(specific_reward_function)])
        trajectory_rankings[str(specific_reward_function)] = rankings
    return {
        "returns": trajectory_returns,
        "rankings": trajectory_rankings
    }
def save_intermediate_data(data_dir, data_to_save):
    filename = f"{data_dir}/preference_data.pkl"
    with open(filename, "wb") as file:
        pickle.dump(data_to_save, file)

def get_data_dir(name):
    date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f'RewardDesignUserStudy2025/{date}/user_{name}', exist_ok=True)
    
    # Save data to a pickle file
    return f'RewardDesignUserStudy2025/{date}/user_{name}'


def convert_to_rankings(sorted_trajectories, user_rankings):
    """
    Converts the sorted list of trajectories into a ranking dictionary.

    Args:
        sorted_trajectories (list): List of trajectory indices in ranked order.
        user_rankings (dict): Dictionary of pairwise comparison results.

    Returns:
        OrderedDict: Dictionary mapping trajectory labels to their rank.
    """
    rank_dict = OrderedDict()
    current_rank = 1

    for i, trajectory in enumerate(sorted_trajectories):
        trajectory_label = f'Traj. {chr(97 + trajectory).upper()}'

        if i == 0 or user_rankings.get((sorted_trajectories[i - 1], trajectory)) is not None:
            rank_dict[trajectory_label] = current_rank
        else:
            # Assign the same rank for tied trajectories
            rank_dict[trajectory_label] = rank_dict[f'Traj. {chr(97 + sorted_trajectories[i - 1]).upper()}']

        if i == len(sorted_trajectories) - 1 or user_rankings.get((trajectory, sorted_trajectories[i + 1])) is not None:
            current_rank += 1

    return rank_dict


def run_quiz(root):
    root.title("Check-in: Hungry-Thirsty Domain Understanding")
    root.config(bg="azure")  # Light background

    # Define questions and choices
    questions = [
        {
            "question": "What is the main goal of the agent in the hungry-thirsty domain?",
            "choices": ["A. To drink as much water as possible",
                        "B. To avoid barriers and move around the grid",
                        "C. To eat as much as possible while managing thirst"],
            "answer": "C. To eat as much as possible while managing thirst"
        },
        {
            "question": "Why can't the agent stay at the food location and keep eating?",
            "choices": ["A. The food disappears after being eaten once",
                        "B. The agent can become thirsty again with 10% probability, in which case the agent needs to drink water before it can eat again",
                        "C. The agent can eat continuously without any restrictions"],
            "answer": "B. The agent can become thirsty again with 10% probability, in which case the agent needs to drink water before it can eat again"
        },
        {
            "question": "What happens if the agent tries to eat while thirsty?",
            "choices": ["A. The action fails, meaning the agent did not eat and remains hungry.",
                        "B. It successfully eats",
                        "C. The agent moves randomly to another location"],
            "answer": "A. The action fails, meaning the agent did not eat and remains hungry."
        },
        {
            "question": "What happens when the agent is both hungry and thirsty at the same time?",
            "choices": ["A. The cell the agent is in turns green",
                        "B. The cell the agent is in turns blue",
                        "C. The agent remains in a white cell"],
            "answer": "C. The agent remains in a white cell"
        }
    ]

    # Store user responses
    responses = {}

    # Create variables to track selected answers
    selected_answers = {i: tk.StringVar(value="") for i in range(len(questions))}

    # Function to handle submission
    def submit_quiz():
        unanswered = [i for i, q in enumerate(questions) if not selected_answers[i].get()]

        if unanswered:
            confirm = messagebox.askyesno("Unanswered Questions", 
                                          "You have unanswered questions. Are you sure you want to submit?")
            if not confirm:
                return

        # Store user responses before exiting
        for i, q in enumerate(questions):
            responses[q["question"]] = selected_answers[i].get() if selected_answers[i].get() else "Skipped"

        # Generate feedback for incorrect answers
        incorrect_feedback = []
        for i, q in enumerate(questions):
            user_answer = selected_answers[i].get()
            if user_answer and user_answer != q["answer"]:  # User answered incorrectly
                incorrect_feedback.append(f"Question: {q['question']}\nYour Answer: {user_answer}\nCorrect Answer: {q['answer']}")

        # Display results
        if incorrect_feedback:
            feedback_text = "\n\n".join(incorrect_feedback)
            messagebox.showinfo("Check-in Feedback", 
                                f"Here are the correct answers for questions you missed:\n\n{feedback_text}\n\n"
                                "If you're unsure about any concepts, it might be helpful to review the environment instructions or ask the PI any questions.")
        else:
            messagebox.showinfo("Check-in Completed", "Your responses have been submitted successfully!")

        print("Check-in Responses:")
        for statement, answer in responses.items():
            print(f"{statement}: {answer}")

        root.quit()
        root.destroy()

    # Display questions with choices
    for i, q in enumerate(questions):
        # Center-aligned question text
        question_label = tk.Label(root, text=q["question"], font=("Helvetica", 18, 'bold'), bg="azure", fg="black", 
                                  wraplength=700, justify="center")
        question_label.pack(pady=10, padx=10)

        # Create a frame for radio buttons (centered)
        frame = tk.Frame(root, bg="azure")
        frame.pack(pady=5)

        # Center-align radio buttons
        for choice in q["choices"]:
            choice_button = tk.Radiobutton(
                frame, 
                text=choice, 
                value=choice,  # Store full text of the answer
                font=("Helvetica", 16),
                variable=selected_answers[i],
                bg="azure",
                fg="black",
                selectcolor="#c0c0c0",
                anchor="w",  # Align text left inside button
                justify="left"
            )
            choice_button.pack(anchor="center", pady=2)  # Centering the radio button group

    # Submit button (centered)
    submit_button = tk.Button(root, text="Submit", font=("Helvetica", 16), command=submit_quiz)
    submit_button.pack(pady=20)

    # Start the event loop
    root.mainloop()

    return responses




def run_condition_experience_survey(root, condition):
    root.title(f"Experience Survey: Condition {condition}")
    root.config(bg="azure")

    # Create frames for different survey pages
    rating_frame = tk.Frame(root, bg="azure")
    open_ended_frame = tk.Frame(root, bg="azure")

    # List of rating scale statements
    statements = [
        "How confident are you in your assessment of the reward functions?",
        "How helpful did you find the information in distinguishing between the reward functions?",
        "How easy was it to incorporate the information into your decision-making?",
        "How mentally demanding was choosing between the reward functions?", 
        "How physically demanding was choosing between the reward functions?",
        "How hurried or rushed was the pace of the study?",
        "How successful were you in selecting between the reward functions?",
        "How hard did you have to work to choose between the reward functions?",
        "How insecure, discouraged, irritated, stressed, and annoyed were you?", 
        "Did you trust the information that was provided to you?",
        "Did you trust the domain expert's preferences?"
    ]

    responses = {}  # Store user responses
    options = list(range(1, 8))  # Rating scale (1 to 7)
    selected_values = {i: tk.IntVar(value=-1) for i in range(len(statements))}

    # Open-ended questions
    open_questions = [
        "How did you decide between reward functions? What information from this condition did you use in your decision-making, if any?",
        "How was your experience selecting reward functions in this condition compared to the previous condition(s)? If this was the first condition, leave blank.",
        "What challenges did you face while selecting the reward functions?",
        "Was there anything about your experience that you think this survey does not capture, or anything you would like to share?"
    ]
    feedback_entries = []

    # Function to go to the next page (Open-ended questions)
    def go_to_open_ended():
        unanswered = [statements[i] for i in range(len(statements)) if selected_values[i].get() == -1]
        if unanswered:
            confirm = messagebox.askyesno(
                "Unanswered Questions", 
                "You have unanswered questions. Are you sure you want to continue?"
            )
            if not confirm:
                return
        rating_frame.pack_forget()
        open_ended_frame.pack(pady=20)

    # Function to submit responses
    def submit_responses():
        for i, statement in enumerate(statements):
            responses[statement] = selected_values[i].get() if selected_values[i].get() != -1 else "Skipped"
        
        # Get feedback from open-ended questions
        for idx, entry in enumerate(feedback_entries):
            responses[f"Q{idx+1}: {open_questions[idx]}"] = entry.get("1.0", tk.END).strip()

        print("Survey Responses:")
        for statement, rating in responses.items():
            print(f"{statement}: {rating}")

        root.quit()
        root.destroy()

    # === Page 1: Rating Questions ===
    for i, statement in enumerate(statements):
        statement_label = tk.Label(
            rating_frame, 
            text=statement, 
            font=("Helvetica", 18, 'bold'), 
            bg="azure", 
            fg="black", 
            bd=0, 
            relief="flat"
        )
        statement_label.pack(pady=10, padx=10)

        frame = tk.Frame(rating_frame, bg="azure")
        frame.pack(pady=5)

        tk.Label(frame, text="Very Low", font=("Helvetica", 14, 'bold'), bg="azure", fg="black").pack(side=tk.LEFT, padx=10)

        for option in options:
            tk.Radiobutton(
                frame, text=str(option), value=option, font=("Helvetica", 16),
                variable=selected_values[i], bg="azure", fg="black",
                selectcolor="#c0c0c0", bd=0, relief="flat"
            ).pack(side=tk.LEFT, padx=5)

        tk.Label(frame, text="Very High", font=("Helvetica", 14, 'bold'), bg="azure", fg="black").pack(side=tk.LEFT, padx=10)

    # "Next" button to proceed to open-ended questions
    next_button = tk.Button(rating_frame, text="Next", font=("Helvetica", 16), command=go_to_open_ended)
    next_button.pack(pady=20)

    # === Page 2: Open-ended Questions ===
    for question in open_questions:
        feedback_label = tk.Label(
            open_ended_frame, text=question,
            font=("Helvetica", 16), bg="azure", fg="black",
            wraplength=600, justify="left"
        )
        feedback_label.pack(pady=10, padx=10)

        feedback_entry = tk.Text(open_ended_frame, height=4, width=70, font=("Helvetica", 14), wrap="word")
        feedback_entry.pack(pady=5, padx=10)
        feedback_entries.append(feedback_entry)

    # Submit button for final submission
    submit_button = tk.Button(open_ended_frame, text="Submit", font=("Helvetica", 16), command=submit_responses)
    submit_button.pack(pady=20)

    # Display first page initially
    rating_frame.pack(pady=20)

    root.mainloop()
    return responses







def run_preferences_survey(root):
    # Create the main window
    root.title("Survey: Providing Preferences")
    root.config(bg="azure")  # Set background color

    # List of statements with modified phrasing
    statements = [
        "How mentally demanding was providing preferences?",
        "How physically demanding was providing preferences?",
        "How hurried or rushed was the pace of providing preferences?",
        "How successful were you in accomplishing what you were asked to do?",
        "How hard did you have to work to accomplish your level of performance?",
        "How insecure, discouraged, irritated, stressed, and annoyed were you?"
    ]

    # Dictionary to store responses
    responses = {}

    # Rating scale options (1 to 7)
    options = list(range(1, 8))  # 7-point scale

    # Create a variable to store the selected values for each question
    selected_values = {i: tk.IntVar(value=-1) for i in range(len(statements))}

    # Function to handle submission
    def submit_responses():
        unanswered = [statements[i] for i in range(len(statements)) if selected_values[i].get() == -1]

        if unanswered:
            confirm = messagebox.askyesno(
                "Unanswered Questions", 
                "You have unanswered questions. Are you sure you want to submit?"
            )
            if not confirm:
                return

        for i, statement in enumerate(statements):
            responses[statement] = selected_values[i].get() if selected_values[i].get() != -1 else "Skipped"
        
        # Get feedback from the text box
        feedback_text = feedback_entry.get("1.0", tk.END).strip()
        responses["Additional Feedback"] = feedback_text

        print("Survey Responses:")
        for statement, rating in responses.items():
            print(f"{statement}: {rating}")

        root.quit()
        root.destroy()

    # Display statements with corresponding radio buttons
    for i, statement in enumerate(statements):
        statement_label = tk.Label(
            root, 
            text=statement, 
            font=("Helvetica", 18, 'bold'), 
            bg="azure", 
            fg="black", 
            bd=0, 
            relief="flat"
        )
        statement_label.pack(pady=10, padx=10)

        # Create a frame for radio buttons
        frame = tk.Frame(root, bg="azure")
        frame.pack(pady=5)

        # Add "Very Low" label
        very_low_label = tk.Label(frame, text="Very Low", font=("Helvetica", 14, 'bold'), bg="azure", fg="black")
        very_low_label.pack(side=tk.LEFT, padx=10)

        # Create radio buttons horizontally
        for option in options:
            rating_button = tk.Radiobutton(
                frame, 
                text=str(option), 
                value=option, 
                font=("Helvetica", 16),
                variable=selected_values[i],
                bg="azure", 
                fg="black", 
                selectcolor="#c0c0c0", 
                bd=0, 
                relief="flat"
            )
            rating_button.pack(side=tk.LEFT, padx=5)

        # Add "Very High" label
        very_high_label = tk.Label(frame, text="Very High", font=("Helvetica", 14, 'bold'), bg="azure", fg="black")
        very_high_label.pack(side=tk.LEFT, padx=10)

    # Add a label for the feedback text box
    feedback_label = tk.Label(
        root, 
        text="Do you have any comments about your experience while providing preferences? (e.g. how did you decide between trajectories? was something unclear?)",
        font=("Helvetica", 16), 
        bg="azure", 
        fg="black",
        wraplength=600,
        justify="left"
    )
    feedback_label.pack(pady=15, padx=10)

    # Create a text box for additional feedback
    feedback_entry = tk.Text(root, height=5, width=70, font=("Helvetica", 14), wrap="word")
    feedback_entry.pack(pady=10, padx=10)

    # Submit button
    submit_button = tk.Button(root, text="Submit", font=("Helvetica", 16), command=submit_responses)
    submit_button.pack(pady=20)

    # Start the main event loop
    root.mainloop()
    return responses


def run_final_comparisons_survey(root):
    root.title("Survey: Comparing Experience Across Conditions")
    root.config(bg="azure")

    # Create frames for multi-page survey
    rating_frame = tk.Frame(root, bg="azure")
    open_ended_frame = tk.Frame(root, bg="azure")

    # Statements for the rating questions
    statements = [
        "Which condition best helped you understand the reward functions?",
        "Which condition provided the most useful information?",
        "Which condition made the decision-making process easiest?",
        "Which condition felt the least mentally demanding?"
    ]

    responses = {}  # Store user responses
    options = ["Visual + Alignment Feedback", "Alignment Feedback", "Reward Feedback"]
    selected_values = {i: tk.StringVar(value="") for i in range(len(statements))}

    # Open-ended questions
    open_questions = [
        "Which condition did you like the best? Why?",
        "Which condition did you like the least? Why?",
        "Were there any conditions that felt overwhelming or too complicated? Why?", 
        "If you had to redo this study over again, is there anything you would do differently?"
    ]
    feedback_entries = []

    # Function to go to open-ended questions
    def go_to_open_ended():
        unanswered = [statements[i] for i in range(len(statements)) if selected_values[i].get() == ""]
        if unanswered:
            confirm = messagebox.askyesno(
                "Unanswered Questions", 
                "You have unanswered questions. Are you sure you want to continue?"
            )
            if not confirm:
                return
        rating_frame.pack_forget()
        open_ended_frame.pack(pady=20)

    # Function to submit survey responses
    def submit_responses():
        for i, statement in enumerate(statements):
            responses[statement] = selected_values[i].get() if selected_values[i].get() != "" else "Skipped"

        # Get feedback from open-ended text boxes
        for idx, entry in enumerate(feedback_entries):
            responses[f"Q{idx+1}: {open_questions[idx]}"] = entry.get("1.0", tk.END).strip()

        print("Survey Responses:")
        for statement, choice in responses.items():
            print(f"{statement}: {choice}")

        root.quit()
        root.destroy()

    # === Page 1: Rating Questions ===
    for i, statement in enumerate(statements):
        statement_label = tk.Label(
            rating_frame, text=statement,
            font=("Helvetica", 18, 'bold'), bg="azure", fg="black",
            bd=0, relief="flat"
        )
        statement_label.pack(pady=10, padx=10)

        frame = tk.Frame(rating_frame, bg="azure")
        frame.pack(pady=10)

        for option in options:
            tk.Radiobutton(
                frame, text=option, value=option, font=("Helvetica", 16),
                variable=selected_values[i], bg="azure", fg="black",
                selectcolor="#c0c0c0", bd=0, relief="flat"
            ).pack(side=tk.LEFT, padx=10, pady=0)

    # "Next" button to move to open-ended page
    next_button = tk.Button(rating_frame, text="Next", font=("Helvetica", 16), command=go_to_open_ended)
    next_button.pack(pady=20)

    # === Page 2: Open-ended Questions ===
    for question in open_questions:
        feedback_label = tk.Label(
            open_ended_frame, text=question,
            font=("Helvetica", 16), bg="azure", fg="black",
            wraplength=600, justify="left"
        )
        feedback_label.pack(pady=10, padx=10)

        feedback_entry = tk.Text(open_ended_frame, height=4, width=70, font=("Helvetica", 14), wrap="word")
        feedback_entry.pack(pady=5, padx=10)
        feedback_entries.append(feedback_entry)

    # Submit button
    submit_button = tk.Button(open_ended_frame, text="Submit", font=("Helvetica", 16), command=submit_responses)
    submit_button.pack(pady=20)

    # Show first page initially
    rating_frame.pack(pady=20)

    root.mainloop()
    return responses


def visualize_rankings_or_replay_no_alignment(self, all_reward_function_names, all_reward_function_rankings):
    """
    Allows the user to choose between visualizing rankings or rendering a trajectory.

    :param env: The environment object with a playback method.
    :param user_rankings: Dictionary of the user's rankings.
    :param ref_rankings: Dictionary of the reference rankings.
    :param trajectories: List of trajectory names.
    :param metadata: List of episode metadata corresponding to each trajectory.
    """


    
    trajectory_labels = [chr(97 + i) for i in range(len(self.trajectories))]  # 'a', 'b', 'c', ...
    all_mistmatched = {}
    input_choices = []
    trajectory_choices = []
    all_clicked_trajectories = []
    while True:
        # Ask the user for their choice
        print("\nTo help you decide which reward to select, you can ")
        print("1: Visualize how your rankings compare with the reward function rankings.")
        print("2: Replay Trajectories")
        print("stop: stop")

        choice = input("Enter your choice (1, 2, or stop): ").strip().lower()

        # Store the user's choice in the 'choices' list
        input_choices.append(choice)

        if choice == 'stop':  # Quit option
            print("Exiting.")
            print('\n')
            break

        elif choice == '1':  # Visualize ranking comparison
            # print(self.user_rankings)
            # input('wait')
            print("Opening ranking comparison visualization...")
            for reward_function_name, reward_func_rankings in list(zip(all_reward_function_names, all_reward_function_rankings)):
                #print(f'For reward function = {reward_function_name}')
                mismatched, clicked_trajectories = self.draw_rankings(self.user_rankings, reward_func_rankings, reward_function_name)
                all_clicked_trajectories.append(clicked_trajectories)


                
                reward_function_name_str = ''
                for key in reward_function_name.keys():
                    reward_function_name_str +=f'{key} = {reward_function_name[key]}\n'
                #print(reward_function_name_str, 'reward_function_name_str')
                all_mistmatched.update({reward_function_name_str: mismatched})
                
        elif choice == '2':  # Replay a trajectory
            print(colored(f"Available Trajectories:", "black", attrs=["bold"]))
            for i in range(len(self.trajectories)):
                print(f"Trajectory {chr(97 + i).upper()}")

            print('\n')
            if 'mismatched' in locals() and mismatched:
                
                for key in all_mistmatched.keys():
                    # Retrieve the list of mismatched trajectories for the current reward function
                    mismatched_trajectories = all_mistmatched[key]
                    print(colored(f"{len(mismatched_trajectories)} trajectories are mismatched under reward function \n{key}", "red", attrs=["bold"]))
                    for i in range(len(mismatched_trajectories)):
                        print(f"Trajectory {chr(97 + i).upper()}")

            try:

                trajectory_choice = input("Enter the number of the trajectory to replay:")

                # Store the user's choice in the 'choices' list
                trajectory_choices.append(trajectory_choice)

                trajectory_idx = trajectory_labels.index(trajectory_choice)  # Get the index of the chosen trajectory


                print(f"Replaying Trajectory {trajectory_choice.upper()}...")  # Display the selected trajectory label

                # Overlay trajectory ID during rendering (if supported)
                if hasattr(self.env, 'overlay_text'):
                    self.env.overlay_text(f"Rendering Trajectory {trajectory_choice.upper()}")

                # Playback the trajectory (ensure to pass the trajectory and metadata correctly)
                self.env.playback(list(self.trajectories[trajectory_idx][:]), self.metadata[trajectory_idx], trajectory_choice)  # Pass a copy of the trajectory

            except ValueError:
                print("Invalid input. Please enter a letter corresponding to the trajectory.")

        else:
            print("Invalid input. Please enter 1, 2, 3, or stop.")
    return input_choices, trajectory_choices, all_clicked_trajectories


def visualization_reward_func_condition(self, reward_function_names_selected,ranks, iteration):
    #current_function_name = inspect.currentframe().f_code.co_name

    print('\n')
    print(colored(f"Reward Function Comparsion # {iteration+1} out of {self.num_iterations} for Condition A", "black", attrs=["bold"]))
    #print('=========================================================================================')
    if iteration==self.num_iterations-1:
        print(colored(f"This is the last reward function comparsion under condition A.", "black", attrs=["bold"]))
        print('=========================================================================================')
    if iteration==0:
        #print(colored(f"Condition A", "black", attrs=["bold"]))
        print('In this condition, you will be provided with the following information:\n')
        print('1. Reward Functions\n')
        print('2. Visualizations that highlight how your rankings differ from the rankings induced by the reward functions.\n')
        print('3. The ability to rewatch any trajectories you like.\n')
        input('Press any key to continue.')    
        print('\n')

    print(colored(f"The following reward functions are:", "black", attrs=["bold"]))
    for idx, reward_function in enumerate(reward_function_names_selected):
        print(f'{idx+1}:')
        for key in reward_function.keys():
            print(f'{key} = {reward_function[key]}\n')
    
    #input('Press any key to continue.')    
    

    input_choices, trajectory_choices, clicked_trajectories = self.visualize_rankings_or_replay_no_alignment(reward_function_names_selected, ranks)

    while True:
        print(colored(f"Which reward function do you think is better?", "black", attrs=["bold"]))
        for idx, reward_function in enumerate(reward_function_names_selected):
            print(colored(f'Select {idx+1} for reward function:', "black", attrs=["bold"]))
            for key in reward_function.keys():
                print(f'{key} = {reward_function[key]}')
        print(colored(f'Select 3 if you think they are the same.', "black", attrs=["bold"]))
        print(colored(f'Select 4 if you cannot decide.', "black", attrs=["bold"]))
        choice= input('')

        if choice in ['1', '2', '3', '4', 'stop']:
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, 4, or stop.")

    if choice in ['1', '2']:
        self.data_to_save[self.current_function_name]['reward_function_choices'].append((choice, reward_function_names_selected[int(choice)-1]))
    else:
        self.data_to_save[self.current_function_name]['reward_function_choices'].append((choice, None))

    self.data_to_save[self.current_function_name]['input_choices'].append(input_choices)
    self.data_to_save[self.current_function_name]['trajectory_choices'].append(trajectory_choices)
    self.data_to_save[self.current_function_name]['clicked_trajectories'].append(clicked_trajectories)
    print('=========================================================================================')
    """Old functions I am no longer using"""



    # def replay_and_rank_trajectories(self):
    #     """
    #     Allows the user to replay trajectories and rank them, while saving user information and time spent.

    #     :param env: The environment object with a playback method.
    #     :param trajectories: A list of trajectory data.
    #     :param metadata: A list of episode metadata corresponding to each trajectory.
    #     """ 
        
    #     # Initialize rankings and start time
    #     user_rankings = { f"Traj. {chr(97 + i).upper()}": len(self.trajectories) for i in range(len(self.trajectories))}  # Default rankings to 5 for all trajectories
    #     start_time = time.time()  # Track the start time

    #     while True:
    #         # Generate labels for trajectories
    #         trajectory_labels = [chr(97 + i) for i in range(len(self.trajectories))]  # 'a', 'b', 'c', ...
    #         print('\n')
    #         print(colored(f"Available Trajectories:", "black", attrs=["bold"]))
    #         for i in range(len(self.trajectories)):
    #             print(f"Trajectory {chr(97 + i).upper()}")
    #         print('\n')
    #         choice = input("Enter the letter of the trajectory to replay, 'rank' to rank trajectories, or 'stop' to stop: ").strip().lower()

    #         if choice == 'stop':  # Check if the user wants to quit
    #             print("Finished Task 1.")
    #             print('\n')
    #             break

    #         if choice == 'rank':  # Open ranking interface
    #             print("Opening ranking interface...")
    #             root = tk.Tk()
    #             user_rankings = user_study_utils.full_order_rank_trajectories_interface(self.trajectories, user_rankings, root)
    #             root.quit()
                
    #             print("Updated Rankings:")
    #             for traj, rank in user_rankings.items():
    #                 print(f"{traj}: Rank {rank}")
    #             continue

    #         if choice not in trajectory_labels:  # Handle invalid input for trajectory labels
    #             print("Invalid input. Please enter a valid trajectory letter, 'rank', or 'stop'.")
    #             continue

    #         trajectory_idx = trajectory_labels.index(choice)  # Get the index of the chosen trajectory


    #         print(f"Replaying Trajectory {choice.upper()}...")  # Display the selected trajectory label

    #         # Overlay trajectory ID during rendering (if supported)
    #         if hasattr(self.env, 'overlay_text'):
    #             self.env.overlay_text(f"Rendering Trajectory {choice.upper()}")

            
    #         # Playback the trajectory

    #         self.env.playback(list(self.trajectories[trajectory_idx][:]), self.metadata[trajectory_idx], choice)  # Pass a copy of the trajectory

    #     # Calculate the total time taken
    #     #total_time = time.time() - start_time




    #     self.user_rankings = user_rankings

    #     # print(colored(f"Final Rankings:", "black", attrs=["bold"]))
    #     # for traj, rank in user_rankings.items():
    #     #     print(f"{traj}: Rank {rank}")

    #     print(colored(f"\nFinal Ranking", "black", attrs=["bold"]))
    #     for trajectory in self.user_rankings.keys():
    #         print(f"Rank {self.user_rankings[trajectory]}: {trajectory}")



    # def draw_rankings(self, rankings_1, rankings_2, reward_function_name):
        
    #     reward_function_print_name = ''
    #     for key in reward_function_name.keys():
    #         reward_function_print_name+=f'{key}: {reward_function_name[key]}\n'

    #     # Create the GUI window
    #     root = tk.Tk()
    #     root.title("Ranking Connections")

    #     canvas = tk.Canvas(root, width=1400, height=1400, bg="white")
    #     canvas.pack()

    #     num_ranks = len(list(rankings_1.keys()))
    #     # Canvas dimensions and parameters
    #     canvas_width = 1400
    #     canvas_height = 800
    #     x1_start = 300
    #     x2_start = 700
    #     x1_start = canvas_width // 4
    #     x2_start = (canvas_width * 3) // 4

    #     y_spacing = canvas_height // (num_ranks + 1)
    #     # Define an offset to move everything downward
    #     y_offset = 140
    #     # Add headers for the columns
    #     canvas.create_text(x1_start, 80, text="Your rankings", font=("Arial", 20, "bold underline"), anchor="s")
    #     #canvas.create_text(x2_start, 150, text=f"Reward rankings:\n{reward_function_print_name}", font=("Arial", 20, "bold"), anchor="s")
    #     canvas.create_text(
    #         x2_start, 80, 
    #         text="Reward rankings:", 
    #         font=("Arial", 20, "bold underline"), 
    #         anchor="s"
    #     )

    #     # Draw the reward function names without underline
    #     canvas.create_text(
    #         x2_start, 100,  # Adjust y-coordinate for proper spacing
    #         text=reward_function_print_name, 
    #         font=("Arial", 18, "bold"), 
    #         anchor="n"  # Anchor set to "n" so text starts below the previous one
    #     )
    #     # Map ranks to y-coordinates
    #     # rank_to_y = {rank: (i + 1) * y_spacing for i, rank in enumerate(range(1, num_ranks + 1))}
    #     rank_to_y = {rank: (i + 1) * y_spacing + y_offset for i, rank in enumerate(range(1, num_ranks + 1))}

    #     # Store trajectory IDs for each rank
    #     left_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}
    #     right_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}

    #     for trajectory, rank in rankings_1.items():
    #         left_rank_to_trajectories[rank].append(trajectory)
        
    #     for trajectory, rank in rankings_2.items():
    #         right_rank_to_trajectories[rank].append(trajectory)

    #     # Draw circles and trajectory IDs for each rank on both sides
    #     for rank, y in rank_to_y.items():
    #         # Left column
    #         canvas.create_oval(x1_start - 10, y - 10, x1_start + 10, y + 10, fill="lightgray", outline="black")
    #         canvas.create_text(x1_start - 30, y, text=f"{rank}", font=("Arial", 18), anchor="e")
    #         traj_text = ", ".join(left_rank_to_trajectories[rank]) 
           
    #         canvas.create_text(x1_start - 50, y, text=traj_text, font=("Arial", 18), anchor="e")

    #         # Right column
    #         canvas.create_oval(x2_start - 10, y - 10, x2_start + 10, y + 10, fill="lightgray", outline="black")
    #         canvas.create_text(x2_start + 30, y, text=f"{rank}", font=("Arial", 18), anchor="w")
    #         traj_text = ", ".join(right_rank_to_trajectories[rank])

    #         canvas.create_text(x2_start + 50, y, text=traj_text, font=("Arial", 18), anchor="w")

    #     # Draw connecting lines for trajectories
    #     mismatched_trajectories = []
    #     for trajectory, rank1 in rankings_1.items():
    #         rank2 = rankings_2.get(trajectory, None)
    #         if rank2 is not None:
    #             y1 = rank_to_y[rank1]
    #             y2 = rank_to_y[rank2]

    #             # Draw connecting line
    #             line_color = "green" if rank1 == rank2 else "red"
    #             canvas.create_line(x1_start, y1, x2_start, y2, fill=line_color, width=2)

    #             if rank1 != rank2:
    #                 mismatched_trajectories.append(trajectory)

    #     root.mainloop()
    #     return mismatched_trajectories



    # def draw_rankings_parallel(self, rankings_1, rankings_2, reward_function_name, canvas, x_offset):
    #     #This is for the draw rankings if we want to display both visuzliations at the same time
    #     num_ranks = len(list(rankings_1.keys()))
    #     # Canvas dimensions and parameters
    #     canvas_width = 800
    #     canvas_height = 600
    #     x1_start = x_offset + 200
    #     x2_start = x_offset + 600
    #     y_spacing = canvas_height // (num_ranks + 1)

    #     # Add headers for the columns
    #     canvas.create_text(x1_start, 20, text="Your rankings", font=("Arial", 20, "bold"), anchor="s")
    #     canvas.create_text(x2_start, 20, text=f"Reward Function {reward_function_name} Rankings", font=("Arial", 20, "bold"), anchor="s")

    #     # Map ranks to y-coordinates
    #     rank_to_y = {rank: (i + 1) * y_spacing for i, rank in enumerate(range(1, num_ranks + 1))}

    #     # Store trajectory IDs for each rank
    #     left_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}
    #     right_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}

    #     for trajectory, rank in rankings_1.items():
    #         left_rank_to_trajectories[rank].append(trajectory)

    #     for trajectory, rank in rankings_2.items():
    #         right_rank_to_trajectories[rank].append(trajectory)

    #     # Draw circles and trajectory IDs for each rank on both sides
    #     for rank, y in rank_to_y.items():
    #         # Left column
    #         canvas.create_oval(x1_start - 10, y - 10, x1_start + 10, y + 10, fill="lightgray", outline="black")
    #         canvas.create_text(x1_start - 30, y, text=f"{rank}", font=("Arial", 18), anchor="e")
    #         traj_text = ", ".join(left_rank_to_trajectories[rank]) or "(not assigned)"
    #         canvas.create_text(x1_start - 70, y, text=traj_text, font=("Arial", 18), anchor="e")

    #         # Right column
    #         canvas.create_oval(x2_start - 10, y - 10, x2_start + 10, y + 10, fill="lightgray", outline="black")
    #         canvas.create_text(x2_start + 30, y, text=f"{rank}", font=("Arial", 18), anchor="w")
    #         traj_text = ", ".join(right_rank_to_trajectories[rank]) or "(not assigned)"
    #         canvas.create_text(x2_start + 70, y, text=traj_text, font=("Arial", 18), anchor="w")

    #     # Draw connecting lines for trajectories
    #     for trajectory, rank1 in rankings_1.items():
    #         rank2 = rankings_2.get(trajectory, None)
    #         if rank2 is not None:
    #             y1 = rank_to_y[rank1]
    #             y2 = rank_to_y[rank2]

    #             # Draw connecting line
    #             line_color = "green" if rank1 == rank2 else "red"
    #             canvas.create_line(x1_start, y1, x2_start, y2, fill=line_color, width=2)

    # def draw_multiple_rankings(self, rankings_1, rankings_2, rankings_3, reward_function_name_2, reward_function_name_3):
    #     # Create the GUI window
    #     root = tk.Tk()
    #     root.title("Ranking Connections")

    #     canvas = tk.Canvas(root, width=2000, height=600, bg="white")
    #     canvas.pack()

    #     # Draw the first set of rankings
    #     self.draw_rankings_parallel(rankings_1, rankings_2, reward_function_name_2, canvas, x_offset=0)

    #     # Draw the second set of rankings
    #     self.draw_rankings_parallel(rankings_1, rankings_3, reward_function_name_3, canvas, x_offset=1000)

    #     root.mainloop()


    """This code creates a full ranking via merge sort."""
    # def generate_pairwise_comparisons(self):
    #     """
    #     Generate all pairwise comparisons for a set of trajectories.
        
    #     Returns:
    #     - comparisons (list): A list of tuples representing pairwise comparisons.
    #     """
    #     return list(combinations(range(len(self.trajectories)), 2))

    # def replay_and_pair_wise_rank_trajectories(self):
    #     """
    #     Allows the user to replay trajectories and rank them using pairwise comparisons.
    #     The user is asked to compare pairs of trajectories and select which one is better.
    #     """
        
    #     # Initialize rankings (None means no ranking yet)
    #     user_rankings = {}
        
    #     # Generate pairwise comparisons
    #     pairwise_comparisons = self.generate_pairwise_comparisons()
        
    #     start_time = time.time()  # Track the start time
        
    #     # Iterate over all pairwise comparisons
    #     for i, (idx1, idx2) in enumerate(pairwise_comparisons):
    #         print(f"\nComparison {i+1}/{len(pairwise_comparisons)}:")

            
    #         # Replay both trajectories in the pair
    #         self.replay_trajectory(idx1, idx2)

    #         # Ask the user which trajectory is better
    #         choice = None
    #         while choice not in ['1', '2', 'equal']:
    #             if self.real_user == True:
    #                 choice = input(f"Which trajectory is better? Enter '1' for Trajectory {chr(97 + idx1).upper()}, "
    #                            f"'2' for Trajectory {chr(97 + idx2).upper()}, or 'equal' if they are the same: ").strip().lower()
    #             else:
    #                 choice = '2'
    #             # Record the ranking based on the user's input
    #             if choice == '1':
    #                 user_rankings[(idx1, idx2)] = idx1
    #                 user_rankings[(idx2, idx1)] = idx1  # Mirror the ranking for reverse comparison
    #             elif choice == '2':
    #                 user_rankings[(idx1, idx2)] = idx2
    #                 user_rankings[(idx2, idx1)] = idx2  # Mirror the ranking for reverse comparison
    #             elif choice == 'equal':
    #                 user_rankings[(idx1, idx2)] = None
    #                 user_rankings[(idx2, idx1)] = None  # Mirror the "equal" status for reverse comparison
    #             else:
    #                 print("Invalid input, please enter '1', '2', or 'equal'.")
            
    #     # Calculate the full ranking based on pairwise comparisons
    #     self.user_rankings = self.compute_full_ranking(user_rankings)
     

    #     # Output the final ranking
    #     print(colored(f"\nFinal Ranking", "black", attrs=["bold"]))
    #     for trajectory in self.user_rankings.keys():
    #         print(f"Rank {self.user_rankings[trajectory]}: {trajectory}")
    #     total_time = time.time() - start_time
    #     print(f"\nTotal Time Taken: {total_time:.2f} seconds")
    

    # def compute_full_ranking(self, user_rankings):
    #     """
    #     Calculate a full ranking based on pairwise comparisons using merge sort.
    #     Handles ties correctly by assigning the same rank to equal items and skipping ranks.

    #     Returns:
    #     - rank_dict (OrderedDict): An ordered dictionary mapping each trajectory label (e.g., 'Trajectory A') to its rank.
    #     """
    #     def merge_sort_ranking():
    #         """Merge sort to calculate the full ranking based on pairwise comparisons."""
    #         def merge(left, right):
    #             merged = []
    #             while left and right:
    #                 A, B = left[0], right[0]
    #                 if user_rankings.get((A, B)) == A:
    #                     merged.append(A)
    #                     left.pop(0)
    #                 elif user_rankings.get((A, B)) == B:
    #                     merged.append(B)
    #                     right.pop(0)
    #                 else:
    #                     merged.append(A)
    #                     left.pop(0)
    #             merged.extend(left or right)
    #             return merged

    #         def merge_sort(lst):
    #             if len(lst) <= 1:
    #                 return lst
    #             mid = len(lst) // 2
    #             left = merge_sort(lst[:mid])
    #             right = merge_sort(lst[mid:])
    #             return merge(left, right)

    #         # Trajectories are indexed 0 to N-1
    #         return merge_sort(list(range(len(self.trajectories))))

    #     # Calculate the sorted trajectory indices
    #     sorted_trajectories = merge_sort_ranking()

    #     # Rank assignment with ties handled correctly
    #     rank_dict = OrderedDict()
    #     current_rank = 1
    #     for i, trajectory in enumerate(sorted_trajectories):
    #         # Generate the label for the trajectory (e.g., 'Trajectory A', 'Trajectory B')
    #         trajectory_label = f'Trajectory {chr(97 + trajectory).upper()}'
            
    #         # If it's the first item or the current item is not tied with the previous one
    #         if i == 0 or user_rankings.get((sorted_trajectories[i - 1], trajectory)) != None:
    #             rank_dict[trajectory_label] = current_rank
    #         else:
    #             # Handle tie by assigning the same rank as the previous item
    #             rank_dict[trajectory_label] = rank_dict[f'Trajectory {chr(97 + sorted_trajectories[i - 1]).upper()}']
            
    #         # If not tied, increment the rank
    #         if i == len(sorted_trajectories) - 1 or user_rankings.get((trajectory, sorted_trajectories[i + 1])) != None:
    #             current_rank += 1

    #     # Sort the dictionary by keys to maintain alphabetical order
    #     rank_dict = OrderedDict(sorted(rank_dict.items()))
    #     return rank_dict
