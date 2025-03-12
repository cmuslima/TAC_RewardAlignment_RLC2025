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
import copy

import user_study_utils
import get_alignment
sys.path.append('../Domains/gym-hungry-thirsty/gym_hungry_thirsty/envs')
sys.path.append('../Domains/gym-hungry-thirsty')
import hungry_thirsty_user_control

import hungry_thirsty_env 
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

class interface_backend:
    def __init__(self, random_seed=28, use_human_stakeholder_preferences=True):
        
        self.study_start_time = time.time()
        self.use_human_stakeholder_preferences = use_human_stakeholder_preferences
        if self.use_human_stakeholder_preferences:
            # self.user_rankings = OrderedDict({'Traj. A': 10, 'Traj. B': 12, 'Traj. C': 9, 'Traj. D': 11, \
            # 'Traj. E': 8, 'Traj. F': 7, 'Traj. G': 5, 'Traj. H': 6, 'Traj. I': 4, 'Traj. J': 1, 'Traj. K': 2, \
            # 'Traj. L': 3})

            self.user_rankings = OrderedDict({'Traj. A': 10, 'Traj. B': 12, 'Traj. C': 9, 'Traj. D': 11, 'Traj. E': 5, \
                'Traj. F': 7, 'Traj. G': 6, 'Traj. H': 8, 'Traj. I': 1, 'Traj. J': 4, 'Traj. K': 2, 'Traj. L': 3})
            


        """These are the other three changes the researcher should be making during the actual user study. 
        Especially the random_seed"""
        self.real_user = True
        self.random_seed = random_seed
        self.DEBUG = False
        self.answer_key_mode = False
        self.set_seed()


        if self.real_user:
            assert self.answer_key_mode == False

        self.env_seed = 0
        self.num_conditions = 3

        if self.DEBUG:
            self.num_iterations = 14 #will be down to 4 per group. tomorrow if i get the other result, maybe we can add that
        else:
            self.num_iterations = 4
        self.reward_func_index = 0
        self.complete_counterbalance = True

        

        
        

        reward_funcs_considered=[(48, 53), (49, 43), (48, 21), (46, 47), (34, 37), (32, 29), (12, 21), (13, 14), (7, 29), (28, 29), (24, 29), (22, 21)] 
        
        self.ground_truth, self.selected_reward_functions = user_study_utils.get_specific_reward_functions(reward_funcs_considered, random_seed=random_seed, debug=self.DEBUG)   
        
        self.set_order_of_conditions()

        self.data_to_save = OrderedDict()
        if self.use_human_stakeholder_preferences:
            self.data_to_save.update({'name': None, 'user_id': None, 'random_seed': self.random_seed, 'env_quiz_responses': None, 'user_rankings': None,\
                'preference_time': None,  'total_study_time': None, 'comparison_survey_time': None, 'comparison_survey_responses': None, \
                'close_window_early_rewatching': [], 'close_window_early': [], 'trajectory_choices': []})
        else:
            additional_data_dict = {'practice_rankings': None, 'preference_survey': None}
            self.data_to_save.update(additional_data_dict)
        

        for condition in self.conditions:
            user_inputs = {
            'input_choices': [],
            'trajectory_choices': [], 
            'survey_responses': None, 
            'reward_function_choices': [],
            'close_window_early': [], #this is for the gifs 
            'time': []}
            self.data_to_save.update({condition.__name__: user_inputs})

        self.current_function_name = None


 
        
        self.get_trajectories()
        self.env, self.metadata = user_study_utils.get_env_and_meta_data(self.env_seed, self.num_traj)
        self.condition_name_map = {'alignment_visualization_reward_func_condition': 'Visual + Alignment Feedback',\
                                   'reward_func_only_condition': 'Reward Feedback', 'alignment_reward_func_condition': 'Alignment Feedback'}
    def set_seed(self):
        assert self.random_seed in range(0, 6), f"Error: self.random_seed must be an integer between 0 and 5, but got {self.random_seed}."

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def set_order_of_conditions(self):
    
        if self.complete_counterbalance:
            possible_orders = [
                [self.alignment_visualization_reward_func_condition, self.reward_func_only_condition, self.alignment_reward_func_condition],
                [self.alignment_visualization_reward_func_condition, self.alignment_reward_func_condition, self.reward_func_only_condition],
                [self.reward_func_only_condition, self.alignment_visualization_reward_func_condition, self.alignment_reward_func_condition],
                [self.reward_func_only_condition, self.alignment_reward_func_condition, self.alignment_visualization_reward_func_condition],
                [self.alignment_reward_func_condition, self.alignment_visualization_reward_func_condition, self.reward_func_only_condition],
                [self.alignment_reward_func_condition, self.reward_func_only_condition, self.alignment_visualization_reward_func_condition]
            ]
            self.conditions = possible_orders[self.random_seed]

        else:
            self.conditions = [self.alignment_visualization_reward_func_condition, self.reward_func_only_condition, self.alignment_reward_func_condition]
            np.random.shuffle(self.conditions)

        if self.DEBUG:
            self.conditions = [self.alignment_visualization_reward_func_condition]
    def set_study_id(self):
        # Get user information
        if self.real_user:
            self.name = input("Please type your name and press then Enter: ").strip()
        else:
            self.name = 'temp'
        self.user_id = str(uuid.uuid4())  # Generate a random unique ID
        self.data_to_save['name'] = self.name
        self.data_to_save['user_id']= self.user_id

        print(f"Welcome, {self.name}! Your user ID is: {self.user_id}\n")
        self.data_dir = user_study_utils.get_data_dir(self.name)


    def replay_and_pair_wise_rank_trajectories(self, practice_mode=True):
        """
        Allows the user to replay trajectories and rank them using pairwise comparisons.
        In practice mode, the user can practice interactions before the real study starts.
        """
        def run_comparison(idx1, idx2, practice_mode=False):
            """ Helper function to run one comparison """
            print(colored(f'Preference Comparison {self.num_preferences_complete}:', "black", attrs=["bold"]))
            print(f"Comparing Trajectory {chr(97 + idx1).upper()} and Trajectory {chr(97 + idx2).upper()}...\n")

            if self.real_user:
                self.replay_pairs_of_trajectories(idx1, idx2)
            else:
                fakechoice = self.replay_pairs_of_trajectories(idx1, idx2)

            choice = ""
    

            while True:
                valid_choices = {chr(97 + idx1).upper(), chr(97 + idx2).upper(), 'equal'}

                if self.real_user:
                    text = colored(
                        f"Which trajectory is better? Enter '{chr(97 + idx1).upper()}' for Trajectory {chr(97 + idx1).upper()}, "
                        f"'{chr(97 + idx2).upper()}' for Trajectory {chr(97 + idx2).upper()}, or 'equal' if they are the same:", 
                        "black", attrs=["bold"]
                    )
                    print(text)
                    choice = input().strip()  # Keep input as is (case-sensitive)
                else:
                    choice = fakechoice
                    print(f'Selected {choice}')

                # Normalize choice (convert single chars to uppercase for comparison)
                if choice.lower() == "equal":
                    break  # "equal" is valid in any case
                elif len(choice) == 1 and choice.upper() in valid_choices:
                    break  # Valid single-character choice
                else:
                    print(colored("Invalid choice. Please enter a valid option.", "red", attrs=["bold"]))


            if practice_mode == False:
                self.num_preferences_complete += 1
            return choice

        # ---------------- PRACTICE PHASE ----------------
        practice_rankings = {}  # Store practice responses separately

        if practice_mode:
            print(colored("\nPractice Mode: Try making a few comparisons before the real task starts.", "cyan", attrs=["bold"]))

            practice_trajectories = list(range(min(4, len(self.trajectories))))  # Use first few trajectories
            for practice_round in range(2):  # Let them practice 2 times
                print(f'Practice round {practice_round}:\n')
                idx1, idx2 = random.sample(practice_trajectories, 2)  # Pick random pairs
                choice = run_comparison(idx1, idx2, practice_mode=True)
                practice_rankings[(idx1, idx2)] = choice
                practice_rankings[(idx2, idx1)] = choice
                print(colored(f"Your choice: {choice.upper()}. This was just practice.", "yellow"))

            print(colored("\nPractice complete! Now starting the real task...\n", "green", attrs=["bold"]))
        
            # Save practice rankings separately
            self.data_to_save['practice_rankings'] = practice_rankings
            user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)

        # ---------------- REAL TASK STARTS ----------------
        user_rankings = {}
        sorted_trajectories = []
        start_time = time.time()

        for trajectory in range(len(self.trajectories)):
            if not sorted_trajectories:
                sorted_trajectories.append(trajectory)
                continue

            low, high = 0, len(sorted_trajectories)
            while low < high:
                mid = (low + high) // 2
                idx1, idx2 = trajectory, sorted_trajectories[mid]
                choice = run_comparison(idx1, idx2)

                if choice == chr(97 + idx1).lower():
                    user_rankings[(idx1, idx2)] = idx1
                    user_rankings[(idx2, idx1)] = idx1
                    high = mid
                elif choice == chr(97 + idx2).lower():
                    user_rankings[(idx1, idx2)] = idx2
                    user_rankings[(idx2, idx1)] = idx2
                    low = mid + 1
                elif choice == 'equal':
                    user_rankings[(idx1, idx2)] = None
                    user_rankings[(idx2, idx1)] = None
                    low = mid + 1

            sorted_trajectories.insert(low, trajectory)

        
        # Convert sorted trajectories to rankings
        rank_dict = user_study_utils.convert_to_rankings(sorted_trajectories, user_rankings)
        print(colored("\nFinal Ranking", "black", attrs=["bold"]))
        for trajectory in rank_dict.keys():
            print(f"Rank {rank_dict[trajectory]}: {trajectory}")

        total_time = time.time() - start_time
        print(f"\nTotal Time Taken: {total_time:.2f} seconds")

        rank_dict = OrderedDict(sorted(rank_dict.items()))  # Sort alphabetically
        self.user_rankings = rank_dict
        self.data_to_save['user_rankings'] = self.user_rankings

        user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)


    def replay_pairs_of_trajectories(self, idx1, idx2):
        """
        Replay the two trajectories for comparison, allowing the user to replay them
        as many times as they want, going back and forth between them before deciding to stop.
        """

        # Overlay trajectory ID during rendering (if supported)
        if hasattr(self.env, 'overlay_text'):
            self.env.overlay_text(f"Rendering Trajectory {chr(97 + idx1).upper()} and {chr(97 + idx2).upper()}")


        if self.real_user:
            # Playback both trajectories initially
            input(f'Press Enter to Watch Trajectory {chr(97 + idx1).upper()}')
            close_early_1, score1 = self.env.playback(list(self.trajectories[idx1][:]), self.metadata[idx1], chr(97 + idx1).upper())
            
            input(f'Press Enter to Watch Trajectory {chr(97 + idx2).upper()}')
            close_early_2, score2 = self.env.playback(list(self.trajectories[idx2][:]), self.metadata[idx2], chr(97 + idx2).upper())
            
            
            self.data_to_save['close_window_early'].append((chr(97 + idx1), close_early_1))
            self.data_to_save['close_window_early'].append((chr(97 + idx2), close_early_2))
            # Combined replay loop
            while True:
                # Ask the user which trajectory they want to replay
                choice = input(f"Which trajectory do you want to replay? Enter '{chr(97 + idx1).upper()}' for Trajectory {chr(97 + idx1).upper()}, "
                            f"'{chr(97 + idx2).upper()}' for Trajectory {chr(97 + idx2).upper()}, or 'stop' to finish: ").strip().lower()

        
                if choice.strip().lower() == chr(97 + idx1):
                    close_early_1, _ = self.env.playback(list(self.trajectories[idx1][:]), self.metadata[idx1], chr(97 + idx1).upper())
                    self.data_to_save['close_window_early_rewatching'].append((chr(97 + idx1), close_early_1))
                elif choice.strip().lower() == chr(97 + idx2):
                    close_early_2, _ = self.env.playback(list(self.trajectories[idx2][:]), self.metadata[idx2], chr(97 + idx2).upper())
                    self.data_to_save['close_window_early_rewatching'].append((chr(97 + idx2), close_early_2))
                    
                elif choice == 'stop':
                    print("Stopping the replay.\n")
                    break  # Exit the replay loop
                else:
                    print(f"Invalid choice. Please enter '{chr(97 + idx1).upper()}' to replay Trajectory {chr(97 + idx1).upper()}, '{chr(97 + idx2).upper()}' for Trajectory {chr(97 + idx2).upper()}, or 'stop' to finish.")

        else:
            close_early_1, score1 = self.env.playback(list(self.trajectories[idx1][:]), self.metadata[idx1], chr(97 + idx1).upper(), render=False)
            
            close_early_2, score2 = self.env.playback(list(self.trajectories[idx2][:]), self.metadata[idx2], chr(97 + idx2).upper(), render=False)


            if score1 > score2:
                return chr(97 + idx1)
            elif score1 < score2:
                return chr(97 + idx2)
            else:
                return 'equal'
                

    def draw_rankings(self, rankings_1, rankings_2, reward_function_name):
        def initialize_canvas():
            root = tk.Tk()
            root.title("Ranking Connections")

            canvas = Canvas(root, width=1800, height=1400, bg="white")
            canvas.pack()
            return root, canvas

        def format_reward_function_name(reward_function_name):
            reward_function_print_name = ''
            for key, value in reward_function_name.items():
                reward_function_print_name += f'{key}: {value}\n'
            return reward_function_print_name

        def load_gif(trajectory_id):
            if trajectory_id in self.gif_cache:
                return self.gif_cache[trajectory_id]

            gif_path = f"TrajsGifs/trajectory{trajectory_id.upper()}.gif"
            try:
                gif = Image.open(gif_path)
            except IOError:
                print(f"Error: Unable to open the gif file at {gif_path}")
                return None

            target_width = 450
            aspect_ratio = gif.height / gif.width
            target_height = int(target_width * aspect_ratio)

            frames = [
                ImageTk.PhotoImage(
                    gif.copy().resize((target_width, target_height), Image.ANTIALIAS)
                )
                for frame in range(gif.n_frames)
                if gif.seek(frame) is None
            ]

            self.gif_cache[trajectory_id] = (frames, gif.info.get('duration', 100))
            return self.gif_cache[trajectory_id]

        def show_trajectory(event, trajectory_id):
            clicked_trajectories.append(trajectory_id)

            if trajectory_id in self.open_windows:
                self.open_windows[trajectory_id].focus()
                return

            gif_data = load_gif(trajectory_id)
            if not gif_data:
                return

            frames, frame_duration = gif_data

            new_window = Toplevel(root)
            new_window.title(f"Trajectory Playback: {trajectory_id}")

            label = Label(new_window)
            label.pack()
            self.open_windows[trajectory_id] = new_window

            def animate(frame=0):
                if frame < len(frames):
                    label.configure(image=frames[frame])
                    label.image = frames[frame]
                    new_window.after(frame_duration, animate, frame + 1)
                else:
                    new_window.destroy()
                    del self.open_windows[trajectory_id]

            animate()

        def rank_difference(ranking1, ranking2):
            largest_drop, largest_increase = float('inf'), -float('inf')
            drop_items, increase_items = [], []
            mismatched_trajectories = []

            for item in ranking1:
                rank1, rank2 = ranking1.get(item), ranking2.get(item)

                if rank1 is None or rank2 is None:
                    continue

                if rank1 != rank2:
                    mismatched_trajectories.append(item)

                rank_diff = rank2 - rank1

                if rank_diff < largest_drop:
                    largest_drop, drop_items = rank_diff, [item]
                elif rank_diff == largest_drop:
                    drop_items.append(item)

                if rank_diff > largest_increase:
                    largest_increase, increase_items = rank_diff, [item]
                elif rank_diff == largest_increase:
                    increase_items.append(item)

            return drop_items, largest_drop, increase_items, largest_increase, mismatched_trajectories

        def draw_columns_and_circles():
            # Add titles for the left and right columns
            canvas.create_text(
                x1_start, y_offset - 50,  # Position above the left column
                text="Domain Expert's Rankings",
                font=("Arial", 20, "bold underline"),
                anchor="center"
            )
            canvas.create_text(
                x2_start, y_offset - 50,  # Position above the right column
                text=f"Reward Function Rankings\n({reward_function_print_name.strip()})",
                font=("Arial", 20, "bold underline"),
                anchor="center"
            )

            rank_to_y = {rank: (i + 1) * y_spacing + y_offset for i, rank in enumerate(range(1, num_ranks + 1))}
            left_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}
            right_rank_to_trajectories = {rank: [] for rank in range(1, num_ranks + 1)}

            for trajectory, rank in rankings_1.items():
                left_rank_to_trajectories[rank].append(trajectory)
            for trajectory, rank in rankings_2.items():
                right_rank_to_trajectories[rank].append(trajectory)

            for rank, y in rank_to_y.items():
                # Draw the rank circles
                canvas.create_oval(x1_start - 20, y - 20, x1_start + 20, y + 20, fill="blue", outline="black")
                canvas.create_text(x1_start - 40, y, text=f"{rank}", font=("Arial", 18), anchor="e")
                canvas.create_oval(x2_start - 20, y - 20, x2_start + 20, y + 20, fill="blue", outline="black")
                canvas.create_text(x2_start + 40, y, text=f"{rank}", font=("Arial", 18), anchor="w")
                
                
                # Handle User Rankings (side-by-side trajectory IDs)
                if left_rank_to_trajectories[rank]:
                    combined_text = "  |  ".join(left_rank_to_trajectories[rank])
                    canvas.create_text(
                        x1_start - 70, y,  # Align with the rank position
                        text=combined_text,
                        font=("Arial", 18),
                        anchor="e",
                        fill="blue"
                    )


                # Handle Reward Function Rankings (side-by-side trajectory IDs)
                if right_rank_to_trajectories[rank]:
                    combined_text = "  |  ".join(right_rank_to_trajectories[rank])
                    canvas.create_text(
                        x2_start + 70, y,  # Align with the rank position
                        text=combined_text,
                        font=("Arial", 18),
                        anchor="w",
                        fill="blue"
                    )

        def draw_lines_and_annotations():
            drop_items, largest_drop, increase_items, largest_increase, mismatched_trajectories = rank_difference(rankings_1, rankings_2)
            rank_to_y = {rank: (i + 1) * y_spacing + y_offset for i, rank in enumerate(range(1, num_ranks + 1))}

            for trajectory, rank1 in rankings_1.items():
                rank2 = rankings_2.get(trajectory)
                if rank2 is not None:
                    y1, y2 = rank_to_y[rank1], rank_to_y[rank2]
                    line_color = "green" if rank1 == rank2 else "red"
                    

                    # Ensure bolding happens only if the rank has changed
                    if rank1 != rank2 and trajectory in drop_items + increase_items:
                        line_width = 10
                    else:
                        line_width = 2
                    # if largest_drop == 0:
                    #     line_width = 2
                    # if largest_increase == 0:
                    #     line_width = 2
                    canvas.create_line(x1_start, y1, x2_start, y2, fill=line_color, width=line_width)

            return drop_items, increase_items, mismatched_trajectories

        def display_mismatch_details(drop_items, increase_items, mismatched_trajectories):
            
            # Only proceed if there are mismatched trajectories
            if mismatched_trajectories:
                details = [
                    "Bolded lines are trajectories with the\ngreatest difference in rank.\n\n"
                    #f"{len(mismatched_trajectories)} trajectories are mismatched.", removing this part for now
                    #"This includes:\n\n\n",
                ]

                # Show all the trajectories with the greatest increase
                for trajectory in increase_items:
                    rank1 = rankings_1.get(trajectory, None)
                    rank2 = rankings_2.get(trajectory, None)
                    if rank1 is not None and rank2 is not None:
                        increase_diff = rank2 - rank1
                        # if abs(increase_diff) > 0:
                        #     details.append(f"\n-{trajectory}: The domain expert ranked it {increase_diff}\nslot(s) higher.\n\n")

                # Show all the trajectories with the greatest drop
                for trajectory in drop_items:
                    rank1 = rankings_1.get(trajectory, None)
                    rank2 = rankings_2.get(trajectory, None)
                    if rank1 is not None and rank2 is not None:
                        drop_diff = rank2 - rank1
                        # if abs(drop_diff) > 0:
                        #     details.append(f"\n-{trajectory}: The domain expert ranked it {abs(drop_diff)}\nslot(s) lower.\n\n")

                # Print or display the details
                for i, detail in enumerate(details):
                    canvas.create_text(
                        right_text_start_x, right_text_start_y + i * 30,
                        text=detail, font=("Arial", 20, "bold"), fill="red", anchor="w"
                    )

        # Main Logic
        root, canvas = initialize_canvas()
        reward_function_print_name = format_reward_function_name(reward_function_name)

        num_ranks = len(rankings_1)
        canvas_width, canvas_height = 1800, 800
        x1_start, x2_start = canvas_width // 6, (canvas_width * 3) // 6
        y_spacing, y_offset = canvas_height // (num_ranks + 1), 140
        clicked_trajectories, callbacks = [], {}

        right_text_start_x = x2_start + 200  # Positioning the text to the right of the right column
        right_text_start_y = y_offset +100 # Aligning with the y-offset

        draw_columns_and_circles()
        drop_items, increase_items, mismatched_trajectories = draw_lines_and_annotations()
        display_mismatch_details(drop_items, increase_items, mismatched_trajectories)

        root.mainloop()
        return mismatched_trajectories, clicked_trajectories
    def replay_trajectories_no_alignment(self, all_reward_function_names=None, all_reward_function_alignment_scores=None, use_alignment=False):
        """
        Allows the user to choose between visualizing rankings or rendering a trajectory.

        :param env: The environment object with a playback method.
        :param user_rankings: Dictionary of the user's rankings.
        :param ref_rankings: Dictionary of the reference rankings.
        :param trajectories: List of trajectory names.
        :param metadata: List of episode metadata corresponding to each trajectory.
        """

        input_choices = []
        trajectory_choices = []
        all_clicked_trajectories = []
        while True:
            # Ask the user for their choice
            print("\nTo help you decide which reward to select, you can also")
            print("1: Replay Trajectories")
            if use_alignment:
                print("2: Reload the Alignment Scores")
            print("stop: stop")
            if use_alignment:
                choice = input("Enter your choice (1, 2 or stop): ").strip().lower()
            else:
                choice = input("Enter your choice (1 or stop): ").strip().lower()

            # Store the user's choice in the 'choices' list
            input_choices.append(choice)

            if choice == 'stop':  # Quit option
                print("Exiting.")
                print('\n')
                break

            elif choice == '1':  # Replay a trajectory
                print(colored(f"Available Trajectories:", "black", attrs=["bold"]))
                for i in range(len(self.trajectories)):
                    print(f"Trajectory {chr(97 + i).upper()}")

                print('\n')
                trajectory_choices = self.select_trajectory_for_replay(trajectory_choices)
            
            elif choice == '2' and use_alignment:
                print('\n')
                for idx, reward_function in enumerate(all_reward_function_names):
                    print(colored(f"Reward Function {idx+1}", "black", attrs=["bold"]))
                    for key in reward_function.keys():
                        print(f'{key} = {reward_function[key]}')
                    print(colored(f"With alignment score = {np.round(all_reward_function_alignment_scores[idx], 4)}\n", "yellow", attrs=["bold"]))

            else:
                if use_alignment:
                    print("Invalid input. Please enter 1, 2 or stop.")
                else:
                    print("Invalid input. Please enter 1 or stop.")
        return input_choices, trajectory_choices, all_clicked_trajectories
    
    def visualize_rankings_or_replay_no_alignment(self, all_reward_function_names, all_reward_function_rankings):
        """
        Allows the user to choose between visualizing rankings or rendering a trajectory.

        :param env: The environment object with a playback method.
        :param user_rankings: Dictionary of the user's rankings.
        :param ref_rankings: Dictionary of the reference rankings.
        :param trajectories: List of trajectory names.
        :param metadata: List of episode metadata corresponding to each trajectory.
        """

        all_mistmatched = {}
        input_choices = []
        trajectory_choices = []
        all_clicked_trajectories = []
        while True:
            # Ask the user for their choice
            print("\nTo help you decide which reward to select, you can")
            print("1: Visualize how the domain expert's preferences over trajectories differ from the preferences induced by the reward functions.")
            print("2: Replay Trajectories")
            print("stop: stop")

            choice = input("Enter your choice (1, 2 or stop): ").strip().lower()

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
                trajectory_choices = self.select_trajectory_for_replay(trajectory_choices)

            else:
                print("Invalid input. Please enter 1, 2 or stop.")
        return input_choices, trajectory_choices, all_clicked_trajectories


    def play_trajectory(self, trajectory_choice):
        """
        Plays a given trajectory.
        """
        try:
            # Get the index of the chosen trajectory
            trajectory_idx = self.trajectory_labels.index(trajectory_choice)

            print(f"Playing Trajectory {trajectory_choice.upper()}...")

            # Overlay trajectory ID during rendering (if supported)
            if hasattr(self.env, 'overlay_text'):
                self.env.overlay_text(f"Rendering Trajectory {trajectory_choice.upper()}")

            # Playback the trajectory
            close_early, _ = self.env.playback(
                list(self.trajectories[trajectory_idx][:]),
                self.metadata[trajectory_idx],
                trajectory_choice.upper()
            )

            # Save whether the trajectory was exited early
            if self.current_function_name is not None:
                self.data_to_save[self.current_function_name]['close_window_early'].append((trajectory_choice, close_early))
            else:
                self.data_to_save['close_window_early'].append((trajectory_choice, close_early))

        except ValueError:
            print("Invalid input. Please enter a valid trajectory letter.")

    def select_trajectory_for_replay(self, trajectory_choices):
        """
        Allows the user to select and replay a trajectory.
        """
        try:
            # Get user input for trajectory selection
            trajectory_choice = input("Enter the letter of the trajectory to replay: ").lower()

            if trajectory_choice not in self.trajectory_labels:
                print("Invalid input. Please enter a letter corresponding to the trajectory.")
                return trajectory_choices  # Keep the list unchanged and prompt again

            trajectory_choices.append(trajectory_choice)
            self.play_trajectory(trajectory_choice)

        except ValueError:
            print("Invalid input. Please enter a valid trajectory letter.")

        return trajectory_choices  # Return updated choices list


    def visualize_rankings_or_replay(self, all_reward_function_names, all_reward_function_alignment_scores, all_reward_function_rankings):
        """
        Allows the user to choose between visualizing rankings or rendering a trajectory.

        :param env: The environment object with a playback method.
        :param user_rankings: Dictionary of the user's rankings.
        :param ref_rankings: Dictionary of the reference rankings.
        :param trajectories: List of trajectory names.
        :param metadata: List of episode metadata corresponding to each trajectory.
        """


        all_mistmatched = {}
        input_choices = []
        trajectory_choices = []
        all_clicked_trajectories = []
        while True:
            # Ask the user for their choice
            print("\nTo help you decide which reward to select, you can")
            print("1: Reload the Alignment Scores")
            print("2: Visualize how the domain expert's preferences over trajectories differ from the preferences induced by the reward functions.")
            print("3: Replay Trajectories")
            print("stop: stop")

            choice = input("Enter your choice (1, 2, 3, or stop): ").strip().lower()

            # Store the user's choice in the 'choices' list
            input_choices.append(choice)

            if choice == 'stop':  # Quit option
                print("Exiting.")
                print('\n')
                break

            elif choice == '2':  # Visualize ranking comparison
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
            elif choice == '1': #Show alignment scores
                print('\n')
                for idx, reward_function in enumerate(all_reward_function_names):
                    print(colored(f"Reward Function {idx+1}:", "black", attrs=["bold"]))
                    for key in reward_function.keys():
                        print(f'{key} = {reward_function[key]}')
                    print(colored(f"With alignment score = {np.round(all_reward_function_alignment_scores[idx], 4)}\n", "yellow", attrs=["bold"]))
                
            elif choice == '3':  # Replay a trajectory
                print(colored(f"Available Trajectories:", "black", attrs=["bold"]))
                for i in range(len(self.trajectories)):
                    print(f"Trajectory {chr(97 + i).upper()}")

                print('\n')
                trajectory_choices = self.select_trajectory_for_replay(trajectory_choices)

            else:
                print("Invalid input. Please enter 1, 2, 3, or stop.")
        return input_choices, trajectory_choices, all_clicked_trajectories
            

    def print_condition_intro(self, condition_name, iteration, num_iterations):
        print('\n')
        print(colored(f"Reward Function Comparison # {iteration+1} out of {self.num_iterations} for Condition:\n{condition_name}", "black", attrs=["bold"]))
        if iteration == num_iterations - 1:
            print(colored(f"This is the last reward function comparison under Condition: {condition_name}.", "black", attrs=["bold"]))
            print('=' * 90)
        if iteration == 0:
            print('In this condition, you will be provided with the following information:\n')
            print('1. Reward Functions\n')
            print("2. The ability to rewatch any trajectories you like.\n")
            if condition_name in ['Visual + Alignment Feedback', 'Alignment Feedback']:
                print('3. An alignment measure, which is a score ranging in the interval [-1,1].\n')
                print("\tThis score is based on the domain expert's preferences in Task 1 and the preferences that are induced by the reward function.")
                print("\tThis score will tell you how similar the domain expert's preferences are with the reward function's preferences.\n")
                
                self.print_alignment_score_explanation()
            if condition_name == 'Visual + Alignment Feedback':
                print("4. A visualization that highlights how the domain expert's preferences differ from the reward function's preferences.\n")
  
            if condition_name == 'Visual Feedback':
                print("2. Visualizations that highlight how the domain expert's preferences over trajectories differ from the preferences induced by the reward functions.\n")
                print('3. The ability to rewatch any trajectories you like.\n')        
            input('Press Enter to continue.')
            print('\n')

    def print_alignment_score_explanation(self):
        print(f"\tAn alignment score close to {colored('1', 'green', attrs=['bold'])} indicates that the reward function induces {colored('similar preferences as the domain expert!', 'green', attrs=['bold'])}\n")
        print(f"\tAn alignment score close to {colored('-1', 'red', attrs=['bold'])} indicates that the reward function induces {colored('opposite preferences as the domain expert!', 'red', attrs=['bold'])}\n")
        print(f"\tAt a high level, this score is a ratio based on the number of matched preferences versus unmmatched preferences.")
        print(colored("\tReminder: Assume that the domain expert is the source of ground truth.", "black", attrs=["bold"]))


    def print_reward_functions(self, reward_function_names_selected):
        print(colored("You will be comparing the following reward functions:", "black", attrs=["bold"]))
        for idx, reward_function in enumerate(reward_function_names_selected):
            print(colored(f"Reward Function {idx+1}", "black", attrs=["bold"]))
            for key in reward_function.keys():
                print(f'{key} = {reward_function[key]}')
            

    def get_user_choice(self, reward_function_names_selected):
        while True:
            print(colored("Which reward function do you think is better?", "black", attrs=["bold"]))
            for idx, reward_function in enumerate(reward_function_names_selected):
                print(colored(f'Select {idx+1} for reward function:', "black", attrs=["bold"]))
                for key in reward_function.keys():
                    print(f'{key} = {reward_function[key]}')
            print(colored("Select 3 if you think they are the same.", "black", attrs=["bold"]))
            print(colored("Select 4 if you cannot decide.", "black", attrs=["bold"]))
            choice = input('')
            if choice in ['1', '2', '3', '4', 'stop']:
                return choice
            print("Invalid input. Please enter 1, 2, 3, 4, or stop.")

    def handle_reward_function_choice(self, choice, reward_function_names_selected):
        if choice in ['1', '2']:
            self.data_to_save[self.current_function_name]['reward_function_choices'].append((choice, reward_function_names_selected[int(choice)-1]))
        else:
            self.data_to_save[self.current_function_name]['reward_function_choices'].append((choice, None))

    def print_condition_summary(self):

        print(colored("\nYou will now be asked to compare your experiences under the differing conditions.\n", "black", attrs=["bold"]))
        print("\nHere is a Summary of Information Provided in Each Condition:\n")
        print(colored("Condition: Alignment Feedback", "black", attrs=["bold"]))
        print("1. Reward Functions\n")
        print("2. An alignment score that indicates how similar the domain expert's preferences are to those induced by the reward functions.\n")
        print("3. The ability to rewatch any trajectories you like.\n")
        print(colored("\nCondition: Visual + Alignment Feedback", "black", attrs=["bold"]))
        print("1. Reward Functions\n")
        print("2. An alignment score that indicates how similar the domain expert's preferences are to those induced by the reward functions.\n")
        print("3. Visualizations that highlight how the domain expert's preferences over trajectories differ from the preferences induced by the reward functions.\n")
        print("4. The ability to rewatch any trajectories you like.\n")
        print(colored("Condition: Reward Feedback", "black", attrs=["bold"]))
        print("1. Reward Functions\n")
        print("2. The ability to rewatch any trajectories you like.\n")

    def alignment_reward_func_condition(self, reward_function_names_selected, reward_function_alignment_scores_selected, iteration):
        self.print_condition_intro('Alignment Feedback', iteration, self.num_iterations)
        #self.print_reward_functions(reward_function_names_selected)
        for idx, reward_function in enumerate(reward_function_names_selected):
            print(colored(f"Reward Function {idx+1}:", "black", attrs=["bold"]))
            for key in reward_function.keys():
                print(f'{key} = {reward_function[key]}')
            print(colored(f"With alignment score = {np.round(reward_function_alignment_scores_selected[idx], 4)}\n", "yellow", attrs=["bold"]))

            
        self.replay_trajectories_no_alignment(all_reward_function_names=reward_function_names_selected, all_reward_function_alignment_scores=reward_function_alignment_scores_selected,use_alignment=True)
        choice = self.get_user_choice(reward_function_names_selected)
        self.handle_reward_function_choice(choice, reward_function_names_selected)
        print('=' * 90)

    def alignment_visualization_reward_func_condition(self, reward_function_names_selected, reward_function_alignment_scores_selected, ranks, iteration):
        self.print_condition_intro('Visual + Alignment Feedback', iteration, self.num_iterations)
        #self.print_reward_functions(reward_function_names_selected)
        for idx, reward_function in enumerate(reward_function_names_selected):
            print(colored(f"Reward Function {idx+1}", "black", attrs=["bold"]))
            for key in reward_function.keys():
                print(f'{key} = {reward_function[key]}')
            print(colored(f"With alignment score = {np.round(reward_function_alignment_scores_selected[idx], 4)}\n", "yellow", attrs=["bold"]))

            
        input_choices, trajectory_choices, clicked_trajectories = self.visualize_rankings_or_replay(
            reward_function_names_selected, reward_function_alignment_scores_selected, ranks
        )
        choice = self.get_user_choice(reward_function_names_selected)
        self.handle_reward_function_choice(choice, reward_function_names_selected)
        self.data_to_save[self.current_function_name]['input_choices'].append(input_choices)
        self.data_to_save[self.current_function_name]['trajectory_choices'].append(trajectory_choices)
        #self.data_to_save[self.current_function_name]['clicked_trajectories'].append(clicked_trajectories)
        print('=' * 90)
    def visualization_reward_func_condition(self, reward_function_names_selected, ranks, iteration):
        self.print_condition_intro('Visual Feedback', iteration, self.num_iterations)
        self.print_reward_functions(reward_function_names_selected)
        input_choices, trajectory_choices, clicked_trajectories = self.visualize_rankings_or_replay_no_alignment(
            reward_function_names_selected, ranks
        )
        choice = self.get_user_choice(reward_function_names_selected)
        self.handle_reward_function_choice(choice, reward_function_names_selected)
        self.data_to_save[self.current_function_name]['input_choices'].append(input_choices)
        self.data_to_save[self.current_function_name]['trajectory_choices'].append(trajectory_choices)
        #self.data_to_save[self.current_function_name]['clicked_trajectories'].append(clicked_trajectories)
        print('=' * 90)
    def reward_func_only_condition(self, reward_function_names_selected, iteration, reward_function_alignment_scores_selected=None):
        self.print_condition_intro('Reward Feedback', iteration, self.num_iterations)
        self.print_reward_functions(reward_function_names_selected)
        if self.answer_key_mode:
            for idx, reward_function in enumerate(reward_function_names_selected):
                print(colored(f"Reward Function {idx+1}", "black", attrs=["bold"]))
                for key in reward_function.keys():
                    print(f'{key} = {reward_function[key]}')
                print(colored(f"With alignment score = {np.round(reward_function_alignment_scores_selected[idx], 4)}\n", "yellow", attrs=["bold"]))
        self.replay_trajectories_no_alignment()
        choice = self.get_user_choice(reward_function_names_selected)
        self.handle_reward_function_choice(choice, reward_function_names_selected)
        print('=' * 90)




    def function_loop(self) -> None:
        """
        Executes the function loop for a defined number of iterations
        and applies different conditions to the reward functions.
        """
        start_time = time.time()
        # Initialize the current function and name
        self.current_func = self.conditions[0]
        self.current_function_name = self.current_func.__name__
        for iteration in range(self.num_iterations):
            #print(f"Iteration {iteration + 1}/{self.num_iterations}")

            # Retrieve the selected reward functions for this iteration
            try:
                reward_function_names_selected = self.selected_reward_functions[self.reward_func_index]
            except IndexError:
                raise IndexError("Reward function index out of range. Check `self.selected_reward_functions`.")

            

            # Retrieve rankings and alignment scores
            try:
                ranks = [self.reward_function_rankings[str(name)] for name in reward_function_names_selected]
                alignment_scores = [
                    self.all_reward_function_alignment_scores[str(name)]
                    for name in reward_function_names_selected
                ]
            except KeyError as e:
                raise KeyError(f"Reward function name {e} not found in rankings or alignment scores.")


            if self.DEBUG:
                print(f"Selected reward functions: {reward_function_names_selected}") # will comment out later
                print(f"Ranks: {ranks}")
                print(f"Alignment Scores: {alignment_scores}")

            # Dynamically prepare arguments based on the condition
            args = self._prepare_arguments(
                reward_function_names_selected, ranks, alignment_scores, iteration
            )

            # Execute the current function with prepared arguments
            self.current_func(*args)
            self.reward_func_index += 1

            # Optional pause for user input
            if iteration < self.num_iterations - 1:
                input("Press Enter to continue to the next comparison.")

        print("You have finished comparing reward functions for this condition.\n")
        user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)
        self.conditions.pop(0)

        finish_time = time.time()
        total_time = finish_time-start_time
        self.data_to_save[self.current_function_name]['time'].append({'condition_time': total_time})

        if self.DEBUG:
            for key in self.data_to_save[self.current_function_name].keys():
                print('\n key', key)
                print('\n value', self.data_to_save[self.current_function_name][key])

    def _prepare_arguments(
        self, 
        reward_function_names_selected: list, 
        ranks: list, 
        alignment_scores: list, 
        iteration: int
    ) -> tuple:
        """
        Prepares arguments dynamically based on the current condition.
        """
        if self.current_func == self.visualization_reward_func_condition:
            return reward_function_names_selected, ranks, iteration
        elif self.current_func == self.alignment_reward_func_condition:
            return reward_function_names_selected, alignment_scores, iteration
        elif self.current_func == self.alignment_visualization_reward_func_condition:
            return reward_function_names_selected, alignment_scores, ranks, iteration
        elif self.current_func == self.reward_func_only_condition:
            return reward_function_names_selected, iteration, alignment_scores
        else:
            raise ValueError(f"Unknown condition: {self.current_func.__name__}")

    def condition_survey(self):
        #Call the survey function to start the process
        start_time = time.time()
        root = tk.Tk()
        responses = user_study_utils.run_condition_experience_survey(root, condition=f'{self.condition_name_map[self.current_function_name]}')
        root.quit() 
        self.data_to_save[self.current_function_name]['survey_responses'] = responses


        # print("Survey Results:")
        # for statement, (score, label) in responses.items():
        #     print(f"{statement}: {score} ({label})")
        finish_time = time.time()
        total_time = finish_time-start_time
        self.data_to_save[self.current_function_name]['time'].append({f'survey_time': total_time})



        user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)
        if self.DEBUG:
            for key in self.data_to_save[self.current_function_name].keys():
                print('\n key', key)
                print('\n value', self.data_to_save[self.current_function_name][key])

    def env_understanding_checkin(self):
      
        
        root = tk.Tk()
        responses = user_study_utils.run_quiz(root)
        root.quit() 
        self.data_to_save['env_quiz_responses'] = responses

        user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)


    
    def final_survey(self):
        #self.print_condition_summary()
        #Call the survey function to start the process
        start_time = time.time()
        root = tk.Tk()
        responses = user_study_utils.run_final_comparisons_survey(root)
        root.quit() 
        self.data_to_save['comparison_survey_responses'] = responses


        finish_time = time.time()
        total_time = finish_time-start_time
        self.data_to_save['comparison_survey_time'] = total_time

        print('This is the end of the study')
        print(f'Thank you for your participation {self.name}!') 
        
        study_over_time = time.time()
        total_time = study_over_time - self.study_start_time 
        self.data_to_save['total_study_time'] = total_time

        user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)
        if self.DEBUG:
            print(self.data_to_save)
  
        
    

    def upload_preference_data(self):
        filename = f"{self.data_dir}/preference_data.pkl"
        #self.data_to_save['user_rankings'] =utils.get_pickle_data(filename)
        #self.user_rankings = self.data_to_save['user_rankings'] 
        #filename = f"{self.data_dir}/preference_data.pkl"
        self.data_to_save =utils.get_pickle_data(filename) 
        self.user_rankings = self.data_to_save['user_rankings']
   

    def get_trajectories(self):

        # file = f'{parent_dir}/Experiments/mixture_low_medium_high/env_seed{self.env_seed}/trajectory_selection_num_traj_{self.num_traj}_10.pkl'
        # traj_ids =utils.get_pickle_data(file)['selected_trajectories']
        # trajectories, _ = rank_reward_funcs.get_trajectories(self.ground_truth, 'Q_Learn', self.env_seed+1)
        # print(type(trajectories),np.shape(trajectories))

        # self.trajectories = trajectories[self.env_seed][traj_ids]
        # print(type(self.trajectories),np.shape(self.trajectories))


        trajectory_data_dir = f'{os.getcwd()}/Experiments/Env_Easy/SameStartState/0_0/alg_Q_learn/env_timesteps_200/envs_10/alpha_lr_0.05/eps_0.15/gamma_0.99/num_episodes_10000/record_freq_1/num_tests_1/(0, 0, 1.0, 1.0)'
        file = f'{os.getcwd()}/User_Study_Data/Trajectories/trajectory_comparsions_start_state_12_0_0_seed0_v2.pkl'
        traj_ids =utils.get_pickle_data(file)


        trajectories, _ = utils.get_trajectories(num_runs=self.env_seed+1, dir=trajectory_data_dir)
        self.trajectories = trajectories[self.env_seed][traj_ids]
        self.num_traj = len(self.trajectories)
      
        self.trajectory_labels = [chr(97 + i) for i in range(len(self.trajectories))] 
        
        flat_list = [item for sublist in self.selected_reward_functions for item in sublist]

        self.reward_function_rankings = user_study_utils.compute_trajectory_rewards_and_rankings(self.trajectories, flat_list)
        self.reward_function_rankings = self.reward_function_rankings['rankings']



            





    def studypart1(self):
        if self.use_human_stakeholder_preferences:
            start_time = time.time()
            self.show_stakeholder_preferences_and_replay()
            finish_time = time.time()
            total_time = finish_time-start_time
            self.data_to_save['preference_time'] = total_time
            user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)

        else:
            if self.use_loaded_preference_data:
                self.upload_preference_data()
                print(f'Using loaded preferences from user {self.name}')
                if self.DEBUG:
                    print(self.data_to_save)
            else:
                start_time = time.time()
                self.replay_and_pair_wise_rank_trajectories()
                finish_time = time.time()
                total_time = finish_time-start_time
                self.data_to_save['preference_time'] = total_time
                print('user_rankings', self.user_rankings)
        
            root = tk.Tk()
            responses = user_study_utils.run_preferences_survey(root)
            root.quit() 
            self.data_to_save['preference_survey'] = responses
            user_study_utils.save_intermediate_data(self.data_dir, self.data_to_save)

        self.all_reward_function_alignment_scores = {}

        for reward_function in self.reward_function_rankings.keys():

            alignment_score = get_alignment.get_trajectory_alignment_coefficent_score(
                list(self.reward_function_rankings[reward_function].values()), list(self.user_rankings.values())
            )
            self.all_reward_function_alignment_scores.update({str(reward_function):alignment_score})

    def allow_user_control(self):
        hungry_thirsty_user_control.user_control()

    def show_stakeholder_preferences_and_replay(self):
        """This is the function where users will see the ranking of a set of trajectories according to a human stakeholder. 
        Then the user will be able to watch the trajectories."""
        
      
        #print("\nYou will now watch all trajectories in ranked order (1 = Best, Higher = Worse). After that, you can replay any trajectory you like.")


        sorted_rankings = OrderedDict(sorted(copy.deepcopy(self.user_rankings).items(), key=lambda item: item[1]))

        for trajectory_label, rank in sorted_rankings.items():
            traj_id = trajectory_label.split()[-1].lower()
            print(f"\n{colored('Rank', 'black', attrs=['bold'])} ({rank}):")

            self.play_trajectory(traj_id)

        trajectory_choices = []
        # Step 2: Allow user to replay trajectories
        while True:
            user_input = input("\nDo you want to replay any trajectory? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                trajectory_choices = self.select_trajectory_for_replay(trajectory_choices)
            elif user_input in ['n', 'no']:
                print("Exiting trajectory replay.")
                break
            else:
                print("Invalid Input")

    

        self.data_to_save['trajectory_choices'] = trajectory_choices

