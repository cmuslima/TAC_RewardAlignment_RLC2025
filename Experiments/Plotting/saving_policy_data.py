import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict
import pickle

combined_dict = {
    "alignment_reward_func_condition": {
        "6": 0.75,
        "10": 1,
        "2": 1,
        "5": 1,
        "1": 0.75,
        "4": 1,
        "9": 1,
        "7": 1, 
        "3": 0.75, 
        "8": 1.0,
        '11': 1
    },
    "reward_func_only_condition": {
        "6": 0,
        "10": 0.50,
        "2": 1,
        "5": 1,
        "1": 1,
        "4": 0.50,
        "9": 0.50,
        "7": 0.50,
        "3": 1.0, 
        "8": 0.75, 
        '11': 0.5
    },
    "alignment_visualization_reward_func_condition": {
        "6": 0.50,
        "10": 1.0,
        "2": 1,
        "5": 1,
        "1": 1,
        "4": 1,
        "9": 1,
        "7": 1,
        "3": 1.0, 
        "8": 1.0, 
        '11': 1
    }
}
combined_dict = OrderedDict(combined_dict)
file_name = f'/Users/cmuslimani/Projects/RewardDesign/Reward_Alignment/Experiments/RewardDesignUserStudy2025/lastest_pilot_version_2_21_anonymized/percent_correct_reward_selection_policy_performance'
with open(f"{file_name}.pkl", 'wb') as file:
    pickle.dump(combined_dict, file)
input('wait')


