import pickle
from plotting_utils import *
from analysis_utils import *
import pandas as pd
import os
import sys
from collections import OrderedDict
import scipy.stats as stats


def main(dir_of_user_study_data, compare_performance=False, compare_reward_selection_times=False, compare_nasa_survey=False, compare_voting_survey=False):
    parent_dir = os.path.dirname(os.getcwd())
    dir = f'{parent_dir}/{dir_of_user_study_data}'
    sub_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    pickle_files = []
    print(sub_dirs)
    for name in sub_dirs:
        pickle_files.append(f'{dir}/{name}/preference_data.pkl')

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-adjust width
    pd.set_option('display.max_colwidth', None)  # Show full column content


    #df_survey = extract_condition_data(pickle_files, 'comparison_survey_responses', is_global=True)
    #filtered_df = df_survey[df_survey['sub_key'].str.contains(r'Q[2]', na=False)]





    all_workload_keys = {
        'How mentally demanding was choosing between the reward functions?': False,
        'How physically demanding was choosing between the reward functions?': False,
        'How hard did you have to work to choose between the reward functions?': False,
        'How hurried or rushed was the pace of the study?': False,
        'How insecure, discouraged, irritated, stressed, and annoyed were you?': False,
        'How successful were you in selecting between the reward functions?': True,  # Reverse scored
        'How easy was it to incorporate the information into your decision-making?': True,  # Reverse scored
        'How confident are you in your assessment of the reward functions?': True  # Reverse scored
    }

    workload_keys_to_plot = [
        'How mentally demanding was choosing between the reward functions?',
        'How hard did you have to work to choose between the reward functions?',
        'How insecure, discouraged, irritated, stressed, and annoyed were you?',
        'How easy was it to incorporate the information into your decision-making?',
        'How confident are you in your assessment of the reward functions?'
        ,'workload']
    sub_key_mapping_nasa = OrderedDict([
        ('workload', 'Overall\nWork Load'),
        ('How hard did you have to work to choose between the reward functions?', 'Effort'),
        ('How mentally demanding was choosing between the reward functions?', 'Mental\nDemand'),
        ('How insecure, discouraged, irritated, stressed, and annoyed were you?', 'Stress'),
        ('How easy was it to incorporate the information into your decision-making?', 'Ease\nof Use'),
        ('How confident are you in your assessment of the reward functions?', 'Confidence')
    
    ])
    sub_key_mapping_voting= {
    'Which condition best helped you understand the reward functions?': 'Improved Reward\nUnderstanding', \
    'Which condition provided the most useful information?': 'Most Useful\nFeedback', \
    'Which condition made the decision-making process easiest?': 'Easiest Decision\nMaking', \
    'Which condition felt the least mentally demanding?': "Least Mental\nDemand"
    }



    categorical_questions = {
        'Which condition best helped you understand the reward functions?',
        'Which condition provided the most useful information?',
        'Which condition made the decision-making process easiest?',
        'Which condition felt the least mentally demanding?'}

    condition_mapping = OrderedDict([
        ('reward_func_only_condition', 'Reward Only (Control)'),
        ('alignment_reward_func_condition', 'Reward + Alignment'),
        ('alignment_visualization_reward_func_condition', 'Reward + Alignment + Visual')])



    if compare_performance:
        compare_policy_performance(parent_dir, user_ids=sub_dirs)
    if compare_reward_selection_times:
        compare_times(pickle_files)
    if compare_nasa_survey:
        analyze_nasa_survey(pickle_files, all_workload_keys, workload_keys_to_plot,  sub_key_mapping_nasa, condition_mapping)
    if compare_voting_survey:
        analyze_categorical_responses(pickle_files, categorical_questions, sub_key_mapping_voting, plot=True)



dir_of_user_study_data = '/RewardDesignUserStudy2025/pilot_version_2_21_anonymized'
main(dir_of_user_study_data, compare_voting_survey=True)
