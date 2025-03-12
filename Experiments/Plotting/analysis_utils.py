import pickle
from plotting_utils import *
import pandas as pd
import os
import sys
from collections import OrderedDict
import scipy.stats as stats
import os
import pandas as pd
import seaborn as sns
from collections import OrderedDict

def load_pickle_files(parent_dir, names):
    """Generate file paths for user preference data."""
    dir_path = f'{parent_dir}/RewardDesignUserStudy2025/pilot_version_2_21_anonymized'
    return [f'{dir_path}/user_{name}/preference_data.pkl' for name in names]

def compare_policy_performance(parent_dir, user_ids):
    """Extracts condition data, runs statistical tests, and prints results."""
    print(f'Comparing Policy Performance')

    condition_mapping = OrderedDict([
        ('reward_func_only_condition', 'Reward Only\n(Control)'),
        ('alignment_reward_func_condition', 'Reward\n + Alignment'),
        ('alignment_visualization_reward_func_condition', 'Reward + Alignment\n + Visual')])


    dir_path = f'{parent_dir}/RewardDesignUserStudy2025/pilot_version_2_21_anonymized'
    policy_performance = get_pickle_data(f'{dir_path}/percent_correct_reward_selection_policy_performance.pkl')
    policy_performance = convert_dict_into_proper_data_framed(policy_performance)
    statistical_testing(policy_performance, conditions=policy_performance['condition'].unique(), log_transformation=False, alternative='greater')
    plot_heatmap(policy_performance, x_labels = condition_mapping, user_ids=user_ids)
    print(f'Done Comparing Policy Performance')


def compare_times(pickle_files):
    """Comparing times"""
    print(f'Comparing Time')
    condition_mapping = OrderedDict([
        ('reward_func_only_condition', 'Reward Only\n(Control)'),
        ('alignment_reward_func_condition', 'Reward\n + Alignment'),
        ('alignment_visualization_reward_func_condition', 'Reward + Alignment\n + Visual')])
    # Extract the condition data
    df = extract_condition_data(pickle_files, dict_key='time', nested_key='condition_time')
    # Run the statistical tests
    statistical_testing(df, conditions=list(condition_mapping.keys()), alpha=0.05, log_transformation=False, alternative='less')
    colors = get_colors(list(condition_mapping.values()))    
    fig = plot_survey_responses(df, y_label='Reward Selection Time (Seconds)',colors=colors,sub_keys=['condition_time'],
    condition_mapping=condition_mapping, sub_key_mapping=OrderedDict([('condition_time', '')]), rotate_labels=0, yticks_upper_bound=None, max_y_limit=None, figsize=(12, 8))
    print(f'Done Comparing Time')

def analyze_nasa_survey(pickle_files, workload_keys, workload_keys_plot,  sub_key_mapping=None, condition_mapping=None):
    """Analyzes survey responses and workload scores."""
    print('Analyzing survey responses and workload scores.')
    # Step 1: Extract data from pickle files
    df_survey = extract_condition_data(pickle_files, dict_key="survey_responses")


    #  #Compute workload scores
    df_workload = compute_general_workload(df_survey, workload_keys)

    # plot specific questions and total workload 
    df_combined = pd.concat([df_survey, df_workload], ignore_index=True)
    colors = get_colors(list(condition_mapping.values())) 
    fig = plot_survey_responses(df_combined, colors=colors,sub_keys=workload_keys_plot, condition_mapping=condition_mapping, sub_key_mapping=sub_key_mapping, rotate_labels=0)

    # # Step 2: Run statistical tests
    conditions = list(condition_mapping.keys())

    statistical_testing(df_workload, conditions, log_transformation=False, alternative='less')
    print('Done analyzing survey responses and workload scores.')



def analyze_categorical_responses(pickle_files, categorical_questions, sub_key_mapping, plot=False):
    """Analyzes categorical survey responses."""
    
    condition_mapping = OrderedDict([
     ('Reward Feedback', 'Reward Only (Control)'),
     ('Visual + Alignment Feedback', 'Reward + Alignment + Visual'),
     ('Alignment Feedback', 'Reward + Alignment')])
    df_survey = extract_condition_data(pickle_files, 'comparison_survey_responses', is_global=True)
    df_counts = compute_categorical_response_counts(df_survey, categorical_questions, new_col_name="count", debug=False)
    df_counts_merged = compute_categorical_response_counts_merged_alignment(df_survey, categorical_questions)

    # Run the analysis
    #Define your conditions
    alignment_conditions = ["Either Alignment Conditions"]
    reward_condition = "Reward Feedback"

    statistical_testing_for_counts(
        df_counts_merged,
        alignment_conditions=alignment_conditions,
        reward_condition=reward_condition,
        debug=True  # set to False to hide detailed output
    )
    if plot:
        alignment_conditions = ["Visual + Alignment Feedback", "Alignment Feedback"]
        colors = get_colors(list(condition_mapping.values())) 

        fig = plot_survey_responses_stacked_barplot(df_counts, colors=colors,
                            is_categorical=True, 
                            y_label='Votes', 
                            sub_key_mapping=sub_key_mapping,
                            condition_mapping=condition_mapping, 
                            title='', 
                            stack_alignment=True)
