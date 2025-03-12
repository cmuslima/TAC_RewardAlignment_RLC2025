import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy.stats as stats
import numpy as np
import seaborn as sns
import copy
from matplotlib.colors import LinearSegmentedColormap,BoundaryNorm,ListedColormap
np.random.seed(42)
def get_pickle_data(file_name):
    with open(file_name,'rb') as input:
        data = pickle.load(input)
    return data
def convert_dict_into_proper_data_framed(data):
    # Convert dictionary into a DataFrame
    df_list = []
    for condition, users in data.items():
        for user, value in users.items():
            df_list.append({'user_id': user, 'value': value, 'condition': condition})

    # Create DataFrame
    df = pd.DataFrame(df_list)
    return df

def statistical_testing(df, conditions, alpha=0.05, log_transformation=False, show_visualize_distribution=False, alternative='two-sided'):
    """
    Performs paired statistical tests between conditions on a filtered dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing user responses.
                         Must have columns: ['user_id', 'condition', 'value']
        conditions (list): List of condition names to compare (must be at least 2).
        alpha (float): Significance level for normality testing and FDR correction (default: 0.05).
        log_transformation (bool): Whether to apply log transformation to values (default: False).

    Returns:
        dict: Dictionary containing test results for each comparison.
    """

    if len(conditions) < 2:
        raise ValueError("Need at least two conditions to compare.")
        
    # Input validation
    required_cols = {'user_id', 'condition', 'value'}
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
    # Convert values to numeric, warning about any conversions
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if df['value'].isna().any():
        print(f"Warning: {df['value'].isna().sum()} non-numeric values were converted to NaN")
    
    # Pivot data for paired comparisons
    df_pivot = df.pivot(index='user_id', columns='condition', values='value')
    
    # Check if we have enough paired data
    available_pairs = df_pivot.dropna(subset=conditions)
    if len(available_pairs) < 3:  # Minimum sample size for meaningful tests
        raise ValueError(f"Insufficient paired data. Only {len(available_pairs)} complete pairs found.")
    
    # Apply log transformation if requested
    if log_transformation:
        # Check for non-positive values before log transform
        if (df_pivot[conditions] <= 0).any().any():
            print("Warning: Non-positive values found. Using log1p transformation.")
            for cond in conditions:
                df_pivot[cond] = np.log1p(df_pivot[cond])
        else:
            for cond in conditions:
                df_pivot[cond] = np.log(df_pivot[cond])
    
    results = {}
    if 'sub_key' in df.columns:
        print(f'Testing for {df["sub_key"][0]}:')
    else:
        print('Testing conditions:')
    p_values_dict = {}  # Dictionary to store p-values with their comparison keys
    
    # Perform pairwise comparisons
    for i in range(len(conditions)-1):
        for j in range(i+1, len(conditions)):
            cond1, cond2 = conditions[i], conditions[j]
            if 'reward_func_only_condition' in cond1:
                tempcond1 = copy.deepcopy(cond2)
                cond2 = copy.deepcopy(cond1)
                cond1 = tempcond1
            if not all(c in df_pivot.columns for c in [cond1, cond2]):
                print(f"Warning: Skipping comparison {cond1} vs {cond2} - missing data")
                continue
                
            # Get paired data
            paired_data = df_pivot[[cond1, cond2]].dropna()
            if len(paired_data) < 3:
                print(f"Warning: Insufficient paired data for {cond1} vs {cond2}")
                continue
                
            # Compute descriptive statistics
            desc_stats = {
                cond1: {
                    'mean': paired_data[cond1].mean(),
                    'std': paired_data[cond1].std(),
                    'median': paired_data[cond1].median(),
                    'n': len(paired_data)
                },
                cond2: {
                    'mean': paired_data[cond2].mean(),
                    'std': paired_data[cond2].std(),
                    'median': paired_data[cond2].median(),
                    'n': len(paired_data)
                }
            }
            
            # Test normality
            shapiro_1 = stats.shapiro(paired_data[cond1])
            shapiro_2 = stats.shapiro(paired_data[cond2])
            normal_1, normal_2 = shapiro_1.pvalue > alpha, shapiro_2.pvalue > alpha
            
            # Store comparison results
            comparison_key = f"{cond1}_vs_{cond2}"
            results[comparison_key] = {
                'descriptive_stats': desc_stats,
                'normality_tests': {
                    cond1: {'statistic': shapiro_1.statistic, 'p_value': shapiro_1.pvalue},
                    cond2: {'statistic': shapiro_2.statistic, 'p_value': shapiro_2.pvalue}
                }
            }
            
            # Print current comparison
            print(f"\nComparing: {cond1} vs {cond2}")
            print(f"  Sample size: {len(paired_data)}")
            print(f"  {cond1}: mean={desc_stats[cond1]['mean']:.3f} ± {desc_stats[cond1]['std']:.3f}")
            print(f"  {cond2}: mean={desc_stats[cond2]['mean']:.3f} ± {desc_stats[cond2]['std']:.3f}")
            print(f"  Normality p-values: {cond1}={shapiro_1.pvalue:.3f}, {cond2}={shapiro_2.pvalue:.3f}")
            
            # Perform appropriate statistical test
            if normal_1 and normal_2:
                # Paired t-test (two-sided)
                t_stat, p_value = stats.ttest_rel(paired_data[cond1], paired_data[cond2], alternative=alternative)
                test_name = "Paired t-test"
                results[comparison_key]['test'] = {
                    'name': test_name,
                    'statistic': t_stat,
                    'p_value': p_value
                }
                print(f"  {test_name}: t={t_stat:.3f}, p={p_value:.3f}")
                p_values_dict[comparison_key] = p_value

            else:
                # Wilcoxon signed-rank test (two-sided)
                print([cond1])
                w_stat, p_value = stats.wilcoxon(paired_data[cond1], paired_data[cond2], alternative=alternative)
                test_name = "Wilcoxon signed-rank test"
                results[comparison_key]['test'] = {
                    'name': test_name,
                    'statistic': w_stat,
                    'p_value': p_value
                }
                print(f"  {test_name}: W={w_stat:.3f}, p={p_value:.3f}")
                p_values_dict[comparison_key] = p_value
            
            # Add effect size
            cohens_d = (paired_data[cond1].mean() - paired_data[cond2].mean()) / \
                      np.sqrt((paired_data[cond1].std()**2 + paired_data[cond2].std()**2) / 2)
            results[comparison_key]['effect_size'] = {
                'cohens_d': cohens_d
            }
            print(f"  Cohen's d: {cohens_d:.3f}")
            
            # Visualize if function exists in namespace
            if 'visualize_distribution' in globals() and show_visualize_distribution:
                visualize_distribution(paired_data, cond1, cond2)

    # # Perform FDR correction
    # if p_values_dict:
    #     # Key-value pair to remove because we are only interested in whether either alignment conditions was better than reward only

    #     keys_to_remove = ['alignment_visualization_reward_func_condition_vs_alignment_reward_func_condition', 'alignment_reward_func_condition_vs_alignment_visualization_reward_func_condition']
        
    #     p_values_for_correction = copy.deepcopy(p_values_dict)
    #     for key_to_remove in keys_to_remove:
    #         if key_to_remove in p_values_for_correction:
    #             del p_values_for_correction[key_to_remove]

    #     p_values_dict = {k: v for k, v in copy.deepcopy(p_values_for_correction).items() if k != key_to_remove}
    #     adjusted_p_values = stats.false_discovery_control(list(p_values_dict.values()))
        
    #     # Print FDR correction results
    #     print("\nFDR Correction Results:")
    #     for (comparison, original_p), adjusted_p in zip(p_values_dict.items(), adjusted_p_values):
    #         is_significant = adjusted_p < alpha
    #         results[comparison]['fdr_correction'] = {
    #             'adjusted_p_value': adjusted_p,
    #             'is_significant': is_significant
    #         }
    #         print(f"  {comparison}:")
    #         print(f"    Original p-value: {original_p:.3f}")
    #         print(f"    Adjusted p-value: {adjusted_p:.3f}")
    #         print(f"    Significant: {'Yes' if is_significant else 'No'}")

    return results
def statistical_testing_for_counts(df_counts, alignment_conditions, reward_condition, alpha=0.05,debug=False):
    """
    Compares each alignment condition separately against the reward condition.
    
    Args:
        df_counts (pd.DataFrame): DataFrame from compute_categorical_response_counts
        alignment_conditions (list): List of alignment conditions (as values/numbers)
        reward_condition (int/float): Value representing the "Reward Feedback" condition
        debug (bool): If True, prints detailed test information.
    """
    # Get unique sub_keys (survey questions)

    unique_questions = df_counts['sub_key'].unique()
    p_values_dict = {}  # Dictionary to store p-values with their comparison keys
    for question in unique_questions:
        print(f"\n🔹 Testing for: {question}")
        
        for alignment_condition in alignment_conditions:
            print(f"\nComparing Condition {alignment_condition} vs Condition {reward_condition}")
            
            # Filter data for the current question and conditions
            mask = (df_counts['sub_key'] == question) & \
                  (df_counts['response_value'].isin([alignment_condition, reward_condition]))
            df_filtered = df_counts[mask].copy()
            
            if debug:
                print("\n*** Filtered Data ***")
                print(df_filtered)
            
            # Create the 2x2 contingency table
            # [alignment_votes, non_alignment_votes]
            # [reward_votes, non_reward_votes]
            alignment_votes = df_filtered[df_filtered['response_value'] == alignment_condition]['count'].iloc[0]
            reward_votes = df_filtered[df_filtered['response_value'] == reward_condition]['count'].iloc[0]
            
            # Calculate total votes for this question
            total_votes = df_counts[df_counts['sub_key'] == question]['count'].sum()
            
            # Create contingency table
            contingency_array = [
                [alignment_votes, total_votes - alignment_votes],
                [reward_votes, total_votes - reward_votes]
            ]
            
            if debug:
                print("\n*** Contingency Table ***")
                print(f"                    Selected  Not Selected")
                print(f"Condition {alignment_condition}: {contingency_array[0]}")
                print(f"Condition {reward_condition}:  {contingency_array[1]}")
            
            # Fisher's Exact Test
            odds_ratio, fisher_p = stats.fisher_exact(contingency_array)
            p_values_dict[question] = fisher_p
            
            # Print results
            print(f"\n*** Fisher's Exact Test Results ***")
            print(f"Counts - Condition {alignment_condition}: {alignment_votes}, Condition {reward_condition}: {reward_votes}")
            print(f"P-value: {fisher_p:.4f}")
            print(f"   Odds ratio: {odds_ratio:.2f}")
        


    # # Perform FDR correction
    # if p_values_dict:
    #     adjusted_p_values = stats.false_discovery_control(list(p_values_dict.values()))
        
    #     # Print FDR correction results
    #     print("\nFDR Correction Results:")
    #     for (comparison, original_p), adjusted_p in zip(p_values_dict.items(), adjusted_p_values):
    #         is_significant = adjusted_p < alpha
    #         print(f"  {comparison}:")
    #         # Check significance
            
    #         if is_significant:
    #             print("✅ Significant difference found")
    #             # Add effect size calculation
    #             effect_size = odds_ratio - 1  # Simple effect size measure
    #             direction = "more" if odds_ratio > 1 else "less"
    #             print(f"   Condition {alignment_condition} was chosen {direction} frequently than Condition {reward_condition}")
    #             print(f"   Odds ratio: {odds_ratio:.2f}")

    #             """An odds ratio of infinity (inf) occurs when there is a "zero cell" in your contingency table - specifically when one of the conditions has zero occurrences in one category while having some occurrences in the other category."""
    #         else:
    #             print("❌ No significant difference found")
    #             print(f"   Odds ratio: {odds_ratio:.2f}")


    #         print(f"    Original p-value: {original_p:.3f}")
    #         print(f"    Adjusted p-value: {adjusted_p:.3f}")




def compute_general_workload(df, workload_questions, new_col_name="workload_score", 
                             is_global=False, scale=7, debug=False):
    """
    Computes a workload score by averaging selected survey responses, handling reverse scoring.
    
    Args:
        df (pd.DataFrame): DataFrame containing survey responses with columns ['user_id', 'condition', 'sub_key', 'value'].
        workload_questions (dict or set): If a dict, keys are survey question texts, and values are True (if reverse scored) or False.
                                          If a set, assumes all questions are NOT reverse-scored.
        new_col_name (str): Name for the computed workload score column.
        is_global (bool): If True, treats all responses as global (not condition-based).
        scale (int): The scale used (default: 1-7 scale). If 0-100, set to 100.
        debug (bool): If True, prints detailed individual scores before aggregation.

    Returns:
        pd.DataFrame: Standardized DataFrame with ['user_id', 'condition', 'sub_key', 'value'].
    """
    if scale not in [7, 100]:
        raise ValueError("Scale must be either 7 or 100.")
        
    # Convert set to dictionary (assume no reverse scoring)
    if isinstance(workload_questions, set):
        workload_questions = {q: False for q in workload_questions}

    df_selected = df[df['sub_key'].isin(workload_questions.keys())].copy(deep=True)
    
    # Convert value column to numeric, coercing errors to NaN
    df_selected['value'] = pd.to_numeric(df_selected['value'], errors='coerce')
    
    # Drop any NaN values
    nan_count = df_selected['value'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} non-numeric or missing values were dropped")
    df_selected = df_selected.dropna(subset=['value'])

    # Reverse scoring
    for question, reverse in workload_questions.items():
        if reverse:
            mask = df_selected['sub_key'] == question
            if scale == 7:
                df_selected.loc[mask, 'value'] = 8 - df_selected.loc[mask, 'value']
            else:  # scale == 100
                df_selected.loc[mask, 'value'] = 100 - df_selected.loc[mask, 'value']

    # Aggregate workload score
    if is_global:
        df_aggregated = df_selected.groupby(['user_id'])['value'].mean().round(2).reset_index()
        df_aggregated['condition'] = 'global'
    else:
        df_aggregated = df_selected.groupby(['user_id', 'condition'])['value'].mean().round(4).reset_index()
    
    # Rename and standardize output format
    df_aggregated.rename(columns={'value': new_col_name}, inplace=True)
    df_aggregated['sub_key'] = "workload"
    df_aggregated.rename(columns={new_col_name: "value"}, inplace=True)

    return df_aggregated


def extract_condition_data(pickle_files, dict_key, nested_key=None, is_global=False):
    """
    Extracts specified data from multiple pickle files based on a given dictionary key.
    
    Args:
        pickle_files (list): List of paths to pickle files.
        dict_key (str): The dictionary key to extract data for.
        nested_key (str, optional): The key inside a nested dictionary to extract data from.
        is_global (bool): If True, extracts data from the top-level keys instead of condition-specific keys.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['user_id', 'condition', 'sub_key', 'value'].
        
    Raises:
        FileNotFoundError: If a pickle file cannot be found
        ValueError: If required keys are missing or data structure is invalid
    """
    data = []
    condition_keys = {
        'alignment_reward_func_condition',
        'reward_func_only_condition',
        'alignment_visualization_reward_func_condition'
    }
    
    def validate_and_process_value(value):
        """Helper function to validate and process values"""

      
        if isinstance(value, (int, float, str, bool)):  # Added bool
            return value
        elif isinstance(value, np.ndarray):  # Handle numpy arrays
            return value.item() if value.size == 1 else str(value)
        elif pd.isna(value):  # Handle NaN/None values
            return None
        else:
            return str(value)  # Convert other types to string rather than dropping
    
    for file in pickle_files:

        try:
            with open(file, 'rb') as f:
                user_data = pickle.load(f)
            
                
            if not isinstance(user_data, dict):
                print(f"Warning: Skipping file {file} - user_data is not a dictionary")
                continue
                
            user_id = user_data.get('user_id')
            if not user_id:
                print(f"Warning: No user ID found in {file}, using filename")
                user_id = Path(file).stem
            
            if is_global:
                if dict_key not in user_data:
                    print(f"Warning: {dict_key} not found in global data for user {user_id}")
                    continue
                    
                extracted_data = user_data[dict_key]
                if isinstance(extracted_data, dict):
                    for sub_key, value in extracted_data.items():
                        processed_value = validate_and_process_value(value)
                        if processed_value is not None:
                            data.append({
                                'user_id': user_id,
                                'condition': 'global',
                                'sub_key': sub_key,
                                'value': processed_value
                            })
                else:
                    processed_value = validate_and_process_value(extracted_data)
                    if processed_value is not None:
                        data.append({
                            'user_id': user_id,
                            'condition': 'global',
                            'sub_key': None,
                            'value': processed_value
                        })
            else:
                for condition in condition_keys:
                    if condition not in user_data:
                        continue
                        
                    condition_data = user_data[condition]
                    if not isinstance(condition_data, dict) or dict_key not in condition_data:
                        continue
                        
                    extracted_data = condition_data[dict_key]
                    
                    # Handle nested list of dictionaries
                    if nested_key and isinstance(extracted_data, list):
                        for item in extracted_data:
                            if isinstance(item, dict) and nested_key in item:
                                processed_value = validate_and_process_value(item[nested_key])
                                if processed_value is not None:
                                    data.append({
                                        'user_id': user_id,
                                        'condition': condition,
                                        'sub_key': nested_key,
                                        'value': processed_value
                                    })
                    
                    # Handle dictionary
                    elif isinstance(extracted_data, dict):
                        for sub_key, value in extracted_data.items():
                            processed_value = validate_and_process_value(value)
                            if processed_value is not None:
                                data.append({
                                    'user_id': user_id,
                                    'condition': condition,
                                    'sub_key': sub_key,
                                    'value': processed_value
                                })
                    
                    # Handle single value
                    else:
                        processed_value = validate_and_process_value(extracted_data)
                        if processed_value is not None:
                            data.append({
                                'user_id': user_id,
                                'condition': condition,
                                'sub_key': None,
                                'value': processed_value
                            })
                            
        except FileNotFoundError:
            print(f"Error: File not found - {file}")
            continue
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if not data:
        print("Warning: No data was extracted. Check if the input keys and data structure are correct.")
        return pd.DataFrame(columns=['user_id', 'condition', 'sub_key', 'value'])
        
    return pd.DataFrame(data)


def visualize_distribution(df_pivot, cond1, cond2):
    """
    Visualizes the distribution of data for two conditions using histograms and boxplots.
    """
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df_pivot[cond1], kde=True, color='blue', label=cond1, bins=10, alpha=0.6)
    sns.histplot(df_pivot[cond2], kde=True, color='red', label=cond2, bins=10, alpha=0.6)
    plt.axvline(df_pivot[cond1].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(df_pivot[cond2].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title(f"Histogram of {cond1} and {cond2}")
    plt.legend()

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_pivot[[cond1, cond2]])
    plt.title(f"Boxplot of {cond1} and {cond2}")

    plt.tight_layout()
    plt.show()



def create_condition_barplot(df, colors, sub_keys=None, title=None, figsize=(12, 6),
                           rotate_labels=45, y_label="Count",
                           show_plot=True, condition_mapping=None, sub_key_mapping=None,
                           is_categorical=False, show_ci=False, show_standard_error=True, 
                           max_y_limit=7.9, yticks_upper_bound=8):
    """
    Creates a bar plot for categorical response counts or numerical data with optional 95% confidence intervals.
    
    Args:
        df (pd.DataFrame): DataFrame from compute_categorical_response_counts or numerical data
        sub_keys (list, optional): List of sub_keys to plot
        title (str): Plot title
        figsize (tuple): Figure size in inches
        rotate_labels (int): Rotation angle for x-axis labels
        y_label (str): Label for y-axis
        palette (str): Color palette for the bars
        show_plot (bool): Whether to display the plot immediately
        condition_mapping (dict): Mapping of response values to display labels
        sub_key_mapping (dict): Mapping of sub_key names to display labels
        is_categorical (bool): Whether the data is from categorical responses
        show_ci (bool): Whether to show 95% confidence intervals
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Copy DataFrame to avoid modifications
    df = df.copy()
    
    def calculate_error_bars(data, confidence=0.95, show_standard_error=True):
        """Calculate confidence interval or standard error for a series of data."""
        if len(data) < 2:
            return 0
        
        if show_standard_error:
            sem = stats.sem(data)
            return sem
        else:
            n = len(data)
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
            return ci
    
    # Filter for specified sub_keys if provided
    if sub_keys:
        df = df[df['sub_key'].isin(sub_keys)]
        if df.empty:
            raise ValueError("No data found for specified sub_keys")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if is_categorical:
        # Get unique questions and response values
        questions = sorted(df['sub_key'].unique())
        response_values = sorted(df['response_value'].unique())
        
        # Set up bar positions
        x = np.arange(len(questions))
        width = 0.8 / len(response_values)
        
        # Plot bars for each response value
        for i, response in enumerate(response_values):
            counts = []
            cis = []
            for question in questions:
                mask = (df['sub_key'] == question) & (df['response_value'] == response)
                count = df[mask]['count'].values
                counts.append(count[0] if len(count) > 0 else 0)
                
                if show_ci:
                    # For categorical data, use binomial confidence interval
                    total = df[df['sub_key'] == question]['count'].sum()
                    if total > 0:
                        success = count[0] if len(count) > 0 else 0
                        ci = stats.binom.interval(0.95, total, success/total if success > 0 else 0)
                        ci_width = (ci[1] - ci[0]) / 2
                    else:
                        ci_width = 0
                    cis.append(ci_width)
                
            
            position = x + (i - len(response_values)/2 + 0.5) * width
            label = condition_mapping.get(response, str(response)) if condition_mapping else str(response)
            bars = ax.bar(position, counts, width,
                         label=label,
                         color=colors[label])
            
            if show_ci:
                # Add error bars
                ax.errorbar(position, counts, yerr=cis, fmt='none', color='black', capsize=3)
            max_y_limit = 8.9
        
        # Set x-axis labels
        ax.set_xticks(x)
        labels = [sub_key_mapping.get(q, q) if sub_key_mapping else q for q in questions]
        ax.set_xticklabels(labels, rotation=rotate_labels, ha='right')
        
    else:
        # Get unique questions and conditions

        questions = [key for key in sub_key_mapping.keys() if key in df['sub_key'].unique()] if sub_key_mapping else sorted(df['sub_key'].unique())
        conditions = [key for key in condition_mapping.keys() if key in df['condition'].unique()] if condition_mapping else sorted(df['condition'].unique())

        # Set up bar positions
        x = np.arange(len(questions))

        
        if len(questions) > 1:
            width = 0.8 / len(conditions)
            positions = None
            ax.set_xticks(x)
            x_labels = [sub_key_mapping.get(q, q) if sub_key_mapping else q for q in questions]
        else:
            # Set bar positions explicitly
            positions = np.arange(len(conditions))
            width = 0.5  # Width of the bars
            x_labels = [condition_mapping.get(condition, condition) if condition_mapping else condition for condition in conditions]
            ax.set_xticks(positions) 

            


        # Plot bars for each condition

        for i, condition in enumerate(conditions):
            condition_data = df[df['condition'] == condition]
            means = []
            cis = []
            
            for question in questions:
                question_data = condition_data[condition_data['sub_key'] == question]['value']
                mean = question_data.mean() if not question_data.empty else 0
                means.append(mean)
                
                if show_ci:
                    ci = calculate_error_bars(question_data, show_standard_error=show_standard_error) if not question_data.empty else 0
                    cis.append(ci)
            
            print(positions)
            if positions is not None:
                position = positions[i]
            else:
                position = x + (i - len(conditions)/2 + 0.5) * width
   
            label = condition_mapping.get(condition, condition) if condition_mapping else condition
            bars = ax.bar(position, means, width,
                         label=label,
                         color=colors[label])

            
            if show_ci:
                # Add error bars
                ax.errorbar(position, means, yerr=cis, fmt='none', color='black', capsize=3)
        


        ax.set_xticklabels(x_labels, rotation=rotate_labels, ha='center', fontsize=23)
    
    # Customize the plot
    if title:
        ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(y_label, fontsize=23)
  
    
    # Add legend
    # ax.legend(title='Conditions' if not is_categorical else 'Responses', 
    #          bbox_to_anchor=(1.05, 1), 
    #          loc='upper left')
    # ax.legend(title='' if not is_categorical else 'Responses', 
    #          bbox_to_anchor=(1.0, 1.2), ncol=2, fontsize=15)
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=23)  # y-axis numbers font size
    if yticks_upper_bound is not None:
        ax.set_yticks(range(0, yticks_upper_bound))  # This will create ticks at 0, 1, 2, ..., 7
        ax.set_ylim(0,max_y_limit)

    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig
def plot_survey_responses(df, colors, sub_keys=None, show_plot=True, title=None, y_label=None, 
                         is_categorical=False, condition_mapping=None,show_ci=True,show_standard_error=True, **kwargs):
    """
    Convenience wrapper for plotting survey responses.
    
    Args:
        df (pd.DataFrame): DataFrame from compute_categorical_response_counts or numerical data
        sub_keys (list, optional): List of survey questions to plot
        show_plot (bool): Whether to display the plot immediately
        title (str, optional): Plot title
        y_label (str, optional): Label for y-axis
        is_categorical (bool): Whether the data is from categorical responses
        condition_mapping (dict): Mapping of response values to display labels
        **kwargs: Additional arguments passed to create_condition_barplot
    """
    if y_label is None:
        y_label = "Response Count" if is_categorical else "Average Value"
        
    fig = create_condition_barplot(
        df,
        colors = colors, 
        sub_keys=sub_keys,
        title=title,
        y_label=y_label,
        show_plot=show_plot,
        is_categorical=is_categorical,
        condition_mapping=condition_mapping,
        show_ci=show_ci,
        show_standard_error=show_standard_error,
        **kwargs
    )
    return fig



def compute_categorical_response_counts(df, categorical_questions, new_col_name="count", is_global=False, debug=False):
    """
    Computes response counts for categorical survey questions, ensuring all possible responses appear, even with 0 counts.

    Args:
        df (pd.DataFrame): DataFrame containing categorical responses with columns ['user_id', 'condition', 'sub_key', 'value'].
        categorical_questions (set): Set of categorical survey question texts.
        new_col_name (str): Name for the response count column (default: "count").
        is_global (bool): If True, treats all responses as global (not condition-based).
        debug (bool): If True, prints detailed counts before returning.

    Returns:
        pd.DataFrame: DataFrame with response counts per question and response value.
    """
    # Filter for relevant questions
    df_selected = df[df['sub_key'].isin(categorical_questions)].copy()

    if df_selected.empty:
        print("No categorical responses found. Check the provided question list.")
        return pd.DataFrame()

    # Get all possible response options (conditions being chosen)
    all_possible_responses = sorted(df_selected['value'].unique())


    # Count responses for each question and value
    df_counts = df_selected.groupby(['sub_key', 'value']).size().reset_index(name=new_col_name)

    # Ensure all possible responses appear for each question
    full_counts = []
    for question in categorical_questions:
        for response in all_possible_responses:
            print('response', response)
            count_value = df_counts.query("sub_key == @question and value == @response")[new_col_name].values
            count = count_value[0] if len(count_value) > 0 else 0
            full_counts.append({
                'sub_key': question,
                'response_value': response,  # renamed from 'value' for clarity
                new_col_name: count
            })

    df_full_counts = pd.DataFrame(full_counts).fillna(0)

    # Debug output
    if debug:
        print("\n*** Categorical Response Counts (Including Zeros) ***")
        for question in categorical_questions:
            print(f"\n🔹 {question}")
            print(df_full_counts[df_full_counts['sub_key'] == question])

    return df_full_counts
def compute_categorical_response_counts_merged_alignment(df, categorical_questions, new_col_name="count", debug=True):
    """
    Similar to compute_categorical_response_counts but combines alignment conditions into one.
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Define alignment responses to combine
    alignment_responses = [
        'Alignment Feedback',
        'Visual + Alignment Feedback'
    ]
    
    # Replace all alignment responses with a single name
    df.loc[df['value'].isin(alignment_responses), 'value'] = 'Either Alignment Conditions'
    
    # Now process as usual
    df_selected = df[df['sub_key'].isin(categorical_questions)].copy()

    if df_selected.empty:
        print("No categorical responses found. Check the provided question list.")
        return pd.DataFrame()

    # Get all possible response options after merging
    all_possible_responses = sorted(df_selected['value'].unique())

    # Count responses for each question and value
    df_counts = df_selected.groupby(['sub_key', 'value']).size().reset_index(name=new_col_name)

    # Ensure all possible responses appear for each question
    full_counts = []
    for question in categorical_questions:
        for response in all_possible_responses:
            count_value = df_counts.query("sub_key == @question and value == @response")[new_col_name].values
            count = count_value[0] if len(count_value) > 0 else 0
            full_counts.append({
                'sub_key': question,
                'response_value': response,
                new_col_name: count
            })

    df_full_counts = pd.DataFrame(full_counts).fillna(0)

    if debug:
        print("\n*** Categorical Response Counts (Including Zeros) ***")
        for question in categorical_questions:
            print(f"\n🔹 {question}")
            print(df_full_counts[df_full_counts['sub_key'] == question])

    return df_full_counts


def plot_survey_responses_stacked_barplot(df, colors, is_categorical=True, y_label='Votes', sub_key_mapping=None, condition_mapping=None, title=None, show_ci=False,
                         stack_alignment=True, show_plot=True):
    """
    Creates a bar plot with optional stacking for alignment conditions.
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not is_categorical:
        raise ValueError("This version only supports categorical data with stacking")
        
    # Get unique questions and responses
    questions = sorted(df['sub_key'].unique())
    
    # Define response categories
    alignment_responses = ['Alignment Feedback', 'Visual + Alignment Feedback']
    reward_feedback_responses = ['Reward Feedback']
    other_responses = [r for r in df['response_value'].unique() 
                      if r not in alignment_responses and r not in reward_feedback_responses]
    
    x = np.arange(len(questions))
    width = 0.35
    
    # Plot reward feedback bars (no stacking)
    reward_counts = []
    for question in questions:
        count = df[(df['sub_key'] == question) & 
                  (df['response_value'] == 'Reward Feedback')]['count'].values[0]
        reward_counts.append(count)
    
    label = condition_mapping.get('Reward Feedback', 'Reward Feedback') if condition_mapping else 'Reward Feedback'
    reward_bars = ax.bar(x, reward_counts, width, label=label, color=colors[label])
    
    # Stack only alignment responses (separate from reward)
    bottom_vals = np.zeros(len(questions))
    if stack_alignment:
        for response in alignment_responses:
            counts = []
            for question in questions:
                count = df[(df['sub_key'] == question) & 
                         (df['response_value'] == response)]['count'].values[0]
                counts.append(count)
            
            label = condition_mapping.get(response, response) if condition_mapping else response
            ax.bar(x + width, counts, width, bottom=bottom_vals, label=label, color=colors[label])
            bottom_vals += np.array(counts)
    
    # Customize the plot
    if title:
        ax.set_title(title)
    
    ax.set_ylabel(y_label, fontsize=23, labelpad=-10)
    
    # Set x-axis labels
    ax.set_xticks(x + width/2)
    labels = [sub_key_mapping.get(q, q) if sub_key_mapping else q for q in questions]
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=23)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=23)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig


def plot_heatmap(df, user_ids, x_labels=None, 
                 column_order=['reward_func_only_condition', 'alignment_reward_func_condition', 'alignment_visualization_reward_func_condition'], 
                 figsize=(12, 8), ax=None):


    df['user_id'] = pd.Categorical(df['user_id'], categories=[f'{i+1}' for i in range(len(user_ids))], ordered=True)
   
    pivot_df = df.pivot(index='user_id', columns='condition', values='value')
    pivot_df = pivot_df[column_order]  # Reorder columns

    
    # Use a custom function to color cells based on exact values
    def custom_color_function(val):
        if val <= 0.5:
            return "#ffffff"  # White for values <= 0.5
        elif val == 0.75:
            return "#ffb3c6"  # Light pink specifically for 0.75
        else:
            return "#800040"  # Dark pink for all other values (which should be 1.0)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap for the color bar only
    bounds = [0, 0.5, 0.75, 1.0]
    colors = ["#ffffff", "#ffb3c6", "#800040"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create the heatmap without using the colormap for cell colors
    sns.heatmap(pivot_df, 
                annot=True,  
                fmt='.2f',
                cmap=cmap,
                norm=norm,
                cbar_kws={'label': 'Fraction Correct'},
                vmin=0,    
                vmax=1,    
                ax=ax,
                annot_kws={'size': 20},
                linewidths=0.2,
                linecolor='black')
    
    # Apply the custom coloring to each cell based on its exact value
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            color = custom_color_function(val)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=1.0, linewidth=0))
    
    ax.set_ylabel('User ID', fontsize=23)
    ax.set_xlabel(None)
    ax.set_xticklabels([x_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()], 
                        rotation=0, ha='center', fontsize=23)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=23)
    
    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(23)  
    cbar.ax.tick_params(labelsize=23)
    cbar.ax.yaxis.set_label_coords(4.5, 0.5)  # Adjust these values as needed

    # Set tick positions to match your segments
    tick_locs = [0.25, 0.625, 0.875]
    cbar.set_ticks(tick_locs)
    
    # Set custom labels
    cbar.set_ticklabels(["≤ 0.5\n(Random\nor Worse)", "0.75", "1.0"])
    
    cbar.outline.set_linewidth(2)  

    # Draw segment boundaries on colorbar
    for bound in bounds:
        cbar.ax.hlines(bound, 0, 1, color='black', linewidth=2, transform=cbar.ax.get_yaxis_transform())
        
    # Bring annotations to the front
    for text in ax.texts:
        text.set_zorder(100)

    plt.tight_layout()
    plt.show()
    return fig

def get_colors(data):
    colors = sns.color_palette("husl", 9)

    histogram_colors = {key: colors[6] if 'Control' in key 
                            else colors[0] if 'Visual' in key 
                            else colors[7] 
                    for key in data}
                
    return histogram_colors