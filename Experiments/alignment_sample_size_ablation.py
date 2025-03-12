import os
import pickle
import numpy as np
from collections import OrderedDict
from scipy import stats
from typing import Dict, List, Tuple, Any, Callable
import random
import utils 
class TrajectoryAnalyzer:
    # Class constants
    TRAJECTORY_LENGTH = 200
    GROUND_TRUTH = str({
        'hungry and thirsty': 0.0,
        'hungry and not thirsty': 0.0,
        'not hungry and thirsty': 1.0,
        'not hungry and not thirsty': 1.0
    })
    GROUND_TRUTH_ID = 2
    NUM_HUMAN_REWARD_FUNCS = 54
    
    FITNESS_THRESHOLDS = {
        'worst': (None, 1),
        'low': (1, 30),
        'medium': (30, 60),
        'high': (60, None)
    }

    def __init__(
        self,
        base_dir: str,
        start_state: Tuple[int, int] = (0, 0),
        sample_size: int = 12,
        mixture_sampling: bool = True,
        save: bool = True
    ):
        self.base_dir = base_dir
        self.start_state = start_state
        self.sample_size = sample_size
        self.mixture_sampling = mixture_sampling
        self.save = save
        

    def load_trajectory_data(self, seed: int) -> Tuple[List, List]:
        """Load trajectory and fitness data for a given seed."""
        trajectory_path = f'{self.base_dir}/trajectories_(0, 0, 1.0, 1.0)_seed{seed}.pkl'
        fitness_path = f'{self.base_dir}/fitness_all_episodes_(0, 0, 1.0, 1.0)_seed{seed}.pkl'
        trajectories = utils.get_pickle_data(trajectory_path)[:-1]
        fitnesses = utils.get_pickle_data(fitness_path)[:-1]

        return trajectories, fitnesses

    def categorize_trajectories(self, trajectories: List, fitnesses: List) -> Dict[str, List[int]]:
        """Categorize trajectories based on fitness scores."""

        split_data = {category: [] for category in self.FITNESS_THRESHOLDS.keys()}
        
        for traj_id, (trajectory, fitness) in enumerate(zip(trajectories, fitnesses)):
            assert len(trajectory) == self.TRAJECTORY_LENGTH
            
            for category, (min_val, max_val) in self.FITNESS_THRESHOLDS.items():
                if ((min_val is None or fitness > min_val) and 
                    (max_val is None or fitness <= max_val)):
                    split_data[category].append(traj_id)
                    break
                    
        return split_data

    def sample_trajectories(self, split_data: Dict[str, List[int]]) -> List[int]:
        """Sample trajectories from each category according to distribution."""
        groups = list(self.FITNESS_THRESHOLDS.keys())
        num_groups_except_worst = len(groups) - 1
        
        # Calculate sample sizes for each group
        sample_sizes = {'worst': 0}  # We don't sample from worst category
        remaining_samples = self.sample_size - sample_sizes['worst']
        base_size = remaining_samples // num_groups_except_worst
        remainder = remaining_samples % num_groups_except_worst
        
        for idx, group in enumerate(groups[1:]):
            sample_sizes[group] = base_size + (1 if idx < remainder else 0)
            
        # Sample from each group
        selected_trajectories = []
        for group in groups:
            num_to_sample = min(sample_sizes[group], len(split_data[group]))
            selected_trajectories.extend(random.sample(split_data[group], num_to_sample))
            
        return selected_trajectories
    def get_sampled_trajectories(
        self,
        seed: int
    ) -> np.ndarray:
        """Get sampled trajectories for a given seed.
        
        Args:
            seed: The seed number for loading trajectory data
            
        Returns:
            np.ndarray: Array of sampled trajectories
        """
        # Load and process data
        trajectories, fitnesses = self.load_trajectory_data(seed)
        
        if self.mixture_sampling:
            # Sample trajectories
            split_data = self.categorize_trajectories(trajectories, fitnesses)
            selected_indices = self.sample_trajectories(split_data)
            return np.array(trajectories)[selected_indices]
        else:
            return np.array(trajectories)

    def ensure_second_position(self, reward_list):
        """Ensure the ground truth reward function is in the second position."""
        target = {'hungry and thirsty': 0.0, 'hungry and not thirsty': 0.0, 
              'not hungry and thirsty': 1.0, 'not hungry and not thirsty': 1.0}
        
        if not (isinstance(reward_list, list) and len(reward_list) == 1 and 
                isinstance(reward_list[0], list) and len(reward_list[0]) == 2):
            raise ValueError("Input should be a list containing one list of exactly two dictionaries.")

        first, second = reward_list[0]

        if first == target:
            return [[second, first]]
        elif second != target:
            return [[first, target]]
        return reward_list

    def calculate_alignment_scores(
        self,
        raw_traj: np.ndarray,
        get_reward_functions: Callable,
        compute_rankings: Callable,
        get_trajectory_alignment_coefficient_score: Callable
    ) -> List[float]:
        """Calculate alignment scores for trajectories."""
        alignment_scores = []
        ground_truth_key = str(self.GROUND_TRUTH)
        
        for id in range(self.NUM_HUMAN_REWARD_FUNCS):
            # Get reward functions
            _, selected_reward_functions = get_reward_functions([(id, self.GROUND_TRUTH_ID)])
            selected_reward_functions = self.ensure_second_position(selected_reward_functions)
            
            # Flatten reward functions
            flat_rewards = [item for sublist in selected_reward_functions for item in sublist]
            returns_rankings = OrderedDict(compute_rankings(raw_traj, flat_rewards))
            
            # Extract the first key dynamically
            first_key = next(iter(returns_rankings['returns']))
            
            # Compute alignment score
            score = get_trajectory_alignment_coefficient_score(
                returns_rankings['returns'][first_key],
                returns_rankings['returns'][ground_truth_key]
            )
            alignment_scores.append(score)
        
        return alignment_scores

    def analyze_correlation(
        self,
        num_iterations: int,
        baseline_scores: List[float],
        seed_range: range,
        get_reward_functions: Callable,
        compute_rankings: Callable,
        get_alignment_score: Callable
    ) -> float:
        """Run correlation analysis for multiple iterations."""
        correlations = {'AllCorrelations': [], 'MeanSTDCorrelation': None}
        
        for _ in range(num_iterations):
            for seed in seed_range:
                 # Load and process data
                raw_traj = self.get_sampled_trajectories(seed)
 
                # Calculate alignment scores
                alignment_scores = self.calculate_alignment_scores(
                    raw_traj=raw_traj,
                    get_reward_functions=get_reward_functions,
                    compute_rankings=compute_rankings,
                    get_trajectory_alignment_coefficient_score=get_alignment_score
                )
                
                # Calculate correlation
                correlation = stats.kendalltau(baseline_scores, alignment_scores).statistic
                correlations['AllCorrelations'].append(correlation)
                print(f"Iteration correlation: {correlation}")

        correlations['MeanSTDCorrelation'] = [np.mean(correlations['AllCorrelations']),np.std(correlations['AllCorrelations'])]
        if self.save:
            utils.save(correlations, f'kendall_tau_correlations_traj_samplesize{self.sample_size}',f'{os.getcwd()}')
        return correlations['MeanSTDCorrelation']


from user_study_utils import get_specific_reward_functions, compute_trajectory_rewards_and_rankings
from get_alignment import get_trajectory_alignment_coefficient_score

analyzer = TrajectoryAnalyzer(
    base_dir=f'{os.getcwd()}/Experiments/Env_Easy/SameStartState/0_0/alg_Q_learn/env_timesteps_200/envs_10/alpha_lr_0.05/eps_0.15/gamma_0.99/num_episodes_10000/record_freq_1/num_tests_1/(0, 0, 1.0, 1.0)',
    sample_size=500
)

#baseline_scores = utils.get_pickle_data(f'{os.getcwd()}/kendall_tau_correlations_traj_samplesize1200.pkl')
baseline_scores  = [0.792527822633971, 0.7339138708563301, 1.0, 0.8331400175646132, 0.7696871390358779, 0.9148949077702867, 0.9277511679827082, 0.7825477638612957, 0.9837302035839364, 0.9758926937758544, 0.9288148902767897, 0.9028349815327478, 1.0, 0.9445301053083295, 0.8583163446799044, 0.9309742912070588, 0.9837302035839364, 0.9083395833307838, 0.901284081555456, 0.7696871390358779, 0.8842194305784683, 0.8978480534142589, 0.9044372406953831, 0.5880708517107479, 0.8038169490570671, 0.9905604507149732, 0.7038641331728468, 0.9622945141325171, 0.7987943729060683, 0.7375129842758131, 0.8195774419077776, 1.0, 0.8081256594604121, 0.9699257645568832, 1.0, 0.9837302035839364, 0.8908258780422589, 0.8166217404260769, 0.9633191781140322, 0.9622945141325171, 0.9431620255020509, 0.8664526250137515, 0.7630490234631482, 0.7780401498288032, 0.9007888186642276, 0.8469508827863501, 0.8135086991819009, 0.7603852154018516, 0.9241964485100489, 0.805972347145285, 0.9622945141325171, 0.7987996770056155, 0.7696871390358779, 0.8062373377738071]
utils.save(baseline_scores, f'kendall_tau_correlations_traj_samplesize1200',f'{os.getcwd()}')
input('wait')
mean_correlation = analyzer.analyze_correlation(
    num_iterations=50,
    baseline_scores=baseline_scores,
    seed_range=range(0, 1),
    get_reward_functions=get_specific_reward_functions,
    compute_rankings=compute_trajectory_rewards_and_rankings,
    get_alignment_score=get_trajectory_alignment_coefficient_score
)
print(f"Mean and std correlation: {mean_correlation}")
