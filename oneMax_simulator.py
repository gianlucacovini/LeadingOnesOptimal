import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
import math
import os
from functools import lru_cache

from optimal_oneMax import variables_calculator

from heuristic_policy import K_calculator, LeadingOnes, OneMax

curr_dir = os.getcwd()

core_num = 24

def mutate(x, k):
    """Flip k random bits in x"""
    n = len(x)
    indices = np.random.choice(n, k, replace=False)
    x_new = x.copy()
    x_new[indices] = 1 - x_new[indices]
    return x_new

@lru_cache(maxsize=None)
def optimal_policy_oneMax(n):
    with multiprocessing.Pool(processes=core_num) as pool:
        K, _, __ = variables_calculator(n, pool)
    return K

def rls_leading_ones(n, K):
    """ (1+1) RLS for LeadingOnes problem using dynamic k policy """
    x = np.random.randint(2, size=n)  # Start with a random bitstring
    lo_best = LeadingOnes(tuple(x))
    om_best = OneMax(tuple(x))
    evaluations = 0
    
    while lo_best < n:
        k = int(K[om_best])
        x_new = mutate(x, k)
        lo_new = LeadingOnes(tuple(x_new))
        om_new = OneMax(tuple(x_new))
        evaluations += 1

        if lo_new > lo_best or (lo_new == lo_best and om_new > om_best):
            x = x_new
            lo_best = lo_new
            om_best = om_new

    return evaluations

def run_rls_leading_ones(n, num_runs, num_cores):
    evaluations = np.zeros(num_runs, dtype=int)
    K = optimal_policy_oneMax(n)  # Precompute K values for given n
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(rls_leading_ones, n, K) for _ in range(num_runs)]
        for i, future in enumerate(as_completed(futures)):
            evaluations[i] = future.result()
    
    return evaluations

def plot_boxplot(data, labels, title, filename):
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels)
    plt.title(title)
    plt.xlabel('Policy')
    plt.ylabel('Evaluations')
    plt.grid(True)
    if not os.path.exists('Box_plots_om'):
        os.makedirs('Box_plots_om')
    plt.savefig(os.path.join('Box_plots_om', filename))
    plt.close()

if __name__ == "__main__":
    for n in range(2, 16):
        num_runs = 500  # Number of Monte Carlo runs
        num_cores = 24  # Number of cores for parallelization
    
        # Run RLS with both policies
        random.seed(42)
        np.random.seed(42)
        evaluations_k_calculator = run_rls_leading_ones(n, num_runs, num_cores)
    
        mean_evals_k_calculator = round(np.mean(evaluations_k_calculator), 3)
        std_evals_k_calculator = round(np.std(evaluations_k_calculator), 3)
        
        print(f"n = {n}")
        print(f"Mean evaluations (K_calculator): {mean_evals_k_calculator}")
        print(f"Standard deviation of evaluations (K_calculator): {std_evals_k_calculator}")
            
        # Plot and save boxplot for both policies
        # plot_boxplot([evaluations_k_calculator, evaluations_optimal_policy], 
        #              ['K_calculator', 'OptimalPolicyFitness'], 
        #              f"Boxplot of Evaluations for LeadingOnes Problem (n={n}, runs={num_runs})",
        #              f'{n}_boxplot.png')
        
        with open('results_simulation.txt', 'a') as file:
            file.write(f"n: {n}\n")
            file.write(f"Mean (K_calculator): {mean_evals_k_calculator}\n")
            file.write(f"Std Dev (K_calculator): {std_evals_k_calculator}\n")
    
        # # Plot expected times
        # plt.figure(figsize=(12, 8))
        # plt.bar(['K_calculator', 'OptimalPolicyFitness'], [mean_evals_k_calculator, mean_evals_optimal_policy],
        #         yerr=[std_evals_k_calculator, std_evals_optimal_policy], capsize=5)
        # plt.title('Expected Time for LeadingOnes Problem')
        # plt.xlabel('Policy')
        # plt.ylabel('Mean Evaluations')
        # plt.grid(True)
        # if not os.path.exists('Box_plots'):
        #     os.makedirs('Box_plots')
        # plt.savefig(os.path.join(curr_dir, 'Box_plots', 'expected_times.png'))
   

