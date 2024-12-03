import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
import math
import os

from heuristic_policy import K_calculator, LeadingOnes, OneMax
from optimal_oneMax import process_iteration

curr_dir = os.getcwd()

# def custom_K(l, m):
#     K = np.array([
#     [6, 6, 3, 3, 1, 6],
#     [0, 5, 4, 2, 2, 1],
#     [0, 0, 3, 2, 2, 1],
#     [0, 0, 0, 2, 1, 1],
#     [0, 0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 0, 1]
#     ])
    
#     return K[l, m]

def custom_K(m):
    K = np.ones(m)
    return K[m]

def K_oneMax(n):
    core_num = 24
    with multiprocessing.Pool(processes=core_num) as pool:
         K_oneMax = process_iteration(n, pool).astype(int)
            
    return K_oneMax

def mutate(x, k):
    """Flip k random bits in x"""
    n = len(x)
    indices = np.random.choice(n, k, replace=False)
    x_new = x.copy()
    x_new[indices] = 1 - x_new[indices]
    return x_new

def rls_leading_ones(n, k_policy, K = None):
    """ (1+1) RLS for LeadingOnes problem using dynamic k policy """
    x = np.random.randint(2, size=n)  # Start with a random bitstring
    lo_best = LeadingOnes(tuple(x))
    om_best = OneMax(tuple(x))
    evaluations = 0
    if k_policy == 'K_calculator':
        K = K_calculator(n)  # Precompute K values for given n
    
    while lo_best < n:
        if k_policy == 'K_calculator':
            k = K[lo_best, om_best]
        elif k_policy == 'OptimalPolicyFitness':
            k = OptimalPolicyFitness(lo_best, n)
        elif k_policy == 'oneMax':
            k = K[om_best]
        elif k_policy == 'custom':
            # k = custom_K(lo_best, om_best)
            # k = custom_K(om_best)
            k = 1
        x_new = mutate(x, k)
        lo_new = LeadingOnes(tuple(x_new))
        om_new = OneMax(tuple(x_new))
        evaluations += 1

        if lo_new >= lo_best: # or (lo_new == lo_best and om_new > om_best):
            x = x_new
            lo_best = lo_new
            om_best = om_new

    return evaluations

def OptimalPolicyFitness(i, n):
    return math.floor(n / (i + 1))

def run_rls_leading_ones(n, num_runs, num_cores, k_policy, K=None):
    evaluations = np.zeros(num_runs, dtype=int)
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(rls_leading_ones, n, k_policy, K) for _ in range(num_runs)]
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
    if not os.path.exists('Box_plots_noGreedy'):
        os.makedirs('Box_plots_noGreedy')
    plt.savefig(os.path.join('Box_plots_noGreedy', filename))
    plt.close()

if __name__ == "__main__":
    k_policy = 'optimalPolicyFitness'
    for n in range(1, 13):
        
        if k_policy == 'optimalPolicyFitness':
            
            random.seed(42)
            np.random.seed(42)
            evaluations_optimal = run_rls_leading_ones(n, 1000, 24, 'optimalPolicyFitness')
            
            mean_evals_optimal = round(np.mean(evaluations_optimal), 3)
            
            print("theory: ", mean_evals_optimal/n**2)
            
            print("mean: ", mean_evals_optimal)
            
            std_evals_optimal = round(np.std(evaluations_optimal), 3)
            print("std: ", std_evals_optimal)
    
        if k_policy == 'oneMax':
            K = K_oneMax(n)
            
            random.seed(42)
            np.random.seed(42)
            evaluations_oneMax = run_rls_leading_ones(n, 1000, 24, 'oneMax', K)
            
            mean_evals_oneMax = round(np.mean(evaluations_oneMax), 3)
            print(mean_evals_oneMax)
            
            std_evals_oneMax = round(np.std(evaluations_oneMax), 3)
            print(std_evals_oneMax)
    
        elif k_policy == 'custom':
            random.seed(42)
            np.random.seed(42)
            evaluations_custom = run_rls_leading_ones(n, 500, 24, 'custom')
            
            print("n = ", n)
            
            
            mean_evals_custom = round(np.mean(evaluations_custom), 3)
            
            print("theory: ", mean_evals_custom/n**2) # Questo è importante come studio empirico!!
            
            print("mean: ", mean_evals_custom)
            
            std_evals_custom = round(np.std(evaluations_custom), 3)
            print("std: ", std_evals_custom)
    
    
    for n in range(2, 2):
        num_runs = 500  # Number of Monte Carlo runs
        num_cores = 24  # Number of cores for parallelization
    
        # Run RLS with both policies
        random.seed(42)
        np.random.seed(42)
        evaluations_k_calculator = run_rls_leading_ones(n, num_runs, num_cores, 'K_calculator')
        
        random.seed(42)
        np.random.seed(42)
        evaluations_optimal_policy = run_rls_leading_ones(n, num_runs, num_cores, 'OptimalPolicyFitness')
    
        mean_evals_k_calculator = round(np.mean(evaluations_k_calculator), 3)
        std_evals_k_calculator = round(np.std(evaluations_k_calculator), 3)
        
        mean_evals_optimal_policy = round(np.mean(evaluations_optimal_policy), 3)
        std_evals_optimal_policy = round(np.std(evaluations_optimal_policy), 3)
        
        print(f"n = {n}")
        print(f"Mean evaluations (K_calculator): {mean_evals_k_calculator}")
        print(f"Standard deviation of evaluations (K_calculator): {std_evals_k_calculator}")
        print(f"Mean evaluations (OptimalPolicyFitness): {mean_evals_optimal_policy}")
        print(f"Standard deviation of evaluations (OptimalPolicyFitness): {std_evals_optimal_policy}")
            
        # Plot and save boxplot for both policies
        plot_boxplot([evaluations_k_calculator, evaluations_optimal_policy], 
                     ['K_calculator', 'OptimalPolicyFitness'], 
                     f"Boxplot of Evaluations for LeadingOnes Problem (n={n}, runs={num_runs})",
                     f'{n}_boxplot.png')
        
        # with open('results_simulation.txt', 'a') as file:
        #     file.write(f"n: {n}\n")
        #     file.write(f"Mean (K_calculator): {mean_evals_k_calculator}\n")
        #     file.write(f"Std Dev (K_calculator): {std_evals_k_calculator}\n")
        #     file.write(f"Mean (OptimalPolicyFitness): {mean_evals_optimal_policy}\n")
        #     file.write(f"Std Dev (OptimalPolicyFitness): {std_evals_optimal_policy}\n")
    
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
   

