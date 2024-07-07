import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import os

from heuristic_policy import LeadingOnes, OneMax

curr_dir = os.getcwd()

def mutate(x, p):
    """Flip bits in x randomly with probability p"""
    # Generate random numbers for each bit in x
    random_numbers = np.random.rand(len(x))
    
    # Flip bits where the random number is less than p
    x_new = np.where(random_numbers < p, 1 - x.astype(int), x.astype(int))

    return x_new

def optimal_adaptive_mr(l, n):
    return 1/(n-l+1)

def EA_leading_ones(n, p_policy):
    """ (1+1) EA for LeadingOnes problem """
    x = np.random.randint(2, size=n)  # Start with a random bitstring
    lo_best = LeadingOnes(tuple(x))
    om_best = OneMax(tuple(x))
    evaluations = 0
    # K = K_calculator(n)  # Precompute K values for given n
    
    while lo_best < n:
        if p_policy == 'static':
            p = 1.59/n
        elif p_policy == 'dynamic':
            p = optimal_adaptive_mr(lo_best, n)
        x_new = mutate(x, p)
        lo_new = LeadingOnes(tuple(x_new))
        om_new = OneMax(tuple(x_new))
        evaluations += 1

        if lo_new > lo_best: # or (lo_new == lo_best and om_new > om_best):
            x = x_new
            lo_best = lo_new
            om_best = om_new

    return evaluations

def run_EA_leading_ones(n, num_runs, num_cores, p_policy):
    evaluations = np.zeros(num_runs, dtype=int)
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(EA_leading_ones, n, p_policy) for _ in range(num_runs)]
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
    if not os.path.exists('Box_plots_EA'):
        os.makedirs('Box_plots_EA')
    plt.savefig(os.path.join('Box_plots_EA', filename))
    plt.close()

if __name__ == "__main__":
    for n in range(2, 7):
        num_runs = 500  # Number of Monte Carlo runs
        num_cores = 24  # Number of cores for parallelization
    
        # Run RLS with both policies
        random.seed(42)
        np.random.seed(42)
        evaluations_static = run_EA_leading_ones(n, num_runs, num_cores, 'static')
        
        random.seed(42)
        np.random.seed(42)
        evaluations_dynamic = run_EA_leading_ones(n, num_runs, num_cores, 'dynamic')
    
        mean_evals_static = round(np.mean(evaluations_static), 3)
        std_evals_static = round(np.std(evaluations_static), 3)
        
        mean_evals_dynamic = round(np.mean(evaluations_dynamic), 3)
        std_evals_dynamic = round(np.std(evaluations_dynamic), 3)
        
        print(f"n = {n}")
        print(f"Mean evaluations (static): {mean_evals_static}")
        print(f"Standard deviation of evaluations (static): {std_evals_static}")
        print(f"Mean evaluations (dynamic): {mean_evals_dynamic}")
        print(f"Standard deviation of evaluations (dynamic): {std_evals_dynamic}")
            
        # Plot and save boxplot for both policies
        plot_boxplot([evaluations_static, evaluations_dynamic], 
                     ['static', 'dynamic'], 
                     f"Boxplot of Evaluations for LeadingOnes Problem (n={n}, runs={num_runs})",
                     f'{n}_boxplot.png')
        
        with open('results_simulation.txt', 'a') as file:
            file.write(f"n: {n}\n")
            file.write(f"Mean (static): {mean_evals_static}\n")
            file.write(f"Std Dev (static): {std_evals_static}\n")
            file.write(f"Mean (dynamic): {mean_evals_dynamic}\n")
            file.write(f"Std Dev (dynamic): {std_evals_dynamic}\n")
    
        # # Plot expected times
        # plt.figure(figsize=(12, 8))
        # plt.bar(['static', 'dynamic'], [mean_evals_static, mean_evals_dynamic],
        #         yerr=[std_evals_static, std_evals_dynamic], capsize=5)
        # plt.title('Expected Time for LeadingOnes Problem')
        # plt.xlabel('Policy')
        # plt.ylabel('Mean Evaluations')
        # plt.grid(True)
        
        # if not os.path.exists('Box_plots_EA'):
        #     os.makedirs('Box_plots_EA')
        # plt.savefig(os.path.join(curr_dir, 'Box_plots_EA', 'expected_times.png'))