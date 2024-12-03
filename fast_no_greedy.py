import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import os
import multiprocessing
from functools import lru_cache
import time

core_num = 24
curr_dir = os.getcwd()

def generate_bit_strings(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))

def sort_bit_strings(bit_strings):
    # Sort the bit strings based on the lexicographic order
    return sorted(map(tuple, bit_strings), key=lambda x: (LeadingOnes(x), OneMax(x)), reverse=False)

@lru_cache(maxsize=None)
def LeadingOnes(x):
    return np.argmax(np.cumprod(x) == 0) if np.any(np.cumprod(x) == 0) else len(x)

@lru_cache(maxsize=None)
def OneMax(x):
    return np.sum(x)

def categorize_bit_strings(n):
    bit_strings = generate_bit_strings(n)
    lo_om = np.array([(LeadingOnes(tuple(bits)), OneMax(tuple(bits))) for bits in bit_strings])
    unique_keys = np.unique(lo_om, axis=0)
    
    results_dict = {tuple(key): [] for key in unique_keys}
    for i, bits in enumerate(bit_strings):
        key = tuple(lo_om[i])
        results_dict[key].append(bits)
    
    return results_dict

def terms_calculator(args):
    l, n, couples, T, in_prob, k = args
    
    A = np.zeros((n-l, n-l))
    
    b = np.ones(n-l) 
    
    def process_combination(comb):
        starting_m, arriving_m = comb
        if (l, starting_m) in couples:
            current_nodes = np.array(couples[(l, starting_m)])
            num_couples = len(current_nodes)
            in_prob[(l, starting_m)] = num_couples / 2**n

            if (l, arriving_m) in couples:
                nodes = np.array(couples[(l, arriving_m)])
                distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
                valid_indices = np.where(distances == k)
                A[starting_m-l, arriving_m-l] = valid_indices[0].size / num_couples
                b[starting_m-l] += A[starting_m-l, arriving_m-l] * T[l, arriving_m]
    
    # Create a list of combinations to be processed
    combinations_list = list(itertools.product(range(l, n), range(l, n)))
    
    # Use multiprocessing to process combinations
    with multiprocessing.Pool(processes=core_num) as pool:
        pool.map(process_combination, combinations_list)
    
    return A, b

def variables_calculator(n, pool):
    couples = categorize_bit_strings(n)
    T = np.zeros((n, n))
    in_prob = {}
    k = 1  # Example value for k

    args = [(l, n, couples, T, in_prob, k) for l in range(n)]
    results = pool.map(terms_calculator, args)

    K = np.zeros((n, n))
    T = np.zeros((n, n))
    for A, b in results:
        K += A  # Example of processing results
        T += b  # Example of processing results

    Expected_time = np.sum(T)  # Example calculation
    return K, T, Expected_time

def plot_2d_matrix(matrix, n, data, save=False):
    fig, ax = plt.subplots()
    matrix_data_masked = np.ma.masked_where(matrix == 0, matrix)
    cax = ax.matshow(matrix_data_masked, cmap='viridis')
    fig.colorbar(cax)
    if data == "K":
        ax.set_title(f'Values of K; n = {n}')
    if data == "T":
        ax.set_title(f'Values of T; n = {n}')
    ax.set_xticks(np.arange(matrix_data_masked.shape[1]))
    ax.set_yticks(np.arange(matrix_data_masked.shape[0]))
    ax.set_xticklabels(np.arange(0, matrix_data_masked.shape[1]))
    ax.set_yticklabels(np.arange(0, matrix_data_masked.shape[0]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(matrix_data_masked.shape[0]):
        for j in range(matrix_data_masked.shape[1]):
            value = matrix_data_masked[i, j]
            if not np.isnan(value):
                if data == "K":
                    ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='black')
                if data == "T":
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black')

    if save and data == "K":
        plt.savefig(os.path.join(curr_dir, 'plots', 'K_plots', 'No_greedy', f'{n}.png'), format='png')
    elif save and data == "T":
        plt.savefig(os.path.join(curr_dir, 'plots', 'T_plots', 'No_greedy', f'{n}.png'), format='png')

def process_iteration(n, pool):
    start_time = time.time()
    K, T, Expected_time = variables_calculator(n, pool)
    end_time = time.time()
    print(f"Expected time for n = {n}: ", round(Expected_time, 3))
    print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds")

    with open('results.txt', 'a') as file:
        file.write("Policy (LO(x), OM(x))\n")
        file.write(f"n: {n}\n")
        file.write(f"Expected time: {Expected_time}\n")
        file.write(f"K: {K}\n")
        file.write(f"T: {T}\n")

    plot_2d_matrix(K, n, "K", True)
    plot_2d_matrix(T, n, "T", True)
    
    return Expected_time

if __name__ == "__main__":
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(1, 11):
            process_iteration(n, pool)

