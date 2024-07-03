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
    # Sort the bit strings based on the OneMax value
    return sorted(map(tuple, bit_strings), key=lambda x: OneMax(x), reverse=False)

@lru_cache(maxsize=None)
def OneMax(x):
    return np.sum(x)

def categorize_bit_strings(n):
    bit_strings = generate_bit_strings(n)
    om_values = np.array([OneMax(tuple(bits)) for bits in bit_strings])
    unique_keys = np.unique(om_values)
    
    results_dict = {key: [] for key in unique_keys}
    for i, bits in enumerate(bit_strings):
        key = om_values[i]
        results_dict[key].append(bits)
    
    return results_dict

def mu_loop(args):
    k, m, n, couples, num_couples, P, current_nodes, mu = args
    
    if (m+mu) in couples:
        nodes = np.array(couples[(m+mu)])
        distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
        valid_indices = np.where(distances == k)
        
        if len(valid_indices[0]) > 0:
            for i, j in zip(*valid_indices):
                node = nodes[i]
                if OneMax(tuple(node)) > OneMax(tuple(current_nodes[j])):
                    P[m+mu] += 1 / (math.comb(n, k) * num_couples)
                    
    return P

def k_loop(args):
    k, m, n, couples, num_couples, T = args

    P = np.zeros(n+1)
    current_nodes = np.array(couples[(m)])
        
    for mu in range(-k+1, n-m+1):
        if (m+mu) in couples:
            nodes = np.array(couples[(m+mu)])
            distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
            valid_indices = np.where(distances == k)
            
            if len(valid_indices[0]) > 0:
                for i, j in zip(*valid_indices):
                    node = nodes[i]
                    if OneMax(tuple(node)) > OneMax(tuple(current_nodes[j])):
                        P[m+mu] += 1 / (math.comb(n, k) * num_couples)
    
    P[m] = 1 - np.sum(P)
    
    if P[m] != 1:
        E_current = round((1 + np.sum(P * T)) / (1 - P[m]), 3)
    else:
        E_current = 1
    
    return E_current

def variables_calculator(n, pool):
    K = np.zeros(n+1)
    T = np.zeros(n+1)
    in_prob = np.zeros(n+1)

    couples = categorize_bit_strings(n)

    c = len(couples) - 2

    current_couple = list(couples.keys())[c]

    K[n-1] = 1
    T[n-1] = n

    in_prob[n] = 1 / 2**n
    in_prob[n-1] = 1 / 2**n

    while c != 0:
        c -= 1
        current_couple = list(couples.keys())[c]

        m = current_couple

        if current_couple == 0:
            break

        num_couples = len(couples[(m)])
        in_prob[current_couple] = num_couples / 2**n
        
        args_list = [(k, m, n, couples, num_couples, T) for k in range(1, n - m + 1)]

        E_couple = pool.map(k_loop, args_list)

        E_opt = np.min(E_couple)
        k_opt = np.argmin(E_couple) + 1

        K[current_couple] = k_opt
        T[current_couple] = E_opt

    K[0] = n
    T[0] = 1
    in_prob[0] = 1 / 2**n

    Expected_time = 1 + (in_prob * T).sum()

    return K, T, Expected_time

def plot_2d_array(array_data, sav_dir=None):
    matrix_data = np.transpose(np.asmatrix(array_data[:-1]))
    
    fig, ax = plt.figure(), plt.gca()
    
    # Plotting the matrix with color encoded values
    c = ax.imshow(matrix_data, cmap='viridis', interpolation='nearest')
    fig.colorbar(c, ax=ax)  # Add a colorbar to a plot

    ax.set_ylabel('OneMax fitness')
    
    if sav_dir == "K":
        ax.set_title('Color Map of values of k')
    if sav_dir == "T":
        ax.set_title('Color Map of values of E[T]')

    # Tick labels for each row and column
    ax.set_xticks(np.arange(matrix_data.shape[1]))
    ax.set_yticks(np.arange(matrix_data.shape[0]))
    ax.set_xticklabels(np.arange(0, matrix_data.shape[1]))
    ax.set_yticklabels(np.arange(0, matrix_data.shape[0]))

    # Rotate the tick labels for better display
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[1]):
            value = matrix_data[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='black')
    
    if sav_dir == "K":
        plt.savefig(os.path.join(curr_dir, 'K_plots', 'OneMax', f'{n}_fitness.png'), format='png')
    if sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'T_plots', 'OneMax', f'{n}_fitness.png'), format='png')

    plt.show()

def process_iteration(n, pool):
    start_time = time.time()

    K, T, Expected_time = variables_calculator(n, pool)

    end_time = time.time()

    print(f"Expected time for n = {n}: ", round(Expected_time, 3))
    print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds")

    with open('results.txt', 'a') as file:
        file.write("Policy (OM(x))\n")
        file.write(f"n: {n}\n")
        file.write(f"Expected time: {Expected_time}\n")
        file.write(f"K: {K}\n")
        file.write(f"T: {T}\n")

    plot_2d_array(K, "K")
    plot_2d_array(T, "T")

if __name__ == "__main__":
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(1, 14):
            process_iteration(n, pool)