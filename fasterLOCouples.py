import numpy as np
import math
import itertools
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import multiprocessing

core_num = 1

curr_dir = os.getcwd()

def generate_bit_strings(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))

def LeadingOnes(x):
    return np.argmax(np.cumprod(x) == 0) if np.any(np.cumprod(x) == 0) else len(x)

def OneMax(x):
    return np.sum(x)

def categorize_bit_strings(n):
    bit_strings = generate_bit_strings(n)
    lo_om = np.array([(LeadingOnes(bits), OneMax(bits)) for bits in bit_strings])
    unique_keys = np.unique(lo_om, axis=0)
    
    results_dict = {tuple(key): [] for key in unique_keys}
    for i, bits in enumerate(bit_strings):
        key = tuple(lo_om[i])
        results_dict[key].append(bits)
    
    return results_dict

def k_loop(args):
    k, l, m, n, couples, num_couples, in_prob, T = args

    P = np.zeros((n+1, n+1))
    current_nodes = np.array(couples[(l, m)])
    
    for lambda_ in range(0, n-l+1):
        for mu in range(-k+1, n-l+1):
            if (l+lambda_, m+mu) in couples:
                nodes = np.array(couples[(l+lambda_, m+mu)])
                distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
                valid_indices = np.where(distances == k)
                
                if len(valid_indices[0]) > 0:
                    for i, j in zip(*valid_indices):
                        node = nodes[i]
                        if LeadingOnes(node) > l:
                            P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
                        elif LeadingOnes(node) == l and OneMax(node) > OneMax(current_nodes[j]):
                            P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
    
    P[l, m] = 1 - np.sum(P)
    
    if P[l, m] != 1:
        E_current = round((1 + np.sum(P * T)) / (1 - P[l, m]), 3)
    else:
        E_current = 1
    
    return E_current

def variables_calculator(n, pool):
    K = np.zeros((n+1, n+1))
    T = np.zeros((n+1, n+1))
    in_prob = np.zeros((n+1, n+1))

    couples = categorize_bit_strings(n)

    c = len(couples) - 2

    current_couple = list(couples.keys())[c]

    K[(n-1, n-1)] = 1
    T[(n-1, n-1)] = n

    in_prob[(n, n)] = 1 / 2**n
    in_prob[(n-1, n-1)] = 1 / 2**n

    while c != 0:
        c -= 1
        current_couple = list(couples.keys())[c]

        l = current_couple[0]
        m = current_couple[1]

        if current_couple == (0, 0):
            break

        num_couples = len(couples[(l, m)])
        in_prob[current_couple] = num_couples / 2**n

        args_list = [(k, l, m, n, couples, num_couples, in_prob, T) for k in range(1, n - l + 1)]

        E_couple = pool.map(k_loop, args_list)

        E_opt = np.min(E_couple)
        k_opt = np.argmin(E_couple) + 1

        K[current_couple] = k_opt
        T[current_couple] = E_opt

    K[(0, 0)] = n
    T[(0, 0)] = 1
    in_prob[(0, 0)] = 1 / 2**n

    Expected_time = 1 + (in_prob * T).sum()

    return K, T, Expected_time

def plot_2d_matrix(matrix_data, sav_dir, n):
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()

    lower_tri_mask = np.tril(np.ones_like(matrix_data, dtype=bool), k=-1)
    last_col_mask = np.zeros_like(matrix_data, dtype=bool)
    last_col_mask[:, -1] = True
    last_row_mask = np.zeros_like(matrix_data, dtype=bool)
    last_row_mask[-1, :] = True

    combined_mask = lower_tri_mask | last_col_mask | last_row_mask

    matrix_data_masked = np.where(combined_mask, np.nan, matrix_data)
    matrix_data_masked = matrix_data_masked[:-1, :-1]

    c = ax.imshow(matrix_data_masked, cmap='viridis', interpolation='nearest')
    combined_mask = combined_mask[:-1, :-1]
    ax.imshow(combined_mask, cmap='gray', interpolation='nearest', alpha=0.3)

    fig.colorbar(c, ax=ax)

    ax.set_xlabel('OneMax fitness')
    ax.set_ylabel('LeadingOnes fitness')
    ax.set_title(f'Values of k - n = {n}')

    ax.set_xticks(np.arange(matrix_data_masked.shape[1]))
    ax.set_yticks(np.arange(matrix_data_masked.shape[0]))
    ax.set_xticklabels(np.arange(0, matrix_data_masked.shape[1]))
    ax.set_yticklabels(np.arange(0, matrix_data_masked.shape[0]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(matrix_data_masked.shape[0]):
        for j in range(matrix_data_masked.shape[1]):
            value = matrix_data_masked[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='black')

    if sav_dir == "K":
        plt.savefig(os.path.join(curr_dir, 'K_plots', f'{n}.png'), format='png')
    elif sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'T_plots', f'{n}.png'), format='png')

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

    plot_2d_matrix(K, "K", n)
    plot_2d_matrix(T, "T", n)

if __name__ == "__main__":
    core_num = 4  # Adjust the number of cores based on your system
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(1, 12):
            process_iteration(n, pool)
