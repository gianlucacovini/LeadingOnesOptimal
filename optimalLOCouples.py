import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import multiprocessing
from functools import lru_cache
import time
from itertools import combinations

"""
Da correggere per fare i fixed portfolio: il massimo che mette all'inizio non può metterlo se non è
nel portfolio. Poi la storia del portfolio non funziona ancora...
"""

core_num = 1
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

def mu_loop(args):
    k, l, m, n, couples, num_couples, P, current_nodes, lambda_, mu = args
    
    if (l+lambda_, m+mu) in couples:
        nodes = np.array(couples[(l+lambda_, m+mu)])
        distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
        valid_indices = np.where(distances == k)
        
        if len(valid_indices[0]) > 0:
            for i, j in zip(*valid_indices):
                node = nodes[i]
                if LeadingOnes(tuple(node)) > l:
                    P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
                elif LeadingOnes(tuple(node)) == l and OneMax(tuple(node)) > OneMax(tuple(current_nodes[j])):
                    P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
                    
    return P

def k_loop(args):
    k, l, m, n, couples, num_couples, T = args

    P = np.zeros((n+1, n+1))
    current_nodes = np.array(couples[(l, m)])
    
    for lambda_ in range(0, n-l+1):
        #args_list = [(k, l, m, n, couples, num_couples, P, current_nodes, lambda_, mu) for mu in  range(-k+1, n-l+1)]
        
        for mu in  range(-k+1, n-l+1):
            if (l+lambda_, m+mu) in couples:
                nodes = np.array(couples[(l+lambda_, m+mu)])
                distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
                valid_indices = np.where(distances == k)
                
                if len(valid_indices[0]) > 0:
                    for i, j in zip(*valid_indices):
                        node = nodes[i]
                        if LeadingOnes(tuple(node)) > l:
                            P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
                        elif LeadingOnes(tuple(node)) == l and OneMax(tuple(node)) > OneMax(tuple(current_nodes[j])):
                            P[l+lambda_, m+mu] += 1 / (math.comb(n, k) * num_couples)
        
        #P = pool.map(mu_loop, args_list)
    
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
        
        #with multiprocessing.Pool(processes=core_num) as pool:
        args_list = [(k, l, m, n, couples, num_couples, T) for k in range(1, n - l + 1)] # CONTROLLARE QUESTO RANGE. NON è min(n, n-l+2)

        E_couple = pool.map(k_loop, args_list)

        E_opt = np.min(E_couple)
        k_opt = np.argmin(E_couple) + 1 # +1 because radius start from 1 but indeces start from 0

        K[current_couple] = k_opt
        T[current_couple] = E_opt

    K[(0, 0)] = n
    T[(0, 0)] = 1
    in_prob[(0, 0)] = 1 / 2**n

    Expected_time = 1 + (in_prob * T).sum()

    return K, T, Expected_time

def plot_2d_matrix(matrix_data, n, sav_dir=None):
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

    plot_2d_matrix(K, n, "K")
    plot_2d_matrix(T, n, "T")
    
    return Expected_time
    
def variables_calculator_fulltime(n, K):
    T = {}
    
    nodes = sort_bit_strings(generate_bit_strings(n))
        
    c = len(nodes)-2 # number of nodes
    
    T[tuple(map(int, np.ones(n)))] = 0
    
    current_node = nodes[c] # It starts with the node with all ones and one zero at the end
                              
    T[current_node] = n
    
    while c != 0: # Verifies that the current node is not the all 0 string
    
        # update node
        c -= 1
        current_node = nodes[c]
    
        m = OneMax(current_node)
        l = LeadingOnes(current_node)
    
        if not np.any(current_node):
            break
    
        k = K[(l, m)]
        P = {}
        for node in T.keys(): # We look only between the keys we have already looked, since the lexicographic improvement is possible only towards them
            if sum(list(abs(a - b) for a, b in zip(node, current_node))) == k: # We consider only the strings where we changed exactly k elements 
                if LeadingOnes(node) > l:
                    P[node] = 1/math.comb(n, k)
                elif LeadingOnes(node) == l:
                    if sum(node) > sum(current_node): 
                        P[node] = 1/math.comb(n, k)
            
        P_current_node = 1 - sum(P.values())
        
        # Calculated expected time with given k
        if P_current_node != 1:
            E_current = round((1 + sum([P[node]*T[node] for node in P.keys()]))/(1-P_current_node), 3)
        else:
            E_current = 1
        
        T[current_node] = E_current
        
    T[tuple(map(int, np.zeros(n)))] = 1
    
    # Calculate the total expected time for the algorithm
    Expected_time = sum([T[node]/(2**n) for node in T.keys()]) # We don't have the +1 because we are summing over all n, not until n-1
    
    print(f"n = {n}, Expected time = {Expected_time} \n")
    
    return T, Expected_time

if __name__ == "__main__":
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(1, 14):
            process_iteration(n, pool)
