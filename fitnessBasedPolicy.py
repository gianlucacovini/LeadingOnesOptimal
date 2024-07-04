import numpy as np
import math
import itertools
import time
import matplotlib.pyplot as plt
import os
from functools import lru_cache

curr_dir = os.getcwd()

def OptimalPolicyFitness(i):
    return math.floor(n/(i+1))
    
# Compose the list of nodes: idea is that we generate them all and then order them

def generate_bit_strings(n):
    return [tuple(map(int, bits)) for bits in itertools.product('01', repeat=n)]

@lru_cache(maxsize=None)
def LeadingOnes(x):
    return max(np.arange(1, len(x) + 1) * (np.cumprod(x) == 1))

@lru_cache(maxsize=None)
def OneMax(x):
    return sum(x)

def sort_bit_strings(bit_strings):
    # Sort the bit strings based on the lexicographic order
    return sorted(bit_strings, key=lambda x: (LeadingOnes(x), OneMax(x)), reverse=False)

def categorize_bit_strings(n):
    bit_strings = generate_bit_strings(n)
    results_dict = {}

    for bits in bit_strings:
        lo = LeadingOnes(bits)
        key = lo

        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(bits)

    return results_dict

def variables_calculator(n):
    K = np.zeros(n+1)
    T = {}
    
    nodes = sort_bit_strings(generate_bit_strings(n))
        
    c = len(nodes)-2 # number of nodes
    
    K[n] = 0
    T[tuple(map(int, np.ones(n)))] = 0
    
    current_node = nodes[c] # It starts with the node with all ones and one zero at the end
                              
    K[n-1] = 1 # optimal k for (n-1, n-1)
    T[current_node] = n
    
    while c != 0: # Verifies that the current node is not the all 0 string
    
        # update node
        c -= 1
        current_node = nodes[c]
    
        m = OneMax(current_node)
        l = LeadingOnes(current_node)
    
        if not np.any(current_node):
            break
    
        k = OptimalPolicyFitness(l)
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
        
        K[l] = k
        T[current_node] = E_current
        
    K[0] = n
    T[tuple(map(int, np.zeros(n)))] = 1
    
    # Calculate the total expected time for the algorithm
    Expected_time = sum([T[node]/(2**n) for node in T.keys()]) # We don't have the +1 because we are summing over all n, not until n-1
    
    return K, T, Expected_time

def E_curr_calc(args):
    k, l, n, values, num_values, in_prob, T = args

    P = np.zeros(n+1)
    current_nodes = np.array(values[l])
    
    for lambda_ in range(0, n-l+1):
        if (l+lambda_) in values:
            nodes = np.array(values[l+lambda_])
            distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
            valid_indices = np.where(distances == k)
            
            if len(valid_indices[0]) > 0:
                for i, j in zip(*valid_indices):
                    node = nodes[i]
                    if LeadingOnes(tuple(node)) > l:
                        P[l+lambda_] += 1 / (math.comb(n, k) * num_values)
                    elif LeadingOnes(tuple(node)) == l and OneMax(tuple(node)) > OneMax(tuple(current_nodes[j])):
                        P[l+lambda_] += 1 / (math.comb(n, k) * num_values)
    
    P[l] = P[l] + (1 - np.sum(P))
    
    if P[l] != 1:
        E_current = round((1 + np.sum(P * T)) / (1 - P[l]), 3)
    else:
        E_current = 1
    
    return E_current

# def variables_calculator(n):
#     K = np.zeros(n+1)
#     T = np.zeros(n+1)
#     in_prob = np.zeros(n+1)
    
#     values = categorize_bit_strings(n)
        
#     c = len(values)-2 # number of nodes
    
#     K[n] = 0
#     T[n] = 0
                                  
#     K[n-1] = 1 # optimal k for n-1
#     T[n-1] = n
    
#     in_prob[n] = 1 / 2**n
#     in_prob[n-1] = 1 / 2**n
    
#     current_value = values[c] # It starts with the node with all ones and one zero at the end
    
#     while c != 0:
#         c -= 1
#         current_value = list(values.keys())[c]

#         l = current_value

#         if current_value == 0:
#             break

#         num_values = len(values[l])
#         in_prob[current_value] = num_values / 2**n
        
#         k = OptimalPolicyFitness(l)
        
#         args = (k, l, n, values, num_values, in_prob, T)

#         E_opt = E_curr_calc(args)

#         K[current_value] = k
#         T[current_value] = E_opt

#     K[0] = n
#     T[0] = 1
#     in_prob[0] = 1 / 2**n

#     Expected_time = 1 + (in_prob * T).sum()

#     return K, T, Expected_time

def variables_calculator_theory(n):
    K = np.zeros(n)
    for i in range(0, n):
        K[i] = OptimalPolicyFitness(i)
        
    K = K.astype(int)
        
    P = np.zeros(n)
    for i in range(0, n):
        P[i] = math.comb(n-i-1, n-i-K[i])/math.comb(n, n-K[i])
    
    # K[n-1] = 1
    # P[n-1] = 1/n
    
    f = lambda x: 1/x
    inv_P = f(P)
    Expected_time = 1/2 * np.sum(inv_P)
    
    return K, Expected_time


def plot_2d_array(array_data, sav_dir=None):
    matrix_data = np.transpose(np.asmatrix(array_data[:-1]))
    
    fig, ax = plt.figure(), plt.gca()
    
    # Plotting the matrix with color encoded values
    c = ax.imshow(matrix_data, cmap='viridis', interpolation='nearest')
    fig.colorbar(c, ax=ax)  # Add a colorbar to a plot

    ax.set_ylabel('LeadingOnes fitness')
    
    if sav_dir == "K":
        ax.set_title('Color Map of values of k')
    if sav_dir == "T":
        if sav_dir == "K":
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
        plt.savefig(os.path.join(curr_dir, 'plots', 'K_plots', 'fitness_based_greedy', f'{n}_fitness.png'), format='png')
    if sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'plots', 'T_plots', 'fitness_based_greedy', f'{n}_fitness.png'), format='png')

    plt.show()

if __name__ == "__main__":
    
    for n in range(2, 102):
    # TBP
        start_time = time.time()

        K, T, Expected_time = variables_calculator(n)
        # K, Expected_time = variables_calculator_theory(n)
        
        end_time = time.time()  # Capture end time
        
        print(f"Expected time for n = {n}: ", round(Expected_time, 3))
        
        print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds") # It is about 5 times higher for each n added
        
        with open('results_LO_noGreedy.txt', 'w') as file:
            file.write("Policy (LO(x), OM(x))\n")
            file.write(f"n: {n}\n")
            file.write(f"Expected time: {Expected_time}\n")
        #     file.write(f"K: {K}\n")
        #     file.write(f"T: {T}\n")
        
        plot_2d_array(K, "K")
        # plot_2d_array(T, "T")