import numpy as np
import math
import itertools
import time
import matplotlib.pyplot as plt
import os

curr_dir = os.getcwd()

def OptimalPolicyFitness(i):
    return math.floor(n/(i+1))
    
# Compose the list of nodes: idea is that we generate them all and then order them

def generate_bit_strings(n):
    return [tuple(map(int, bits)) for bits in itertools.product('01', repeat=n)]

def LeadingOnes(x):
    return max(np.arange(1, len(x) + 1) * (np.cumprod(x) == 1))

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

# NON SI PUò  calcolare meglio il tempo atteso in questo caso? Con una formula?

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
    Expected_time = 1 + sum([T[node]/(2**n) for node in T.keys()])
    
    return K, T, Expected_time


def plot_2d_array(array_data, sav_dir=None):
    matrix_data = np.transpose(np.asmatrix(array_data))
    
    fig, ax = plt.figure(), plt.gca()
    
    # Plotting the matrix with color encoded values
    c = ax.imshow(matrix_data, cmap='viridis', interpolation='nearest')
    fig.colorbar(c, ax=ax)  # Add a colorbar to a plot

    ax.set_ylabel('LeadingOnes fitness')
    ax.set_title('Color Map of values of k')

    # Tick labels for each row and column
    ax.set_xticks(np.arange(matrix_data.shape[1]))
    ax.set_yticks(np.arange(matrix_data.shape[0]))
    ax.set_xticklabels(np.arange(1, matrix_data.shape[1]+1))
    ax.set_yticklabels(np.arange(1, matrix_data.shape[0]+1))

    # Rotate the tick labels for better display
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[1]):
            value = matrix_data[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value}', ha='center', va='center', color='black')
    
    if sav_dir == "K":
        plt.savefig(os.path.join(curr_dir, 'K_plots', 'fitness_based', f'{n}_fitness.png'), format='png')
    if sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'T_plots', 'fitness_based', f'{n}_fitness.png'), format='png')

    plt.show()

if __name__ == "__main__":
    
    for n in range(1, 20):
    # TBP
        start_time = time.time()

        K, T, Expected_time = variables_calculator(n)
        
        end_time = time.time()  # Capture end time
        
        print(f"Expected time for n = {n}: ", round(Expected_time, 3))
        
        print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds") # It is about 5 times higher for each n added
        
        with open('results.txt', 'w') as file:
            file.write("Policy (LO(x), OM(x))\n")
            file.write(f"n: {n}\n")
            file.write(f"Expected time: {Expected_time}\n")
            file.write(f"K: {K}\n")
            file.write(f"T: {T}\n")
        
        plot_2d_array(K, "K")