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
    return [tuple(map(int, bits)) for bits in itertools.product('01', repeat=n)]

def LeadingOnes(x):
    return max(np.arange(1, len(x) + 1) * (np.cumprod(x) == 1), default=0)

def OneMax(x):
    return sum(x)

def categorize_bit_strings(n):
    bit_strings = generate_bit_strings(n)
    results_dict = {}

    for bits in bit_strings:
        lo = LeadingOnes(bits)
        om = OneMax(bits)
        key = (lo, om)

        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(bits)

    return results_dict


def sort_bit_strings(bit_strings):
    # Sort the bit strings based on the lexicographic order
    return sorted(bit_strings, key=lambda x: (LeadingOnes(x), OneMax(x)), reverse=False)



def k_loop(args):
    k, l, m, n, couples, num_couples, in_prob, T = args
    
    # gives the value of expected time for given k. Da portare fuori
    P = {key: 0 for key in couples}
    for current_node in couples[(l, m)]:
        for lambda_ in range(0, n-l+1): # min(k, n-l)):
            for mu in range(-k+1, n-l+1): # min(k, n-l)):
                if (l+lambda_, m+mu) in couples:
                    for node in couples[(l+lambda_, m+mu)]:
                        
                        if sum(list(abs(a - b) for a, b in zip(node, current_node))) == k: # We consider only the strings where we changed exactly k elements
                            if LeadingOnes(node) > l:
                                P[(l+lambda_, m+mu)] += 1/(math.comb(n, k)*num_couples)
                            elif LeadingOnes(node) == l:
                                if sum(node) > sum(current_node): 
                                    P[(l+lambda_, m+mu)] += 1/(math.comb(n, k)*num_couples)

    P[(l, m)] = 1 - sum(P.values())

    # Calculated expected time with given k
    if P[(l, m)] != 1:
        E_current = round((1 + sum([P[couple]*T[(couple[0], couple[1])] for couple in P.keys()]))/(1-P[(l, m)]), 3) # Sostiutire il prodotto con un prodotto numpy
    else:
        E_current = 1 # Non sono del tutto sicuro di questo
        
    return E_current

def variables_calculator(n):
    K = np.zeros((n+1, n+1)) # Creating an empty DP matrix
    T = np.zeros((n+1, n+1))
    in_prob = np.zeros((n+1, n+1))

    couples = categorize_bit_strings(n)
        
    c = len(couples)-2 # number of nodes
    
    current_couple = list(couples.keys())[c] # It starts with the node with all ones and one zero at the end
                              
    K[(n-1, n-1)] = 1 # optimal k for (n-1, n-1)
    T[(n-1, n-1)] = n
    
    in_prob[(n, n)] = 1/2**n
    in_prob[(n-1, n-1)] = 1/2**n
    
    while c != 0: # Verifies that the current node is not the all 0 string
    
        # update node
        c -= 1
        current_couple = list(couples.keys())[c]
    
        l = current_couple[0]
        m = current_couple[1]
    
        if current_couple == (0, 0):
            break
        
        num_couples = len(couples[(l, m)])
        in_prob[current_couple] = num_couples/2**n
        
        args_list = [(k, l, m, n, couples, num_couples, in_prob, T) for k in range(1, n - l + 1)]
        
        with multiprocessing.Pool(processes=core_num) as pool:
            E_couple = pool.map(k_loop, args_list)
        
        # Ora k_loop va applicata al range e deve restituire una sorta di vettore 
        # con tutti gli E_current in modo da prendere il valore minimo come E_opt e 
        # l'indice corrispondente come k_opt
            
        # Il codice sotto va rimpiazzato appunto prendendo il minimo
        E_opt = np.min(E_couple)
        k_opt = np.argmin(E_couple) + 1
        
        K[current_couple] = k_opt
        T[current_couple] = E_opt
        
    K[(0, 0)] = n
    T[(0, 0)] = 1
    in_prob[(0,0)] = 1/2**n

    # Calculate the total expected time for the algorithm
    Expected_time = 1 + (in_prob*T).sum() # Devo moltiplicare ogni entrata per il numero di 
    
    return K, T, Expected_time

def plot_3d_matrix(matrix_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create data
    x_data = np.arange(matrix_data.shape[0])
    y_data = np.arange(matrix_data.shape[1])
    x_data, y_data = np.meshgrid(y_data, x_data)
    z_data = np.array(matrix_data)
    
    # Plotting the surface plot
    ax.plot_surface(x_data, y_data, z_data, cmap='viridis')
    
    ax.set_xlabel('OneMax Index')
    ax.set_ylabel('LeadingOnes Index')
    ax.set_zlabel('k value')
    
    # Showing the plot
    plt.show()

def plot_2d_matrix(matrix_data, sav_dir, n):
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    
    # Create a mask for the strictly lower triangular part
    lower_tri_mask = np.tril(np.ones_like(matrix_data, dtype=bool), k=-1)
    
    # Create a mask for the entire last column
    last_col_mask = np.zeros_like(matrix_data, dtype=bool)
    last_col_mask[:, -1] = True

    # Create a mask for the entire last row
    last_row_mask = np.zeros_like(matrix_data, dtype=bool)
    last_row_mask[-1, :] = True

    # Combine all masks
    combined_mask = lower_tri_mask | last_col_mask | last_row_mask

    # Replace masked elements with NaN
    matrix_data_masked = np.where(combined_mask, np.nan, matrix_data)
    
    # Exclude the last row and last column from plotting
    matrix_data_masked = matrix_data_masked[:-1, :-1]

    # Plotting the matrix with color encoded values
    c = ax.imshow(matrix_data_masked, cmap='viridis', interpolation='nearest')

    # Create a gray background for masked values
    combined_mask = combined_mask[:-1, :-1]
    ax.imshow(combined_mask, cmap='gray', interpolation='nearest', alpha=0.3)

    fig.colorbar(c, ax=ax)  # Add a colorbar to the plot

    ax.set_xlabel('OneMax fitness')
    ax.set_ylabel('LeadingOnes fitness')
    ax.set_title(f'Values of k - n = {n}')

    # Tick labels for each row and column, excluding the last row and column
    ax.set_xticks(np.arange(matrix_data_masked.shape[1]))
    ax.set_yticks(np.arange(matrix_data_masked.shape[0]))
    ax.set_xticklabels(np.arange(1, matrix_data_masked.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, matrix_data_masked.shape[0] + 1))

    # Rotate the tick labels for better display
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate each cell with the corresponding value
    for i in range(matrix_data_masked.shape[0]):
        for j in range(matrix_data_masked.shape[1]):
            value = matrix_data_masked[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black')

    # Save the plot in the specified directory
    if sav_dir == "K":
        plt.savefig(os.path.join(curr_dir, 'K_plots', f'{n}.png'), format='png')
    elif sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'T_plots', f'{n}.png'), format='png')

    # plt.show()
    
def process_iteration(n):
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
    
    plot_2d_matrix(K, "K", n)
    plot_2d_matrix(T, "T", n)
    

if __name__ == "__main__":    
    # with multiprocessing.Pool(processes=core_num) as pool:
    #     # Parallelize the execution of the process_iteration function
    #     pool.map(process_iteration, range(1, 11))
    
    
    for n in range(1, 11):
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
        
        plot_2d_matrix(K, "K", n)
        plot_2d_matrix(T, "T", n)
