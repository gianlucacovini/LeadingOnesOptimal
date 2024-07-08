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
    l, n, couples, T, k = args
    
    A = np.zeros((n-l, n-l))
    
    b = np.ones(n-l) 
    
    for starting_m in range(l, n): # taking values from l to n-1
    
        if (l, starting_m) in couples:
            current_nodes = np.array(couples[(l, starting_m)])
            num_couples = len(current_nodes)
            
            in_prob = num_couples / 2**n
        
            for arriving_m in range(l, n): # taking values from l to n-1
                if (l, arriving_m) in couples:
                    nodes = np.array(couples[(l, arriving_m)])
                    distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
                    valid_indices = np.where(distances == k)
                    
                    if len(valid_indices[0]) > 0:
                        for i, j in zip(*valid_indices):
                            node = nodes[i]
                            if LeadingOnes(tuple(node)) >= l:
                                # ocio ai meno
                                A[starting_m - l, arriving_m - l] -= 1 / (math.comb(n, k) * num_couples) # non sono sicuro di questa formula ma mi fido del me stesso del passato
            
            A[starting_m - l, starting_m - l] = -np.sum(A[starting_m - l, :]) - 2*A[starting_m - l, starting_m - l]
                                                            # Poi attenzione alla vecchia storia del fatto che può stare nella coppia ma esserci transizione di stato e quindi probabilità non nulla: è il motivo di questo -2*
                                                            
            P = np.zeros((n+1, n+1))
        
            for lambda_ in range(l+1, n+1):
                
                for mu in  range(lambda_, n+1):
                    if (lambda_, mu) in couples:
                        nodes = np.array(couples[(lambda_, mu)])
                        distances = np.sum(np.abs(nodes[:, None, :] - current_nodes[None, :, :]), axis=2)
                        valid_indices = np.where(distances == k)
                        
                        if len(valid_indices[0]) > 0:
                            for i, j in zip(*valid_indices):
                                node = nodes[i]
                                if LeadingOnes(tuple(node)) >= l:
                                    P[lambda_, mu] += 1 / (math.comb(n, k) * num_couples)
                        
                        b[starting_m - l] += P[lambda_, mu] * T[lambda_, mu]
                        
                        A[starting_m - l, starting_m - l] += P[lambda_, mu]
                                        
    return A, b
        
def E_calculator(args):
    l, n, couples, T = args
    
    A_dict = {}
    b_dict = {}
    for k in range(1, n-l+1):
        args_terms = l, n, couples, T, k
        A, b = terms_calculator(args_terms)
        A_dict[k] = A
        b_dict[k] = b
        
        
    # Ora che ho tutte le righe per A e b mi faccio un po' di cicli mappazzone e trovo 
    # tutte le x e tengo la migliore
    
    # funzione combinatoria che mi calcoli tutti le combinazioni di k per l fissato
    
        # funzione che mi costruisce le A e b corrispondenti a una combinazione di k
        
        # calcolo della x
        
        # classico: se è minore dell'ottimo, salva se no passa.
        # OSS: visto che l'ottimo va calcolato come media meglio tenersi a mente anche num_couples in modo da non doverlo calcolare ogni volta
        # ma alla fine va bene anche così
        
    # Generate all possible combinations of n-l sequences of values of k
    combinations = list(itertools.product(range(1, n-l+1), repeat=n-l))
    
    x_opt_mean = np.inf
    for comb in combinations:
        # Build the corresponding A matrix and b vector
        A = np.vstack([A_dict[k][i] for i, k in enumerate(comb)])
        b = np.hstack([b_dict[k][i] for i, k in enumerate(comb)])

        try:
            # Solve the linear system Ax = b
            x = np.linalg.solve(A, b)
            
            x_mean = np.mean(x) # DA CORREGGERE: media pesata sul numero di couples
            
            if x_mean < x_opt_mean:
                k_opt = comb
                x_opt = x
                x_opt_mean = x_mean
            
        except np.linalg.LinAlgError:
            # If the matrix A is singular, we can't solve it
            x = None
        
    return x_opt, k_opt

def variables_calculator(n, pool):
    K = np.zeros((n+1, n+1))
    T = np.zeros((n+1, n+1))
    in_prob = np.zeros((n+1, n+1))

    couples = categorize_bit_strings(n)

    num_couples = len(couples) # è giusto??

    K[(n-1, n-1)] = 1
    T[(n-1, n-1)] = n

    in_prob[(n, n)] = 1 / 2**n
    in_prob[(n-1, n-1)] = 1 / 2**n

    for l in reversed(range(0, n-1)):
        args = l, n, couples, T
        
        x_opt, k_opt = E_calculator(args) # it gives a vector with expected times for 
            
        # ok, ora aggiorniamo K e T e il gioco è quasi fatto
        K[l, l:n] = k_opt
        T[l, l:n] = x_opt
            
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

    plot_2d_matrix(K, n)
    plot_2d_matrix(T, n)
    
    return Expected_time

if __name__ == "__main__":
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(3, 11):
            process_iteration(n, pool)
