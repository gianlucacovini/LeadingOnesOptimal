import numpy as np
import itertools
import time
from scipy.special import comb
import math
from functools import lru_cache
from multiprocessing import Pool

def generate_bit_strings(n):
    """ Generate all bit strings of length n. """
    return [tuple(map(int, bits)) for bits in itertools.product('01', repeat=n)]

@lru_cache(maxsize=None)
def LeadingOnes(x):
    """ Return the count of leading ones in a bit string. """
    return max(np.arange(1, len(x) + 1) * (np.cumprod(x) == 1))

@lru_cache(maxsize=None)
def OneMax(x):
    """ Return the sum of the bit string. """
    return sum(x)

def sort_bit_strings(bit_strings):
    """ Sort bit strings by leading ones and one max lexicographic criteria. """
    return sorted(bit_strings, key=lambda x: (LeadingOnes(x), OneMax(x)), reverse=False)

def calculate_for_k(args):
    k, current_node, K, T, n, l, m = args
    P = {}
    for node in K.keys(): # We look only between the keys we have already looked, since the lexicographic improvement is possible only towards them
        if sum(list(abs(a - b) for a, b in zip(node, current_node))) == k: # We consider only the strings where we changed exactly k bits
            if LeadingOnes(node) > l or (LeadingOnes(node) == l and OneMax(node) > m): 
                P[node] = 1 / comb(n, k)
    P_current_node = 1 - sum(P.values())
    # Calculated expected time with given k
    E_current = round((1 + sum([P[node] * T[node] for node in P.keys()])) / (1 - P_current_node), 3)
    return E_current, k

def calculate_expected_time(n):
    """ Calculate the expected time for a given n. """
    K = {} # Creating an empty dict
    T = {}
    
    nodes = sort_bit_strings(generate_bit_strings(n)) # Write just one function
    c = len(nodes) - 2 # number of nodes
    
    K[tuple(map(int, np.ones(n)))] = 0
    T[tuple(map(int, np.ones(n)))] = 0
    
    current_node = nodes[c] # It starts with the node with all ones and one zero at the end
    K[current_node] = 1 # optimal k for (n-1, n-1)
    T[current_node] = n
    
    while c != 0: # Verifies that the current node is not the all 0 string
        # update node
        c -= 1
        current_node = nodes[c]
        
        m = OneMax(current_node)
        l = LeadingOnes(current_node)
        
        if not np.any(current_node):
            break
        
        E_opt = np.inf
        
        args_list = [(k, current_node, K, T, n, l, m) for k in range(1, n - l + 1)]
        
        with Pool() as pool:
            results = pool.map(calculate_for_k, args_list)
        
        for E_current, k in results:
            if E_current < E_opt:
                E_opt = E_current
                k_opt = k
        
        K[current_node] = k_opt
        T[current_node] = E_opt
    
    K[tuple(map(int, np.zeros(n)))] = n
    T[tuple(map(int, np.zeros(n)))] = 1
    
    # Calculate the total expected time for the algorithm
    Expected_time = sum([T[node] / (2 ** n) for node in T.keys()])
    
    return Expected_time, K, T

def main(n):
    start_time = time.time()
    Expected_time, K, T = calculate_expected_time(n)
    end_time = time.time()  # Capture end time
    
    print(f"Expected time for n = {n}: ", round(Expected_time, 3))
    print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds") # It is about 5 times higher for each n added
    
    with open('results.txt', 'w') as file:
        file.write("Policy x\n")
        file.write(f"n: {n}\n")
        file.write(f"Expected time: {Expected_time}\n")
        file.write(f"K: {K}\n")
        file.write(f"T: {T}\n")

if __name__ == "__main__":
    n = 2
    main(n)