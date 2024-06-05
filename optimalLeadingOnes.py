import numpy as np
import itertools
import time
from scipy.special import comb
import math
from functools import lru_cache

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
        
        for k in range(1, n-l+1): # Is it right n-l+1?
            P = {}
            for node in K.keys(): # We look only between the keys we have already looked, since the lexicographic improvement is possible only towards them
                if sum(list(abs(a - b) for a, b in zip(node, current_node))) == k: # We consider only the strings where we changed exactly k elements 
                    if LeadingOnes(node) > l:
                        P[node] = 1/math.comb(n, k)
                    elif LeadingOnes(node) == l:
                        if sum(node) > sum(current_node): 
                            P[node] = 1/math.comb(n, k)
            
            P_current_node = 1 - sum(P.values())
            
            # Calculated expected time with given k
            E_current = round((1 + sum([P[node]*T[node] for node in P.keys()]))/(1-P_current_node), 3)
            
            if E_current < E_opt:
                E_opt = E_current
                k_opt = k
        
        K[current_node] = k_opt
        T[current_node] = E_opt
    
    K[tuple(map(int, np.zeros(n)))] = n
    T[tuple(map(int, np.zeros(n)))] = 1
    
    # Calculate the total expected time for the algorithm
    Expected_time = 1 + sum([T[node]/(2**n) for node in T.keys()])
    
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
    
    # check_keys(K)

def check_keys(K):
    # Create a dictionary to store (LO, OM) tuples and their corresponding dictionary values
    fitness_dict = {}
    for bits, value in K.items():
        # Calculate the fitnesses
        lo_val = LeadingOnes(bits)
        om_val = OneMax(bits)
        # Check if this (LO, OM) pair already exists with a different value
        if (lo_val, om_val) in fitness_dict:
            if fitness_dict[(lo_val, om_val)] != value:
                print(bits, value)
        else:
            fitness_dict[(lo_val, om_val)] = value
    
    print("done")
    return fitness_dict

if __name__ == "__main__":
    n = 9
    main(n)