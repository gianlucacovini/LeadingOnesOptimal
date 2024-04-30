import numpy as np
import math
import itertools
import time

start_time = time.time() 

# Dimension of the problem
n = 10
    
K = {} # Creating an empty dict
T = {}

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

nodes = sort_bit_strings(generate_bit_strings(n)) # Write just one function
    
c = len(nodes)-2 # number of nodes

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

    if not np.any(current_node):
        break

    E_opt = np.Inf
    P = {}
    for k in range(1, n-m+1):
        for node in K.keys(): # We look only between the keys we have already looked, since the lexicographic improvement is possible only towards them
            if sum(list(abs(a - b) for a, b in zip(node, current_node))) == k: # We consider only the strings where we changed exactly k elements 
                if sum(node) > sum(current_node): 
                    P[node] = 1/math.comb(n, k)
            
        P_current_node = 1 - sum(P.values()) # Is it equal to 1 - all the other probabilities??
        
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

print(f"Expected time for n = {n}: ", round(Expected_time, 3))

end_time = time.time()  # Capture end time
print(f"Execution Time for n = {n}: {round(end_time - start_time, 3)} seconds") # It is about 5 times higher for each n added
