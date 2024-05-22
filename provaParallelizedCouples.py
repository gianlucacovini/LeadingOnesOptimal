import numpy as np
import math
import itertools
import multiprocessing

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

def generate_bit_strings(n):
    return [tuple(map(int, bits)) for bits in itertools.product('01', repeat=n)]

def compute_inner_loops(current_node, l, m, n, k, couples, num_couples):
    P = {key: 0 for key in couples}
    for lambda_ in range(0, n - l + 1):
        for mu in range(-k + 1, n - l + 1):
            if (l + lambda_, m + mu) in couples:
                for node in couples[(l + lambda_, m + mu)]:
                    if sum(abs(a - b) for a, b in zip(node, current_node)) == k:
                        if LeadingOnes(node) == l + lambda_ and OneMax(node) == m + mu:
                            key = (l + lambda_, m + mu)
                            if LeadingOnes(node) > l:
                                P[key] += 1 / (math.comb(n, k) * num_couples)
                            elif LeadingOnes(node) == l:
                                if sum(node) > sum(current_node):
                                    P[key] += 1 / (math.comb(n, k) * num_couples)
    return P

def compute_for_k(args):
    k, l, m, n, couples, num_couples, in_prob = args
    P = {key: 0 for key in couples}
    for current_node in couples[(l, m)]:
        partial_P = compute_inner_loops(current_node, l, m, n, k, couples, num_couples)
        for key, value in partial_P.items():
            P[key] += value
    E_value = sum(P.values())
    return k, E_value, E_value * in_prob[(l, m)]

def variables_calculator(n):
    K = np.zeros((n + 1, n + 1))  # Creating an empty DP matrix
    T = np.zeros((n + 1, n + 1))
    in_prob = np.zeros((n + 1, n + 1))

    couples = categorize_bit_strings(n)
    c = len(couples) - 2  # number of nodes
    current_couple = list(couples.keys())[c]  # It starts with the node with all ones and one zero at the end

    K[(n - 1, n - 1)] = 1  # optimal k for (n - 1, n - 1)
    T[(n - 1, n - 1)] = n

    in_prob[(n, n)] = 1 / 2**n
    in_prob[(n - 1, n - 1)] = 1 / 2**n

    while c != 0:  # Verifies that the current node is not the all 0 string
        # Update node
        c -= 1
        current_couple = list(couples.keys())[c]

        l = current_couple[0]
        m = current_couple[1]

        if current_couple == (0, 0):
            break

        num_couples = len(couples[(l, m)])
        in_prob[current_couple] = num_couples / 2**n

        E_opt = np.inf

        # Prepare arguments for parallel execution
        args_list = [(k, l, m, n, couples, num_couples, in_prob) for k in range(1, n - l + 1)]

        # Parallelize the loop over k using pool.map
        with multiprocessing.Pool() as pool:
            results = pool.map(compute_for_k, args_list)
        
        # Process results to find the minimum E_value and corresponding k
        for k, E_value, T_value in results:
            if E_value < E_opt:
                E_opt = E_value
                K[(l, m)] = k
                T[(l, m)] = T_value

    Expected_time = T.sum()
    return K, T, Expected_time

if __name__ == "__main__":
    n = 4  # Set the appropriate value of n
    K, T, Expected_time = variables_calculator(n)
    print(f"K: {K}")
    print(f"T: {T}")
    print(f"Expected_time: {Expected_time}")
