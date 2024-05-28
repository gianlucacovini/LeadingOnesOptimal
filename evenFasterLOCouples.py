import numpy as np
import itertools
import multiprocessing
import os
import math
import time
import matplotlib.pyplot as plt

def generate_bit_strings(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))

def LeadingOnes(x):
    return (np.cumprod(x) == 1).sum()

def OneMax(x):
    return np.sum(x)

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

def k_loop(args):
    k, l, m, n, couples, num_couples, in_prob, T = args
    
    P = {key: 0 for key in couples}
    for current_node in couples[(l, m)]:
        for lambda_ in range(0, n-l+1):
            for mu in range(-k+1, n-l+1):
                if (l+lambda_, m+mu) in couples:
                    for node in couples[(l+lambda_, m+mu)]:
                        if np.sum(np.abs(np.array(node) - np.array(current_node))) == k:
                            if LeadingOnes(node) > l:
                                P[(l+lambda_, m+mu)] += 1 / (math.comb(n, k) * num_couples)
                            elif LeadingOnes(node) == l:
                                if OneMax(node) > OneMax(current_node):
                                    P[(l+lambda_, m+mu)] += 1 / (math.comb(n, k) * num_couples)
    
    P[(l, m)] = 1 - sum(P.values())
    
    if P[(l, m)] != 1:
        E_current = round((1 + sum([P[couple] * T[(couple[0], couple[1])] for couple in P.keys()])) / (1 - P[(l, m)]), 3)
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

def main():
    core_num = 1  # Adjust the number of cores based on your system
    with multiprocessing.Pool(processes=core_num) as pool:
        for n in range(1, 11):
            process_iteration(n, pool)

if __name__ == "__main__":
    main()
