import numpy as np
import math
import itertools
import time
import matplotlib.pyplot as plt
import os
from functools import lru_cache

curr_dir = os.getcwd()

def OptimalPolicyFitness(i, n):
    return math.floor(n/(i+1))

def variables_calculator(n):
    K = np.zeros(n+1)
    T = np.zeros(n+1)
                
    K[n] = 0
    T[n] = 0
                                  
    K[n-1] = 1 # optimal k for (n-1, n-1)
    T[n-1] = n
    
    
    
    # Calculate the total expected time for the algorithm
    Expected_time = 1 + sum([T[node]/(2**n) for node in T.keys()])
    
    return K, T, Expected_time

def plot_2d_array(array_data, sav_dir=None):
    matrix_data = np.transpose(np.asmatrix(array_data[:-1]))
    
    fig, ax = plt.figure(), plt.gca()
    
    # Plotting the matrix with color encoded values
    c = ax.imshow(matrix_data, cmap='viridis', interpolation='nearest')
    fig.colorbar(c, ax=ax)  # Add a colorbar to a plot

    ax.set_ylabel('LeadingOnes fitness')
    ax.set_title('Color Map of values of k')

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
        plt.savefig(os.path.join(curr_dir, 'K_plots', 'fitness_based', f'{n}_fitness.png'), format='png')
    if sav_dir == "T":
        plt.savefig(os.path.join(curr_dir, 'T_plots', 'fitness_based', f'{n}_fitness.png'), format='png')

    plt.show()