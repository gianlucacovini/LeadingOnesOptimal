import numpy as np
import os
import matplotlib.pyplot as plt

from heuristic_policy import K_calculator

curr_dir = os.getcwd()

def plot_2d_matrix(matrix_data, sav_dir, n):
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
    
    
    if n <= 40:
        for i in range(matrix_data_masked.shape[0]):
            for j in range(matrix_data_masked.shape[1]):
                value = matrix_data_masked[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.0f}', ha='center', va='center', color='black')

    if sav_dir == "K_heuristic":
        plt.savefig(os.path.join(curr_dir, 'K_plots_heuristic', f'{n}.png'), format='png')
        plt.close(fig)
        
        
if __name__ == "__main__":
    for n in range(2, 101):
        print(f"n = {n}")
        K = K_calculator(n)
        plot_2d_matrix(K, "K_heuristic", n)