import numpy as np
import math
import itertools

def generate_bit_strings(length):
    return np.array(list(itertools.product([0, 1], repeat=length)))

def Expected_increment_calculator(n, l, m, k): # forse il max_i che ci interessa dev'essere sempre n-l
    max_i = n-l

    strings = {}
    P_string = {} # probability single string
    P_flip = {}
    P = {}
    Prob = np.zeros(max_i) # probability exact fitness increment
    
    strings[1] = generate_bit_strings(length = 1)
    
    i = 1 # current value of increment considering
    
    if l < n-1 and k <= n-l: # feasibility conditions
        P_string[(0,)] = math.comb(n-(l+1), m-l)/math.comb(n-(l+2), m-l)
        if k >= 2:
            P_flip[(0,)] = (math.comb(n-l, k)/math.comb(n, k)) * (math.comb(n-l-1, k-1)/math.comb(n-l, k)) * (1-(math.comb(n-l-2, k-2)/math.comb(n-l-1, k-1)))
        else:
            P_flip[(0,)] = 1
            
    else:
        P_string[(0,)] = 0 # Probabilmente P_string e P_flip sono sbagliati ma li correggiamo poi (anche per 1)
        P_flip[(0,)] = 0
        P[(0,)] = 0
        
    P[(0,)] = P_string[(0,)] * P_flip[(0,)]
        
    if k >= 2 and m-l >= 1 and l < n-1 and k<= n-l:
        P_string[(1,)] = math.comb(n-(l+1), m-l)/math.comb(n-(l+2), m-l)
        P_flip[(1,)] = (math.comb(n-l, k)/math.comb(n, k)) * (math.comb(n-l-1, k-1)/math.comb(n-l, k)) * (math.comb(n-l-2, k-2)/math.comb(n-l-1, k-1))
        
    else:
        P_string[(1,)] = 1
        P_flip[(1,)] = 1
        
        P[(1,)] = 1
        
    P[(1,)] = P_string[(1,)] * P_flip[(1,)]
    
    Prob[i-1] = P[(0,)] + P[(1,)] # Probability of increment exactly 1. The -1 is because the increment starts from 1 and the indeces from 0
    
    # In the case 1...10 we already have the probability of improvement of 1 (only possible improvement) which is 1/n
    if l == n-1:
        Prob[i-1] = 1/n
    
    for i in range(2, max_i + 1):
        strings[i] = generate_bit_strings(i) 
        
        for string in strings[i]: # We are considering strings 1...10string
            num_ones = np.sum(string)
            num_zeros = n - num_ones
                
            if string[-1] == 1: # I flip if l+i+1 is 1
                Cond_p_flip = math.comb(n-l-i-1, k-num_zeros-1)/math.comb(n-l-i, k-num_zeros) if num_zeros < k else 0
                Cond_p_string = (m-num_ones)/(n-l-i) if num_ones < m-l else 0
                
                P_string[tuple(string)] = Cond_p_string * P_string[tuple(string[:-1])]
                
                # alt_string is the string without last element and with the element before inverted
                alt_string = string[:-1].copy()  # Use copy to avoid modifying the original list
                alt_string[-1] = (alt_string[-1] + 1) % 2
                P_flip[tuple(string)] = Cond_p_flip * P_flip[tuple(alt_string)]
                
            elif string[-1] == 0: # I don't flip if l+i+1 is 0
                Cond_p_flip = 1 - math.comb(n-l-i-1, k-num_zeros-1)/math.comb(n-l-i, k-num_zeros) if num_zeros < k else 0
                Cond_p_string = 1 - (m-num_ones)/(n-l-i) if num_ones < m-l else 0
                
                P_string[tuple(string)] = Cond_p_string * P_string[tuple(string[:-1])]
                
                # alt_string is the string without last element and with the element before inverted
                alt_string = string[:-1].copy()  # Use copy to avoid modifying the original list
                alt_string[-1] = (alt_string[-1] + 1) % 2
                P_flip[tuple(string)] = Cond_p_flip * P_flip[tuple(alt_string)]
                
            P[tuple(string)] = P_flip[tuple(string)] * P_string[tuple(string)]
            
            Prob[i-1] += P[tuple(string)]
    
    Expected_increment = np.arange(1, max_i+1) * Prob
    
    return Expected_increment
                
if __name__ == "__main__":
    [n, l, m, k] = [4, 3, 3, 1]
    
    Expected_increment = Expected_increment_calculator(n, l, m, k)
    
    print(Expected_increment)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                