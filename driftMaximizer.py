import numpy as np
import math

def generate_bit_strings(length):
    return np.array(list(itertools.product([0, 1], repeat=length)))

def Expected_increment_calculator(n, l, m, k, max_i):
    strings = {}
    P = {} # probability single string
    Prob = {} # probability exact fitness increment
    
    strings[1] = generate_bit_strings(length = 1)
    
    i = 1 # current value of increment considering
    
    # QUESTI DUE VALORI SOTTO SONO SBAGLIATI PERCHé NON TENGONO CONTO DEL FATTO CHE SIA O MENO POSSIBILE AVERE 11 (in base a m) E IN BASE A k se sia possibile flipparne 2
    
    P[(0)] = k/(n-l-1) * (1-k/(n-l)) * (1 - (m-l)/(n-l-1)) * (math.comb(n-l, k)/math.comb(n, k)) # first value: i, second value: number of ones
    P[(1)] = (k-1)/(n-l-1) * (k/(n-l)) * ((m-l)/(n-l-1)) * (math.comb(n-l, k)/math.comb(n, k))
            
    for i in range(2, max_i+1):
        strings[i] = generate_bit_strings(i)
        
        for string in strings[i]:
            ones = np.sum(string)
            flipped = i-ones
            
            if string[-1] == 0: # flippo
                A = (k-flipped)/(n-l-i) if flipped < k else 0
                B = (n-l-ones)/(n-l-i) if ones < m-l else 0
                
            elif string[-1] == 1: # flippo
                A = 1 - (k-flipped)/(n-l-i) if flipped < k else 1
                B = 1 - (n-l-ones)/(n-l-i) if ones < m-l else 1
                
            P[string] = A*B*P[string[:-1]]
            
            Prob[i] += P[string]
    
    Expected_increment = np.arange(1, max_i+1) * 
    return 
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                