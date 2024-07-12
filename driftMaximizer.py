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
    
    if l < n-1:
        if m == n-1:
            P_string[(0,)] = 0
        elif l == n-2:
            if m > l:
                P_string[(0,)] = 0
            elif m == l:
                P_string[(0,)] = 1
        elif l < n-2:
            P_string[(0,)] = math.comb(n-(l+1), m-l)/math.comb(n-(l+2), m-l)
            
        if k > 1 and k <= n-l:
            P_flip[(0,)] = (math.comb(n-l, k)/math.comb(n, k)) * (math.comb(n-l-1, k-1)/math.comb(n-l, k)) * (1-(math.comb(n-l-2, k-2)/math.comb(n-l-1, k-1)))
        elif k > n-l:
            P_flip[(0,)] = 0
        elif k == 1:
            P_flip[(0,)] = 1/n
    
    # Correggere tutti questi if/elif/else e capire l'ordine
    
    elif l == n-1:
        P_string[(0,)] = 0 # Questo ha poco senso ma non so che mettere, forse non è importante
        P_flip[(0,)] = 0
        P[(0,)] = 0
        
    P[(0,)] = P_string[(0,)] * P_flip[(0,)]
    
    # Domanda stupida: P[1] non è semplicemente 1 - P[0]?
    
    if l < n-1:
        if m == n-1:
            P_string[(1,)] = 1
        elif l == n-2:
            if m > l:
                P_string[(1,)] = 1
        elif m == l:
            P_string[(1,)] = 0
        elif l < n-2:
            P_string[(1,)] = math.comb(n-(l+1), m-l)/math.comb(n-(l+2), m-l)
            
        if k > 1 and k <= n-l:
            P_flip[(1,)] = (math.comb(n-l, k)/math.comb(n, k)) * (math.comb(n-l-1, k-1)/math.comb(n-l, k)) * (math.comb(n-l-2, k-2)/math.comb(n-l-1, k-1))
        elif k > n-l:
            P_flip[(1,)] = 0
        elif k == 1:
            P_flip[(1,)] = 0
    
    elif l == n-1:
        P_string[(1,)] = 1 # Questo ha poco senso ma non so che mettere, forse non è importante
        
        if k == 1:
            P_flip[(1,)] = 1/n
        else:
            P_flip[(1,)] = 0
                
    P[(1,)] = P_string[(1,)] * P_flip[(1,)]
    
    Prob[i-1] = P[(0,)] + P[(1,)] # Probability of increment exactly 1. The -1 is because the increment starts from 1 and the indeces from 0
    
    for i in range(2, max_i + 1):
        if l+i+1 > n:
            for string in strings[i-1]:
                Prob_s = P_string[tuple(string)]
                
                alt_string = string.copy()  # Use copy to avoid modifying the original list
                alt_string[-1] = (alt_string[-1] + 1) % 2
                Prob_f = P_flip[tuple(alt_string)] 
                
                Prob[i-1] += Prob_s * Prob_f
            
        elif l+i+1 <= n:
            strings[i] = generate_bit_strings(i) # Idealmente a questo punto filtro le stringhe possibili con l e m dati. Però forse mi serve tenerle per dopo... O forse no
            
            for string in strings[i]: # We are considering strings 1...10string
                num_ones = np.sum(string)
                num_zeros = n - num_ones
                    
                if string[-1] == 1: # I flip if l+i+1 is 1
                    if P_string[tuple(string[:-1])] == 0:
                        P_string[tuple(string)] = 0
                    else:
                        Cond_p_string = (m-num_ones+1)/(n-l-i) if num_ones <= m-l else 0
                        
                        P_string[tuple(string)] = Cond_p_string * P_string[tuple(string[:-1])]
                    
                    # alt_string is the string without last element and with the element before inverted
                    alt_string = string[:-1].copy()  # Use copy to avoid modifying the original list
                    alt_string[-1] = (alt_string[-1] + 1) % 2
                    if P_flip[tuple(alt_string)] == 0:
                        P_flip[tuple(string)] = 0
                    else:
                        Cond_p_flip = math.comb(n-l-i-1, k-num_zeros-1)/math.comb(n-l-i, k-num_zeros) if (num_zeros < k and k-num_zeros <= n-l-i) else 0
                        P_flip[tuple(string)] = Cond_p_flip * P_flip[tuple(alt_string)]
                    
                elif string[-1] == 0: # I don't flip if l+i+1 is 0
                    if P_string[tuple(string[:-1])] == 0:
                        P_string[tuple(string)] = 0
                    else:   
                        Cond_p_string = 1 - (m-num_ones)/(n-l-i) if num_ones < m else 1
                        P_string[tuple(string)] = Cond_p_string * P_string[tuple(string[:-1])]
                    
                    # alt_string is the string without last element and with the element before inverted
                    alt_string = string[:-1].copy()  # Use copy to avoid modifying the original list
                    alt_string[-1] = (alt_string[-1] + 1) % 2
                    if P_flip[tuple(alt_string)] == 0:
                        P_flip[tuple(string)] = 0
                    else:
                        Cond_p_flip = 1 - math.comb(n-l-i-1, k-num_zeros-1)/math.comb(n-l-i, k-num_zeros) if (num_zeros < k and k-num_zeros <= n-l-i) else 0 # Non mi convince il < stretto
                        P_flip[tuple(string)] = Cond_p_flip * P_flip[tuple(alt_string)]
                    
                P[tuple(string)] = P_flip[tuple(string)] * P_string[tuple(string)]
                
                Prob[i-1] += P[tuple(string)]
    
    Expected_increment = np.dot(np.arange(1, max_i+1), Prob)
    
    return Expected_increment
                
if __name__ == "__main__":
    n = 3
    l = 0
    m = 0
    k = 3
    
    if k < 1 or m < l or l >= n:
        print("Wrong input")
    else:
        Expected_increment = Expected_increment_calculator(n, l, m, k)
    
    print(Expected_increment)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                