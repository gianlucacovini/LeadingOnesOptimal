import numpy as np
import math
import itertools

def generate_bit_strings(length):
    return np.array(list(itertools.product([0, 1], repeat=length)))

def Expected_increment_calculator(n, l, m, k): # forse il max_i che ci interessa dev'essere sempre n-l
    max_i = n-l

    strings = {}
    P = {} # probability single string
    Prob = np.zeros(max_i) # probability exact fitness increment
    
    strings[1] = generate_bit_strings(length = 1)
    
    i = 1 # current value of increment considering
    
    if l < n-1 and k <= n-l:
        P[(0,)] = k/(n-l-1) * (1-k/(n-l)) * (1 - (m-l)/(n-l-1)) * (math.comb(n-l, k)/math.comb(n, k))
    else:
        P[(0,)] = 0
        
    if k >= 2 and m-l >= 1 and l < n-1 and k<= n-l:
        P[(1,)] = (k-1)/(n-l-1) * (k/(n-l)) * ((m-l)/(n-l-1)) * (math.comb(n-l, k)/math.comb(n, k))
    else:
        P[(1,)] = 0
        
    if l == n-1:
        P[(0,)] = 1/n
        P[(1,)] = 0
    
    Prob[0] = P[(0,)] + P[(1,)]
    
    for i in range(2, max_i):
        strings[i] = generate_bit_strings(i)
        
        for string in strings[i]:
            ones = np.sum(string)
            flipped = i-ones
            
            if string[-1] == 0: # I don't flip if l+i+1 is 0
                A = 1 - (k-flipped)/(n-l-i) if flipped < k else 1
                B = 1 - (m-l-ones)/(n-l-i) if ones < m-l else 1
                
            elif string[-1] == 1: # I flip is l+i+1 is 1
                A = (k-flipped)/(n-l-i) if flipped < k else 0
                B = (m-l-ones)/(n-l-i) if ones < m-l else 0
                
            P[tuple(string)] = A*B*P[tuple(string)[:-1]]
            
            Prob[i-1] += P[tuple(string)]
    
    Expected_increment = np.arange(1, max_i+1) * Prob
    
    return Expected_increment
                
if __name__ == "__main__":
    [n, l, m, k] = [4, 1, 3, 1]
    
    Expected_increment = Expected_increment_calculator(n, l, m, k)
    
    print(Expected_increment)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                