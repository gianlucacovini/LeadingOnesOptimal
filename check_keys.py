from optimalLeadingOnes import LeadingOnes
from optimalLeadingOnes import OneMax

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