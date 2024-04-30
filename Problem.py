import numpy as np

# DOMANDE: Non so se è una buona cosa far calcolare ogni volta la fitness e non so se
# tutto ciò era in qualche modo già implementato

class Problem:
    
    def __init__(self,n):
        self.size = n
        
    def step(self,k):
        pass
    
    def isDone(self):
        pass
    
class OneMaxProblem(Problem):
    def __init__(self,n,startPoint = []):
        ''' Initialisation of the OneMax problem'''
        self.size = n
        if len(startPoint) == 0:
            self.vector = np.random.randint(2,size=self.size) # self.vector è lo stato iniziale
        else:
            self.vector = startPoint
        
    def step(self,k):
        ''' flip k bits of the problem '''
        #We get the bits we'll flip
        toFlip = np.random.choice(self.size,k,replace=False)
        
        badFlip = sum([self.vector[i] for i in toFlip]) # OneMax function evaluation
        # If we'll flip more ones into zeroes than the contrary, we return 0
        if badFlip >= k/2:
            return 0
        
        else:
            #We flip the bits
            for i in toFlip:
                self.vector[i] = (self.vector[i] + 1)%2
            return k - badFlip
        
    def isDone(self):
        return sum(self.vector) == self.size
    
    def getState(self): # Evaluation function of OneMax fitness
        return sum(self.vector)
    
class LeadingOnesProblem(Problem):
    def __init__(self,n,startPoint = []):
        ''' Initialisation of the LeadingOnes problem'''
        self.size = n
        
        if len(startPoint) == 0:
            self.vector = np.random.randint(2,size=self.size)
        else:
            self.vector = startPoint
            
    def getLOfitness(self): # Evaluation function of LeadingOnes fitness
        return max(np.arange(1, len(self.vector) + 1) * (np.cumprod(self.vector) == 1))
    
    def getOMfitness(self): # Evaluation function of LeadingOnes fitness
        return sum(self.vector)
        
    def step(self,k):
        ''' flip k bits of the problem '''
        #We get the bits we'll flip
        toFlip = np.random.choice(self.size,k,replace=False)
        
        newSol = [(self.vector[i]+1)%2 for i in toFlip]
        
        oldFitness = max(np.arange(1, len(self.vector) + 1) * (np.cumprod(self.vector) == 1))
        newFitness = max(np.arange(1, len(newSol) + 1) * (np.cumprod(newSol) == 1))
        
        # The function returns the improvement of the solution
        if newFitness > oldFitness:
            for i in toFlip:
                self.vector[i] = (self.vector[i] + 1)%2
            return newFitness - oldFitness
        
        else:
            return 0
        
    def isDone(self):
        return sum(self.vector) == self.size

class LexicographicLeadingOnesProblem(Problem):
    def __init__(self,n,startPoint = []):
        ''' Initialisation of the LeadingOnes problem'''
        self.size = n
        
        if len(startPoint) == 0:
            self.vector = np.random.randint(2,size=self.size)
        else:
            self.vector = startPoint
            
    def getLOfitness(self): # Evaluation function of LeadingOnes fitness
        return max(np.arange(1, len(self.vector) + 1) * (np.cumprod(self.vector) == 1))
    
    def getOMfitness(self): # Evaluation function of LeadingOnes fitness
        return sum(self.vector)
        
    def step(self,k):
        ''' flip k bits of the problem '''
        #We get the bits we'll flip
        toFlip = np.random.choice(self.size,k,replace=False)
        
        newSol = [(self.vector[i]+1)%2 for i in toFlip]
        
        oldFitness = max(np.arange(1, len(self.vector) + 1) * (np.cumprod(self.vector) == 1))
        newFitness = max(np.arange(1, len(newSol) + 1) * (np.cumprod(newSol) == 1))
        
        oldOM = sum(self.vector)
        newOM = sum(newSol)
        
        # The function returns the improvement of the solution - Deve restituire anche OM improvement??
        if newFitness > oldFitness:
            for i in toFlip:
                self.vector[i] = (self.vector[i] + 1)%2
            return newFitness - oldFitness
        
        if newFitness == oldFitness and newOM > oldOM:
           for i in toFlip:
               self.vector[i] = (self.vector[i] + 1)%2
           return 0
        
        else:
            return 0
        
    def isDone(self):
        return sum(self.vector) == self.size