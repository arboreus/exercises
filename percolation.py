### Percolation

#%% Clear variable list
def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
clearall()

#%%
class Percolation(object):
    def __init__(self, N):
        self.N = N
        self.grid = [[0 for i in range(N)] for j in range(N)]
        
    def line(self):
        return ''.join([''.join(x) for x in [[str(x) for x in y] for y in self.grid]])
    
    def plot(self):
        print('\n'.join([''.join(x) for x in [[str(x) for x in y] for y in self.grid]]))
        
    def open(self, i , j):
        self.grid[i][j] = 1
    
    def isopen(self, i, j):
        return self.grid[i][j] > 0
    
    def isfull(self, i, j):
        return self.grid[i][j] == 2
        
    def fill(self):
        self.grid[0] = [2 if self.isopen(0, j) else self.grid[0][j] for j in range(self.N)]
        i = 1
        while i < self.N:
            self.grid[i] = [2 if self.isopen(i, j) and self.isfull(i-1, j) else self.grid[i][j] for j in range(self.N)]
            for j in range(1, self.N):
                self.grid[i][j] = 2 if self.isopen(i, j) and self.isfull(i, j-1) else self.grid[i][j]
            for j in range(self.N-2, -1, -1):
                self.grid[i][j] = 2 if self.isopen(i, j) and self.isfull(i, j+1) else self.grid[i][j]  
            i += 1
            continue
    
    def empty(self):
        self.grid = [[1 if x > 0 else 0 for x in y] for y in self.grid]
       
    def percolates(self):
        self.fill()
        perc = 2 in self.grid[self.N-1]
        self.empty()
        return perc

#%%
from random import choice

def experiment(N):
    count = 0
    obj = Percolation(N)
    blocked = [i for i in range(N*N) if obj.line()[i]=='0']
    while not obj.percolates():
        index = choice(blocked)     
        obj.open(index // N, index % N)
        count += 1
    return count / (N*N)
    
#%%
    



#%%