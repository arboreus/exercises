#%% Clear variable list
def clearall():
    """clear all globals"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]
clearall()

#%%
lx, ly, tx, ty = 36, 17, 0, 0
xlim = [0, 40]
ylim = [0, 18]

#%%
def neighbor_nodes(node):
    nodes = []
    for i in range(-1, 2):
        for j in range(-1, 2):
           nodes.append([node[0]+i, node[1]+j])
    nodes = [[x, y] for x, y in nodes if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1] and [x, y] != node]
    return nodes

def dist_between(nodex, nodey):
    distx = abs(nodex[0]-nodey[0])
    disty = abs(nodex[1]-nodey[1])
    return max(distx, disty)
    
def heuristic_cost_estimate(nodex, nodey):
    return dist_between(nodex, nodey)
    
def reconstruct_path(came_from, current):
    total_path = [current]
    while str(current) in list(came_from.keys()):
        current = came_from[str(current)]
        total_path.append(current)
    return total_path
    
def astar(start, goal):
    closedset = []
    openset = [start]
    came_from = {}
    g_score = {}
    
    g_score[str(start)] = 0
    def f_score(node, goal): return g_score[str(node)] + heuristic_cost_estimate(node, goal)
        
    while openset:
        current = [x for x in openset if f_score(x, goal) == min([f_score(x, goal) for x in openset])][0]
        if current == goal:
            return reconstruct_path(came_from, goal)
        openset.remove(current)
        closedset.append(current)
        for neighbor in neighbor_nodes(current):
            if neighbor in closedset:
                continue
            tentative_g_score = g_score[str(current)] + dist_between(current, neighbor)
            if neighbor not in openset:
                came_from[str(neighbor)] = current
                g_score[str(neighbor)] = tentative_g_score
                openset.append(neighbor)
            elif tentative_g_score < g_score[str(neighbor)]:
                came_from[str(neighbor)] = current
                g_score[str(neighbor)] = tentative_g_score

#%%
path = list(reversed(astar([0, 0], [36, 17])))

#%%


