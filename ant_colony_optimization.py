import numpy as np
import random
from multiprocessing import Pool

'''
procedure ACO_MetaHeuristic is
    while not terminated do
        generateSolutions()
        daemonActions()
        pheromoneUpdate()
    repeat
end procedure
'''

class Graph():
    def __init__(self, nodes, distance, default_pheromone_level = None):
        self.nodes = nodes
        self.distance = distance
        assert distance.shape[1] == distance.shape[0]
        if default_pheromone_level:
            self.intensity = np.full_like(distance, default_pheromone_level).astype('float64')
        else:
            self.intensity = np.full_like(distance, self.distance.mean()*10).astype('float64')
        

    def __str__(self):
        return f'nodes: {str(self.nodes)}\n{self.distance}\n{self.intensity}'

'''
The general algorithm is relatively simple and based on a set of ants, 
each making one of the possible round-trips along the cities. 
At each stage, the ant chooses to move from one city to another according to some rules:

- It must visit each city exactly once;
- A distant city has less chance of being chosen (the visibility);
- The more intense the pheromone trail laid out on an edge between two cities, the greater the probability that that edge will be chosen;
- Having completed its journey, the ant deposits more pheromones on all edges it traversed, if the journey is short;
- After each iteration, trails of pheromones evaporate.

Random ish complete graph: Graph(4, [[0,10,15,20],[10,0,35,25],[10,35,0,30],[20,25,30,0]])
'''

test_graph = Graph(4, np.asarray([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]).astype('float64'), )

best_so_far = [132, 133, 131, 130, 127, 128, 123, 129, 126, 125, 124, 0, 158, 1, 2, 3, 4, 152, 151, 6, 5, 155, 156, 157, 153, 154, 150, 149, 148, 143, 142, 141, 139, 138, 140, 144, 145, 146, 147, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 29, 30, 31, 32, 34, 33, 35, 36, 137, 136, 135, 134, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 73, 74, 75, 72, 70, 71, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 94, 95, 86, 87, 93, 88, 89, 90, 91, 92, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 41]

def cycle_length(g, cycle):
    length = 0
    i = 0
    while i < len(cycle) -1:
        length += g.distance[cycle[i]][cycle[i+1]]
        i+=1
    length+= g.distance[cycle[i]][cycle[0]]
    return length

def add_artificial_good_cycle(g):
    size = g.distance.shape[0]

    for i in range(size-1):
        g.distance[i][i+1]/=10
    g.distance[size-1][0]/=10



def ant_colony_optimization(g, verbose=True, iterations = 100, ants_per_iteration = 50, q = None, degradation_factor = .9, use_inertia = False, run_experiment_break=False, run_experiment_artificial_good_cycle=False):
    total_ants = 0
    
    if q is None:
        q = g.distance.mean()

    best_cycle = None #best_so_far #hardcoded instance. 
    best_length = float('inf') #cycle_length(g, best_so_far) #hardcoded instance. Else use inf

    old_best = None
    inertia = 0
    patience = 100
    index = None
    if run_experiment_break or run_experiment_artificial_good_cycle:
        pheromone_history = []

    for iteration in range(iterations):
        print(f'iteration {iteration} \n' if (verbose and iteration%50==0) else '', end='')
        print(f'best weight so far: {round(best_length,2)}\n' if (verbose and iteration%50==0) else '', end='')
        print(f'average intensity {g.intensity.mean()}\n' if (verbose and iteration%50==0) else '', end='')

        if iteration == 500:
            if run_experiment_artificial_good_cycle:
                add_artificial_good_cycle(g)
            if run_experiment_break:
                index = break_most_traversed_edge(g, 10)
        if iteration >= 500:
            if add_artificial_good_cycle:
                levels = []
                size = g.distance.shape[0]
                for i in range(size-1):
                    levels.append(g.intensity[i][i+1])
                levels.append(g.intensity[size-1][0])
                pheromone_history.append(levels)

            if run_experiment_break:
                pheromone_history.append(g.intensity[index])


        cycles = [traverse_graph(g, random.randint(0, g.nodes -1)) for _ in range(ants_per_iteration)]

        cycles.sort(key = lambda x: x[1])
        cycles = cycles[: ants_per_iteration//2]
        total_ants+=ants_per_iteration

        if best_cycle: #elitism
            cycles.append((best_cycle, best_length))
            
            if use_inertia:
                old_best = best_length

        for cycle, total_length in cycles:

            total_length = cycle_length(g, cycle)
            if total_length < best_length:
                best_length = total_length
                best_cycle = cycle

            delta = q/total_length
            i = 0
            while i < len(cycle) -1:
                g.intensity[cycle[i]][cycle[i+1]]+= delta
                i+=1
            g.intensity[cycle[i]][cycle[0]] += delta
            g.intensity *= degradation_factor
        
        
        if use_inertia and best_cycle:
                        
            if old_best == best_length:
                    inertia+=1
            else:
                inertia = 0

            if inertia > patience:
                print('applying shake')
                g.intensity += g.intensity.mean()
        
    if run_experiment_break or run_experiment_artificial_good_cycle:
        with open('phero_history_exp2_10.txt','w') as f:
            f.write(str(pheromone_history))

    return best_cycle

'''
     -
    - -
'''

def traverse_graph(g, source_node = 0):
    visited = np.asarray([1 for _ in range(g.nodes)])
    visited[source_node] = 0

    cycle = [source_node]
    steps = 0
    current = source_node
    total_length = 0
    while steps < g.nodes -1:

        jumps_neighbors = []
        jumps_values = []
        for node in range(g.nodes):
            if visited[node] != 0:
               sediment = max(g.intensity[current][node], 1e-5)
               v = (sediment**0.9 ) / (g.distance[current][node]**1.5) 
               jumps_neighbors.append(node)
               jumps_values.append(v)

        #jumps = (g.intensity[current]*0.9 ) / ((g.distance[current]+0.00001)**1.5)
        #jumps = np.where(visited > 1e-5, jumps, 0.)
        next_node = random.choices(jumps_neighbors, weights = jumps_values)[0]
        
        visited[next_node] = 0
        
        current = next_node
        cycle.append(current)
        steps+=1

    total_length = cycle_length(g, cycle)
    assert len(list(set(cycle))) == len(cycle)
    return cycle, total_length

def traverse(g, cycle):
    i = 0
    while i < len(cycle) -1:
        print([cycle[i], cycle[i+1]])
        print(g.distance[cycle[i]][cycle[i+1]])
        i+=1
    print([cycle[i], cycle[0]])
    print(g.distance[cycle[i]][cycle[0]])

def break_most_traversed_edge(g, constant):
    index = g.intensity.argmax()
    index = np.unravel_index(index, g.intensity.shape)
    g.distance[index]*=constant
    return index # for logging purposes





