from ant_colony_optimization import *

test_graph = Graph(4, np.asarray([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]))

def test_traversal_works():
    for _ in range(9):
        traversal, weight = traverse_graph(test_graph)
        assert len(traversal) == test_graph.nodes
        assert len(set(traversal)) == len(traversal)

def test_aco_works():
    solution = ant_colony_optimization(test_graph, iterations = 500)
    assert len(solution) == test_graph.nodes
    assert cycle_length(test_graph, solution) == 80.0

def test_aco_bigger():
    dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

    bigger_graph = Graph(8, np.zeros((8,8)).astype('float64') )

    for i, j, dist in dist_list:
        bigger_graph.distance[i][j] = dist
        bigger_graph.distance[j][i] = dist

    solution = ant_colony_optimization(bigger_graph, iterations=500)

    assert cycle_length(bigger_graph, solution) < 18. # between 17~18? √ 17.34

