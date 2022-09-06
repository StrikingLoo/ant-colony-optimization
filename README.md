# Ant Colony Optimization

Implementation from the ground up in Python of the Ant Colony Optimization algorithm for Traveling Salesman Problem.

To adapt it to new problems, just modify the traversal function, or write a new one (and if you do, you can do a PR!).

Tests are run through pytest, and the algorithm includes elitism, can have restarts when the system stagnates (they haven't had a good performance in my benchmarks, but feel free to experiment with them) and manages different levels of verbosity.

