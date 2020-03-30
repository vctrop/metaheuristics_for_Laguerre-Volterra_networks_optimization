# Argument: $1 - number of objective function evaluations
# Num FE: 1k, 5k, 10k, 50k, 100k (plot all 5 in log scale of x axis)
# Infinite order system
nohup python3 results_collection.py infinite $1 sa infinite_sa_1k&
nohup python3 results_collection.py infinite $1 pso infinite_pso_1k&
nohup python3 results_collection.py infinite $1 acor infinite_acor_1k&