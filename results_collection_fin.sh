# Argument: $1 - number of objective function evaluations
# Num FE: 1k, 5k, 10k, 50k, 100k (plot all 5 in log scale of x axis)
# Finite order system
nohup python3 results_collection.py finite $1 sa finite_sa_1k&
nohup python3 results_collection.py finite $1 pso finite_pso_1k&
nohup python3 results_collection.py finite $1 acor finite_acor_1k&