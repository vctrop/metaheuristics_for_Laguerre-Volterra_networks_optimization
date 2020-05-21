# Num FE: 1 k, 5 k, 10 k, 50 k, 100 k (plot all 5 in log scale of x axis)
# Finite order system
nohup python3 results_collection.py finite sa&
nohup python3 results_collection.py finite pso&
nohup python3 results_collection.py finite acor&
nohup python3 results_collection.py infinite sa&
nohup python3 results_collection.py infinite pso&
nohup python3 results_collection.py infinite acor&