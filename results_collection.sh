# Num FE: 1 k, 5 k, 10 k, 50 k, 100 k (plot all 5 in log scale of x axis)
# Finite order system
# 1 k
nohup python3 results_collection.py finite 1000 sa        finite_sa_1k&
nohup python3 results_collection.py finite 1000 pso       finite_pso_1k&
nohup python3 results_collection.py finite 1000 acor      finite_acor_1k&
# 5 k   
nohup python3 results_collection.py finite 5000 sa        finite_sa_5k&
nohup python3 results_collection.py finite 5000 pso       finite_pso_5k&
nohup python3 results_collection.py finite 5000 acor      finite_acor_5k&
# 10 k  
nohup python3 results_collection.py finite 10000 sa       finite_sa_10k&
nohup python3 results_collection.py finite 10000 pso      finite_pso_10k&
nohup python3 results_collection.py finite 10000 acor     finite_acor_10k&
# 50 k  
nohup python3 results_collection.py finite 50000 sa       finite_sa_50k&
nohup python3 results_collection.py finite 50000 pso      finite_pso_50k&
nohup python3 results_collection.py finite 50000 acor     finite_acor_50k&
# 100 k 
nohup python3 results_collection.py finite 100000 sa      finite_sa_100k&
nohup python3 results_collection.py finite 100000 pso     finite_pso_100k&
nohup python3 results_collection.py finite 100000 acor    finite_acor_100k&

# Infinite order system
# 1 k
nohup python3 results_collection.py infinite 1000 sa        infinite_sa_1k&
nohup python3 results_collection.py infinite 1000 pso       infinite_pso_1k&
nohup python3 results_collection.py infinite 1000 acor      infinite_acor_1k&
# 5 k   
nohup python3 results_collection.py infinite 5000 sa        infinite_sa_5k&
nohup python3 results_collection.py infinite 5000 pso       infinite_pso_5k&
nohup python3 results_collection.py infinite 5000 acor      infinite_acor_5k&
# 10 k  
nohup python3 results_collection.py infinite 10000 sa       infinite_sa_10k&
nohup python3 results_collection.py infinite 10000 pso      infinite_pso_10k&
nohup python3 results_collection.py infinite 10000 acor     infinite_acor_10k&
# 50 k  
nohup python3 results_collection.py infinite 50000 sa       infinite_sa_50k&
nohup python3 results_collection.py infinite 50000 pso      infinite_pso_50k&
nohup python3 results_collection.py infinite 50000 acor     infinite_acor_50k&
# 100 k 
nohup python3 results_collection.py infinite 100000 sa      infinite_sa_100k&
nohup python3 results_collection.py infinite 100000 pso     infinite_pso_100k&
nohup python3 results_collection.py infinite 100000 acor    infinite_acor_100k&
