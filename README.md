# Metaheuristics in the optimization of Laguerre-Volterra networks

The Laguerre-Volterra network (LVN) is a Volterra-equivalent connectionist architecture, which combines a bank of discrete Laguerre filters and a layer of polynomial activation functions.
This architecture is designed to model nonlinear dynamic systems from input-output signals.

Here we optimize the continuous parameters of the LVN to model synthetic systems using different metaheuristics, with the purpose of performance evaluation.


## Third party software versions
* Python 3.6.9
    * NumPy 1.17.3 (vector math)
    * Scipy 1.3.0 (Friedman significance test)
    * scikit-posthocs 0.6.1 (Nemenyi post-hoc significance test)
    * Matplotlib 3.0.3 (plotting)
    
    
## List of modules
* base_metaheuristic.py
    + simulated_annealing.py
    + particle_swarm_optimization.py
    + ant_colony_for_continuous_domains.py
* laguerre_volterra_network_structure.py
* optimization_utilities.py
* simulated_systems.py
* data_handling.py

## Scripts and their uses
* generate_datasets.py          - Uses the data_handling module to generate synthetic train and test IO signals from simulated systems.
* optimize_LVN.py               - Optimizes LVNs with arbitrary structure using different metaheuristics (mostly used for verification)
* results_collection.py         - Runs some specified metaheuristic 30 times and stores the solutions found, along with their errors on test signals
* results_stats.py              - With the results from 'results_collection.py', compute averages and standard deviations for train and test errors
* results_stats_significance.py - Compute the statistical significance of the results with the Friedman and Nemenyi tests
* plotting scripts

### If this repository is valuable to you, consider citing:
Costa, V. O. and Müller, M. F. (2020). "Evaluation of Metaheuristics in the Optimization of Laguerre-Volterra Networks for Nonlinear Dynamic System Identification". <i>  9th Brazilian Conference on Intelligent Systems, BRACIS </i> (2020).
