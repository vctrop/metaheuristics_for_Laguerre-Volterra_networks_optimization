import sys
import pickle as pkl

if len(sys.argv) != 2:
    print("Error, please enter the base filename of the desired results")
    exit(-1)
    
file_basename = sys.argv[1]
# Print solutions found for each round
with open("./results/" + file_basename + "_solutions.pkl", "rb") as solutions_file:
   found_solutions = pkl.load(solutions_file)
   print("FOUND SOLUTIONS")
   print(found_solutions)
# Print NMSE on test data for the solutions of each round
with open("./results/" + file_basename + "_test_costs.pkl", "rb") as test_file:
   test_costs = pkl.load(test_file)
   print("TEST COSTS")
   print(test_costs)