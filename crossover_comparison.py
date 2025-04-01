import sol
import numpy as np
import time
import random
from copy import deepcopy

def run_genetic_algorithm(crossover_method, target="Hello, world!", trial_num=0):
    """Run the genetic algorithm with specified parameters and return iterations until convergence"""
    
    # Set up parameters
    sol.GA_POPSIZE = 8192  # Larger population for more realistic convergence
    sol.GA_MAXITER = 5000  # Max iterations
    sol.GA_TARGET = target  # Set target BEFORE initializing population
    sol.GA_CROSSOVER_METHOD = crossover_method  # Crossover method to test
    sol.GA_FITNESS_MODE = "ascii"  # Using ASCII fitness only
    
    # Initialize with a truly unique seed for each trial
    random.seed(int(time.time() * 1000) + trial_num * 10000 + hash(crossover_method) % 1000)
    
    # Initialize population (AFTER setting the target)
    population, buffer = sol.init_population()
    
    # Track fitness history for this run
    best_fitness_history = []
    
    # Run algorithm until convergence or max iterations
    for iteration in range(sol.GA_MAXITER):
        sol.calc_fitness(population)
        sol.sort_by_fitness(population)
        best_fitness_history.append(population[0].fitness)
        
        # Check if we've converged
        if population[0].fitness == 0:
            return {
                'iterations': iteration + 1,  # Return iterations needed (+1 because 0-indexed)
                'fitness_history': best_fitness_history
            }
        
        # Create next generation
        sol.mate(population, buffer)
        population, buffer = sol.swap(population, buffer)
    
    # If we didn't converge, return max iterations and fitness history
    return {
        'iterations': sol.GA_MAXITER,
        'fitness_history': best_fitness_history
    }

def compare_crossover_methods(num_trials=30, target="Hello, world!"):
    """Compare different crossover methods over multiple trials"""
    
    methods = ["single", "two_point", "uniform"]
    results = {method: [] for method in methods}
    
    print(f"Running comparison with target: '{target}'")
    
    # Run each method multiple times
    for method in methods:
        print(f"\nTesting {method} crossover:")
        for trial in range(1, num_trials + 1):
            print(f"  Trial {trial}/{num_trials}... ", end="", flush=True)
            result = run_genetic_algorithm(method, target, trial)
            iterations = result['iterations']
            results[method].append(iterations)
            print(f"converged in {iterations} iterations")
    
    return results

def main():
    print("Comparing genetic algorithm crossover methods")
    print("---------------------------------------------")
    
    # Shorter target for faster convergence
    target = "Hello, world!"
    
    # Run comparison
    results = compare_crossover_methods(num_trials=30, target=target)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-----------------")
    for method in results:
        iterations = results[method]
        print(f"{method.capitalize()} Crossover: Mean: {np.mean(iterations):.2f}, Median: {np.median(iterations):.2f}, Std Dev: {np.std(iterations):.2f}")

if __name__ == "__main__":
    main()

