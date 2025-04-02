import sol
import random
import time
import math
import matplotlib.pyplot as plt

def run_ga(lcs_bonus, crossover_method, mutation_rate, fitness_mode, max_iter=sol.GA_MAXITER):
    # Store old global values to restore later
    old_lcs_bonus = sol.GA_LCS_BONUS
    old_crossover = sol.GA_CROSSOVER_METHOD
    old_mutrate = sol.GA_MUTATIONRATE
    old_fitmode = sol.GA_FITNESS_MODE
    old_maxiter = sol.GA_MAXITER

    # Set new params
    sol.GA_LCS_BONUS = lcs_bonus
    sol.GA_CROSSOVER_METHOD = crossover_method
    sol.GA_MUTATIONRATE = mutation_rate
    sol.GA_FITNESS_MODE = fitness_mode
    sol.GA_MAXITER = max_iter

    # GA loop, similar to main but returning best_history instead of plotting
    population, buffer = sol.init_population()
    best_history = []
    for iteration in range(sol.GA_MAXITER):
        sol.calc_fitness(population)
        sol.sort_by_fitness(population)
        best_history.append(population[0].fitness)
        if iteration % 1000 == 0:
            print(f"[run_ga] iteration={iteration}, best fitness={population[0].fitness}")
        if population[0].fitness == 0:
            break
        sol.mate(population, buffer)
        population, buffer = sol.swap(population, buffer)

    # Restore globals
    sol.GA_LCS_BONUS = old_lcs_bonus
    sol.GA_CROSSOVER_METHOD = old_crossover
    sol.GA_MUTATIONRATE = old_mutrate
    sol.GA_FITNESS_MODE = old_fitmode
    sol.GA_MAXITER = old_maxiter

    return best_history

def main():
    lcs_bonuses = [1, 5, 10, 20, 50]
    mutation_rates = [0.25, 0.5, 0.7]
    fitness_modes = ["ascii", "lcs", "combined"]
    crossover_methods = ["single", "two_point", "uniform"]

    # Prepare data: {method: {(lcs_bonus, mut_rate, fit_mode): best_history}}
    results = {method: {} for method in crossover_methods}

    for method in crossover_methods:
        for lcs_bonus in lcs_bonuses:
            for mut_rate in mutation_rates:
                for fit_mode in fitness_modes:
                    print(f"Running GA with method={method}, LCS bonus={lcs_bonus}, mutation rate={mut_rate}, fitness mode={fit_mode}")
                    combo = (lcs_bonus, mut_rate, fit_mode)
                    history = run_ga(lcs_bonus, method, mut_rate, fit_mode)
                    results[method][combo] = history

    # Plot per method
    for method in crossover_methods:
        plt.figure()
        for combo, history in results[method].items():
            label_str = f"LCS={combo[0]}, MUT={combo[1]}, FIT={combo[2]}"
            plt.plot(history, label=label_str)
        plt.title(f"Best Fitness vs. Generation ({method} crossover)")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
