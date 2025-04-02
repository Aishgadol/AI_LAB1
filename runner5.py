import matplotlib.pyplot as plt
import sol

def run_experiment(lcs_bonus, crossover, mutation, fit_mode):
    sol.GA_LCS_BONUS = lcs_bonus
    sol.GA_CROSSOVER_METHOD = crossover
    sol.GA_MUTATIONRATE = mutation
    sol.GA_FITNESS_MODE = fit_mode

    print(f"Running experiment with LCS={lcs_bonus}, crossover={crossover}, mutation={mutation}, fit_mode={fit_mode}")

    population, buffer = sol.init_population()
    best_history = []

    for i in range(sol.GA_MAXITER):
        sol.calc_fitness(population)
        sol.sort_by_fitness(population)
        best_history.append(population[0].fitness)
        print(f"Iteration {i+1}: Best fitness so far = {population[0].fitness}")

        if population[0].fitness == 0:
            break

        sol.mate(population, buffer)
        population, buffer = sol.swap(population, buffer)

    return best_history

def main():
    lcs_list = [1, 5, 10, 20, 50]
    mut_list = [0.25, 0.5, 0.7]
    fit_modes = ["ascii", "lcs", "combined"]
    crossovers = ["single", "two_point", "uniform"]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    for idx, crossover in enumerate(crossovers):
        ax = axs[idx]
        ax.set_title(f"Crossover: {crossover}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")

        for lcs_bonus in lcs_list:
            for mutation_rate in mut_list:
                for fit_mode in fit_modes:
                    best_hist = run_experiment(lcs_bonus, crossover, mutation_rate, fit_mode)
                    label_str = f"LCS\\_bonus={lcs_bonus}, Mut={mutation_rate}, Fit={fit_mode}"
                    ax.plot(best_hist, label=label_str)

        ax.legend(loc="upper right")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()