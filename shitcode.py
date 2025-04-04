import matplotlib.pyplot as plt
from sol import run_ga  # adjust if your GA code file is named differently

def my_tester_func():
    """
    tests multiple parameter combinations for the ga and produces
    a single figure with 3 subplots (one per crossover method).
    each subplot has 4 lines (2 best combos and 2 worst combos).
    """
    lcs_bonus_values = [1, 5, 10, 20, 50]
    mutation_rates = [0.25, 0.5, 0.75]
    fitness_modes = ["ascii", "lcs", "combined"]
    crossover_methods = ["single", "two_point", "uniform"]

    # store results keyed by crossover method
    results_by_crossover = {method: [] for method in crossover_methods}

    # gather total combos for better printing of progress
    total_combos = len(crossover_methods) * len(fitness_modes) * len(lcs_bonus_values) * len(mutation_rates)
    done_count = 0

    print(f"\nstarting tests for {total_combos} total combinations...\n")

    for method in crossover_methods:
        for fm in fitness_modes:
            for lb in lcs_bonus_values:
                for mr in mutation_rates:
                    done_count += 1
                    print(f"running combo #{done_count}/{total_combos} "
                          f"(method={method}, mode={fm}, lcs_bonus={lb}, mut={mr})...")

                    run_data = run_ga(
                        crossover_method=method,
                        fitness_mode=fm,
                        lcs_bonus=lb,
                        mutation_rate=mr,
                        population_size=4000
                    )

                    # store results along with metadata
                    results_by_crossover[method].append({
                        "fitness_mode": fm,
                        "lcs_bonus": lb,
                        "mutation_rate": mr,
                        "best_fitness_history": run_data["best_fitness_history"],
                        "converged_generation": run_data["converged_generation"]
                    })

                    print(f"  done. converged_generation={run_data['converged_generation']}\n")

    print("\nall runs completed. sorting results and generating final plot...")

    # create a figure with 3 subplots (one per crossover method)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, method in enumerate(crossover_methods):
        ax = axes[i]
        ax.set_title(f"crossover: {method}")
        ax.set_xlabel("generation")
        ax.set_ylabel("best fitness")

        # sort runs by converged_generation ascending
        method_results = sorted(
            results_by_crossover[method],
            key=lambda d: d["converged_generation"]
        )

        # pick the 2 best and 2 worst
        best_2 = method_results[:2]
        worst_2 = method_results[-2:]

        # combine for plotting
        lines_to_plot = best_2 + worst_2

        for run_info in lines_to_plot:
            fm = run_info["fitness_mode"]
            lb = run_info["lcs_bonus"]
            mr = run_info["mutation_rate"]
            conv_gen = run_info["converged_generation"]
            best_fit_hist = run_info["best_fitness_history"]

            # x-axis = generation indices
            generations = range(len(best_fit_hist))

            # label includes (mode, bonus, mut, converged)
            label = f"{fm}, lcs={lb}, mut={mr} => {conv_gen} gens"
            ax.plot(generations, best_fit_hist, label=label, linewidth=2)

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    print("\nplot displayed. testing script is complete.")

if __name__ == "__main__":
    my_tester_func()
