# test_parameters.py
import matplotlib.pyplot as plt
# If your GA code is in ga_code.py, adjust the line below accordingly:
from ga_code import run_ga  # or however you import run_ga


def test_ga_combinations():
    lcs_bonus_values = [1, 5, 10, 20, 50]
    mutation_rates = [0.25, 0.5, 0.75]
    fitness_modes = ["ascii", "lcs", "combined"]
    crossover_methods = ["single", "two_point", "uniform"]

    # We’ll store all runs by crossover method, e.g.:
    # results_by_crossover = {
    #   "single": [
    #       {
    #           "fitness_mode": "ascii",
    #           "lcs_bonus": 1,
    #           "mutation_rate": 0.25,
    #           "best_fitness_history": [...],
    #           "converged_generation": ...
    #       },
    #       ...
    #   ],
    #   "two_point": [...],
    #   "uniform": [...]
    # }
    results_by_crossover = {method: [] for method in crossover_methods}

    # Run GA for each combination
    for method in crossover_methods:
        for fm in fitness_modes:
            for lb in lcs_bonus_values:
                for mr in mutation_rates:
                    run_data = run_ga(
                        crossover_method=method,
                        fitness_mode=fm,
                        lcs_bonus=lb,
                        mutation_rate=mr
                    )

                    # Store the results with metadata
                    results_by_crossover[method].append({
                        "fitness_mode": fm,
                        "lcs_bonus": lb,
                        "mutation_rate": mr,
                        "best_fitness_history": run_data["best_fitness_history"],
                        "converged_generation": run_data["converged_generation"]
                    })

    # Now we produce a single figure with 3 subplots (one per crossover method).
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, method in enumerate(crossover_methods):
        ax = axes[i]
        ax.set_title(f"Crossover: {method}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")

        # Sort the results by converged_generation (fewest to largest)
        method_results = sorted(
            results_by_crossover[method],
            key=lambda d: d["converged_generation"]
        )

        # We want 2 best (lowest generations) and 2 worst (highest generations)
        best_2 = method_results[:2]
        worst_2 = method_results[-2:]

        # Combine them for plotting
        lines_to_plot = best_2 + worst_2

        for run_info in lines_to_plot:
            fm = run_info["fitness_mode"]
            lb = run_info["lcs_bonus"]
            mr = run_info["mutation_rate"]
            conv_gen = run_info["converged_generation"]
            best_fit_hist = run_info["best_fitness_history"]

            # X-axis = generation indices
            generations = list(range(len(best_fit_hist)))

            # Legend label with # of gens to converge
            legend_label = f"{fm}, lb={lb}, mut={mr} => {conv_gen} gens"

            ax.plot(generations, best_fit_hist, label=legend_label, linewidth=2)

        ax.legend()
        ax.grid(True)

    # Show the entire figure with all 3 subplots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_ga_combinations()
