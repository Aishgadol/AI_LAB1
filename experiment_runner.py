import new_sol as sol
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def run_experiment(
    lcs_bonus, crossover_method, mutation_rate, fitness_mode, 
    selection_method, use_linear_scaling, max_iter=170
):
    """
    Updated wrapper to run sol.run_ga with new parameters 
    """
    sol.ga_maxiter = max_iter
    # Run the GA with specified parameters
    results = sol.run_ga(
        crossover_method=crossover_method, 
        fitness_mode=fitness_mode, 
        lcs_bonus=lcs_bonus, 
        mutation_rate=mutation_rate,
        population_size=1000,
        max_runtime=6000,
        distance_metric="levenshtein",
        selection_method=selection_method,
        use_linear_scaling=use_linear_scaling,
        max_fitness_ratio=2.0,
        use_aging=True,
        age_limit=100,
        tournament_k=3,
        tournament_k_prob=3,
        tournament_p=0.75
    )
    
    # Extract relevant data
    best_history = results["best_fitness_history"]
    mean_history = results["mean_fitness_history"]
    final_iteration = results["converged_generation"]
    
    return best_history, mean_history, final_iteration

def calculate_metrics(history, final_iter):
    # Calculate various performance metrics
    final_fitness = history[-1]
    
    # Early convergence (lower is better)
    convergence_speed = 0
    for i, fitness in enumerate(history):
        if fitness <= final_fitness * 1.1:  # Within 10% of final
            convergence_speed = i
            break
    
    # Calculate improvement rate (higher is better)
    if len(history) > 1:
        initial_fitness = history[0]
        improvement_rate = (initial_fitness - final_fitness) / len(history)
    else:
        improvement_rate = 0
        
    # Area under curve (lower is better) - normalized
    auc = sum(history) / len(history)
    
    # Combined score (lower is better)
    combined_score = final_fitness + 0.2 * auc + 0.1 * convergence_speed
    
    return {
        "final_fitness": final_fitness,
        "convergence_at": convergence_speed,
        "improvement_rate": improvement_rate,
        "auc": auc,
        "combined_score": combined_score,
        "solved": final_fitness == 0,
        "iterations": final_iter
    }

def main():
    # Create results directory if it doesn't exist
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}

    # RWS & SUS with or without linear scaling
    for sel_method in ["rws", "sus"]:
        results[sel_method] = {}
        for lin_scaling in [True, False]:
            best_history, mean_history, final_iter = run_experiment(
                5,
                "two_point",
                0.55,
                "combined",
                sel_method,
                lin_scaling,
                250
            )
            metrics = calculate_metrics(best_history, final_iter)
            results[sel_method][f"linScaling={lin_scaling}"] = {
                "mean_history": mean_history,
                "metrics": metrics
            }

    # deterministic tournament (k in [2,3,4])
    results["tournament_deterministic"] = {}
    for k_val in [2, 3, 4]:
        best_history, mean_history, final_iter = run_experiment(
            5,
            "two_point",
            0.55,
            "combined",
            "tournament_deterministic",
            False,
            250,
        )
        sol.ga_tournament_k = k_val
        metrics = calculate_metrics(best_history, final_iter)
        results["tournament_deterministic"][f"k={k_val}"] = {
            "mean_history": mean_history,
            "metrics": metrics
        }

    # probabilistic tournament (k in [2,3,4], p in [0.1, 0.4, 0.75])
    results["tournament_probabilistic"] = {}
    for k_val in [2, 3, 4]:
        for p_val in [0.1, 0.4, 0.75]:
            best_history, mean_history, final_iter = run_experiment(
                5,
                "two_point",
                0.55,
                "combined",
                "tournament_probabilistic",
                False,
                250,
            )
            sol.ga_tournament_k_prob = k_val
            sol.ga_tournament_p = p_val
            metrics = calculate_metrics(best_history, final_iter)
            results["tournament_probabilistic"][f"k={k_val},p={p_val}"] = {
                "mean_history": mean_history,
                "metrics": metrics
            }

    # Create one big figure with 4 graphs
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, sel_method in enumerate(results.keys()):
        ax = axes[i]
        for combo, data in results[sel_method].items():
            mean_history = data["mean_history"]
            metrics = data["metrics"]
            label_str = f"{combo}, Final={metrics['final_fitness']}"
            ax.plot(mean_history, label=label_str)
        ax.set_title(f"{sel_method} Selection", fontsize=12)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Average Fitness (lower=better)", fontsize=10)
        ax.legend()
        ax.grid(True)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/all_selection_methods.png", dpi=300)
    plt.show()

    # Create a summary table for each selection method
    for sel_method in results.keys():
        print(f"\nSummary for {sel_method} selection:")
        print("-" * 80)
        print(f"{'Parameters':<25} {'Final':<10} {'Iterations':<12} {'Conv.Speed':<12} {'Status':<10}")
        print("-" * 80)
        
        combos_by_score = sorted(
            results[sel_method].items(),
            key=lambda x: x[1]["metrics"]["combined_score"]
        )
        
        for combo, data in combos_by_score:
            metrics = data["metrics"]
            param_str = f"{combo}"
            status = "SOLVED" if metrics["solved"] else "Not solved"
            
            print(f"{param_str:<25} {metrics['final_fitness']:<10.2f} "
                  f"{metrics['iterations']:<12} {metrics['convergence_at']:<12} {status:<10}")

if __name__ == "__main__":
    main()

