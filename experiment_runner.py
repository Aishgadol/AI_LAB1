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
    
    # Define parameter ranges to test
    selection_methods = ["rws", "sus", "tournament_deterministic", "tournament_probabilistic"]
    linear_scalings = [True, False]

    # Prepare results structure
    results = {}

    # Run experiments in plain loops
    for sel_method in selection_methods:
        results[sel_method] = {}
        for lin_scaling in linear_scalings:
            best_history, mean_history, final_iter = run_experiment(
                5,  # lcs_bonus changed to 5
                "two_point",
                0.55,
                "combined",
                sel_method,
                lin_scaling,
                250
            )
            metrics = calculate_metrics(best_history, final_iter)
            results[sel_method][lin_scaling] = {
                "mean_history": mean_history,
                "metrics": metrics
            }

    # Plot results for each selection method
    for sel_method in selection_methods:
        # Find best combinations based on combined score
        combos_by_score = sorted(
            results[sel_method].items(),
            key=lambda x: x[1]["metrics"]["combined_score"]
        )
        
        # Get 2 best and 2 worst combinations
        best_combos = [combo for combo, _ in combos_by_score[:2]]
        worst_combos = [combo for combo, _ in combos_by_score[-2:]]
        selected_combos = best_combos + worst_combos
        
        # Create plot with more space for legend
        plt.figure(figsize=(14, 8))
        
        # Plot only selected combinations
        for combo in selected_combos:
            mean_history = results[sel_method][combo]["mean_history"]
            metrics = results[sel_method][combo]["metrics"]
            
            label_str = (f"Linear Scaling={combo}, "
                        f"Final={metrics['final_fitness']}")
            
            # Add marker to differentiate best vs worst
            if combo in best_combos:
                plt.plot(mean_history, label=f"BEST: {label_str}", 
                         linestyle='-', linewidth=2, marker='o', markevery=25)
            else:
                plt.plot(mean_history, label=f"WORST: {label_str}", 
                         linestyle='--', linewidth=1, marker='x', markevery=25)
                
        plt.title(f"Best vs Worst Mean Fitness ({sel_method} selection)", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Average Fitness (lower is better)", fontsize=12)
        plt.grid(True)
        plt.yscale('log')  # Use log scale to better see differences
        
        # Place legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        
        # Save the figure
        plt.savefig(f"{results_dir}/{sel_method}_comparison.png", dpi=300)
        plt.show()
        
        # Create a summary table for this method
        print(f"\nSummary for {sel_method} selection:")
        print("-" * 80)
        print(f"{'Parameters':<25} {'Final':<10} {'Iterations':<12} {'Conv.Speed':<12} {'Status':<10}")
        print("-" * 80)
        
        for combo, data in combos_by_score:
            metrics = data["metrics"]
            param_str = f"Linear Scaling={combo}"
            status = "SOLVED" if metrics["solved"] else "Not solved"
            
            print(f"{param_str:<25} {metrics['final_fitness']:<10.2f} "
                  f"{metrics['iterations']:<12} {metrics['convergence_at']:<12} {status:<10}")

if __name__ == "__main__":
    main()
