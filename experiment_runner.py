import sol
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def run_experiment(lcs_bonus, crossover_method, mutation_rate, fitness_mode, max_iter=150):
    """
    Wrapper function to run sol.run_ga with specific parameters and extract needed data
    """
    # Run the GA with specified parameters
    results = sol.run_ga(
        crossover_method=crossover_method, 
        fitness_mode=fitness_mode, 
        lcs_bonus=lcs_bonus, 
        mutation_rate=mutation_rate,
        population_size=500,  # Default from sol.py
        max_runtime=120,      # Default from sol.py
        distance_metric="levenshtein"  # Default from sol.py
    )
    
    # Extract relevant data
    best_history = results["best_fitness_history"]
    final_iteration = results["converged_generation"]
    
    return best_history, final_iteration

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
    lcs_bonuses = [1,5]#, 5, 10, 20, 50]
    mutation_rates = [0.7,0.25, 0.5]
    fitness_modes = ["ascii"]#, "lcs", "combined"]
    crossover_methods = ["single", "two_point", "uniform"]

    # Prepare results structure
    # Format: {method: {combo: {"history": [...], "metrics": {...}}}}
    results = {method: {} for method in crossover_methods}

    # Run experiments
    for method in crossover_methods:
        print(f"\n--- TESTING CROSSOVER METHOD: {method} ---\n")
        
        for lcs_bonus in lcs_bonuses:
            for mut_rate in mutation_rates:
                for fit_mode in fitness_modes:
                    # Skip irrelevant combinations
                    if fit_mode != "combined" and lcs_bonus != 5:
                        continue
                        
                    combo = (lcs_bonus, mut_rate, fit_mode)
                    print(f"Running GA with method={method}, LCS bonus={lcs_bonus}, "
                          f"mutation rate={mut_rate}, fitness mode={fit_mode}")
                    
                    # Run GA and track metrics
                    history, final_iter = run_experiment(lcs_bonus, method, mut_rate, fit_mode)
                    metrics = calculate_metrics(history, final_iter)
                    
                    # Store results
                    results[method][combo] = {
                        "history": history,
                        "metrics": metrics
                    }
                    
                    # Print summary
                    print(f"  Final fitness: {metrics['final_fitness']}, "
                          f"Iterations: {final_iter}, "
                          f"{'SOLVED!' if metrics['solved'] else 'Not solved'}")

    # Plot results for each crossover method
    for method in crossover_methods:
        # Find best combinations based on combined score
        combos_by_score = sorted(
            results[method].items(),
            key=lambda x: x[1]["metrics"]["combined_score"]
        )
        
        # Get 3 best and 3 worst combinations
        best_combos = [combo for combo, _ in combos_by_score[:3]]
        worst_combos = [combo for combo, _ in combos_by_score[-3:]]
        selected_combos = best_combos + worst_combos
        
        # Create plot with more space for legend
        plt.figure(figsize=(14, 8))
        
        # Plot only selected combinations
        for combo in selected_combos:
            history = results[method][combo]["history"]
            metrics = results[method][combo]["metrics"]
            
            label_str = (f"LCS={combo[0]}, MUT={combo[1]:.2f}, FIT={combo[2]}, "
                        f"Final={metrics['final_fitness']}")
            
            # Add marker to differentiate best vs worst
            if combo in best_combos:
                plt.plot(history, label=f"BEST: {label_str}", 
                         linestyle='-', linewidth=2, marker='o', markevery=50)
            else:
                plt.plot(history, label=f"WORST: {label_str}", 
                         linestyle='--', linewidth=1, marker='x', markevery=50)
                
        plt.title(f"Best vs Worst Parameter Combinations ({method} crossover)", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Best Fitness (lower is better)", fontsize=12)
        plt.grid(True)
        plt.yscale('log')  # Use log scale to better see differences
        
        # Place legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        
        # Save the figure
        plt.savefig(f"{results_dir}/{method}_comparison.png", dpi=300)
        plt.show()
        
        # Create a summary table for this method
        print(f"\nSummary for {method} crossover:")
        print("-" * 80)
        print(f"{'Parameters':<25} {'Final':<10} {'Iterations':<12} {'Conv.Speed':<12} {'Status':<10}")
        print("-" * 80)
        
        for combo, data in combos_by_score:
            metrics = data["metrics"]
            param_str = f"LCS={combo[0]}, MUT={combo[1]}, FIT={combo[2]}"
            status = "SOLVED" if metrics["solved"] else "Not solved"
            
            print(f"{param_str:<25} {metrics['final_fitness']:<10.2f} "
                  f"{metrics['iterations']:<12} {metrics['convergence_at']:<12} {status:<10}")

if __name__ == "__main__":
    main()
