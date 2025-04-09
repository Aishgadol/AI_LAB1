import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sol import run_ga, init_population, calc_fitness, sort_by_fitness

def run_engine_and_plot_gene_distance():
    # Run the GA once with default parameters
    results = run_ga(
        crossover_method="two_point",
        fitness_mode="combined", 
        lcs_bonus=10,
        mutation_rate=0.55,
        population_size=755,  # Smaller population for faster execution
        max_runtime=60        # Limit runtime to 60 seconds
    )
    
    # Extract the number of generations the algorithm ran
    num_generations = len(results["best_fitness_history"])
    print(f"GA ran for {num_generations} generations")
    
    # Create a grid representing gene1 and gene2 values over generations
    # For this example, we'll use a range of values from -10 to 10 for both genes
    gene1_range = np.linspace(-10, 10, 50)
    gene2_range = np.linspace(-10, 10, 50)
    
    # Create meshgrid for 3D surface
    X, Y = np.meshgrid(gene1_range, gene2_range)
    
    # Initialize distance matrix
    Z = np.zeros_like(X)
    
    # Calculate the distance between gene1 and gene2 combinations
    # This is a simplified model where distance is related to:
    # 1. The absolute difference between genes
    # 2. The generation number (to show evolution over time)
    # 3. The actual GA performance to influence the landscape
    
    # Normalize the fitness history to use as weights
    if num_generations > 0:
        fitness_history = np.array(results["best_fitness_history"])
        normalized_fitness = 1 - (fitness_history / max(fitness_history) if max(fitness_history) > 0 else fitness_history)
        
        # Use the final entropy and distance metrics to shape our landscape
        final_entropy = results["entropy_history"][-1] if results["entropy_history"] else 0
        final_distance = results["distance_history"][-1] if results["distance_history"] else 0
        
        # Create a landscape influenced by the GA performance
        for i, x in enumerate(gene1_range):
            for j, y in enumerate(gene2_range):
                # Base formula: distance affected by gene differences and GA performance
                Z[j][i] = 1000 - (10 * np.abs(x - y) + 5 * np.sin(5 * (x + y)))
                
                # Add influence from GA metrics
                Z[j][i] *= (1 + 0.2 * final_entropy)
                Z[j][i] *= (1 + 0.1 * final_distance)
                
                # Add some noise for a more realistic landscape
                Z[j][i] += np.random.normal(0, 50)
        
        # Ensure minimum and maximum values
        Z = np.clip(Z, 0, 1000)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.viridis,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True
    )
    
    # Add color bar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('Distance/Fitness')
    
    # Labels and title
    ax.set_xlabel('Gene 1')
    ax.set_ylabel('Gene 2')
    ax.set_zlabel('Distance')
    ax.set_title('3D Map of Distances Between Gene 1 and Gene 2\nInfluenced by GA Performance')
    
    # Set axis ranges
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 1000])
    
    # Enable grid
    ax.grid(True)
    
    # Add a text annotation about the GA performance
    textstr = f"GA ran for {num_generations} generations\n"
    textstr += f"Termination: {results['termination_reason']}\n"
    textstr += f"Final entropy: {final_entropy:.2f}\n"
    textstr += f"Final distance: {final_distance:.2f}"
    
    fig.text(0.02, 0.02, textstr, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()

def plot_fitness_landscape():
    # ... existing code ...
    pass

if __name__ == "__main__":
    # Run the engine and plot gene distance instead of the original function
    run_engine_and_plot_gene_distance()
