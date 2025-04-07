import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_fitness_landscape():
    # Create a high-resolution grid of points for gene1 (x) and gene2 (y)
    gene1 = np.linspace(-10, 10, 500)  # Gene 1 range
    gene2 = np.linspace(-10, 10, 500)  # Gene 2 range
    X, Y = np.meshgrid(gene1, gene2)

    # Define the fitness function - lowest when Gene 1 ≈ Gene 2
    Fitness = 1000 - (10 * np.abs(X - Y) + 5 * np.sin(5 * (X + Y)))
    
    # Add randomness to create a rugged, noisy surface
    randomness = np.random.normal(0, 50, X.shape)
    Fitness += randomness
    
    # Set a minimum "floor" value to ensure we don't go below a certain threshold
    Fitness = np.clip(Fitness, 0, 1000)  # Floor value of 0 and ceiling value of 1000

    # Create a figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with fitness as the Z value
    surf = ax.plot_surface(
        X, Y, Fitness,
        cmap=cm.plasma,  # Changed colormap to 'plasma'
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True
    )

    # Add color bar for the fitness scale
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('Fitness')

    # Labels and title
    ax.set_xlabel('Gene 1')
    ax.set_ylabel('Gene 2')
    ax.set_title('Fitness Landscape')

    # Set axis ranges
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 1000])  # Updated z-axis limit

    # Enable grid lines
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    plot_fitness_landscape()
