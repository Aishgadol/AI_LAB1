import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_fitness_landscape():
    # Create a grid of points for gene1 (x) and gene2 (y)
    gene1 = np.linspace(0, 10, 50)  # Gene 1 range
    gene2 = np.linspace(0, 10, 50)  # Gene 2 range
    X, Y = np.meshgrid(gene1, gene2)

    # Let's define a simple time function: time = X + Y
    # You can replace this with any function that makes sense for your data
    Z = X + Y

    # Define the fitness function; here we map higher values (near 1000) down to 0
    # by subtracting a factor that grows with X + Y.
    # You could use your own real-world function or data array here.
    Fitness = 1000 - 50 * (X + Y)

    # Normalize the fitness values so we can map them to colors
    fitness_min, fitness_max = Fitness.min(), Fitness.max()
    norm = (Fitness - fitness_min) / (fitness_max - fitness_min)

    # Create a figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface; color is based on Fitness
    # Use facecolors=... so that Z is used for height, and Fitness sets the color
    ax.plot_surface(
        X, Y, Z,
        facecolors=cm.viridis(norm),
        rstride=1,
        cstride=1
    )

    # Create a scalar mappable to add a colorbar
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_clim(fitness_min, fitness_max)
    # We don't pass real data to the mappable, so just set_array([]).
    mappable.set_array([])

    # Add color bar for the fitness scale
    cbar = fig.colorbar(mappable, shrink=0.5, aspect=10)
    cbar.set_label('Fitness')

    # Labels and title
    ax.set_xlabel('Gene 1')
    ax.set_ylabel('Gene 2')
    ax.set_zlabel('Time')
    ax.set_title('Fitness Landscape')

    plt.show()


if __name__ == "__main__":
    plot_fitness_landscape()
