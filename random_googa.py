import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Grid settings
cell_size = 20  # Size of each cell in pixels
grid_width = 120  # Number of cells horizontally
grid_height = 120  # Number of cells vertically

# Window setup
screen = pygame.display.set_mode((grid_width * cell_size, grid_height * cell_size))
pygame.display.set_caption("Conway's Game of Life")
clock = pygame.time.Clock()

# Create a random grid: 0 for dead cells, 1 for live cells
grid = np.random.randint(2, size=(grid_height, grid_width))


def update_grid(current_grid):
    """
    Update the grid based on Conway's Game of Life rules:
      - Any live cell with fewer than two live neighbours dies (underpopulation).
      - Any live cell with two or three live neighbours lives on.
      - Any live cell with more than three live neighbours dies (overpopulation).
      - Any dead cell with exactly three live neighbours becomes a live cell (reproduction).
    """
    new_grid = current_grid.copy()
    for i in range(grid_height):
        for j in range(grid_width):
            # Count live neighbors using periodic boundary conditions (wrap-around)
            total = (
                    current_grid[(i - 1) % grid_height, (j - 1) % grid_width] +
                    current_grid[(i - 1) % grid_height, j] +
                    current_grid[(i - 1) % grid_height, (j + 1) % grid_width] +
                    current_grid[i, (j - 1) % grid_width] +
                    current_grid[i, (j + 1) % grid_width] +
                    current_grid[(i + 1) % grid_height, (j - 1) % grid_width] +
                    current_grid[(i + 1) % grid_height, j] +
                    current_grid[(i + 1) % grid_height, (j + 1) % grid_width]
            )
            # Apply the rules of the game
            if current_grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0  # Die
            else:
                if total == 3:
                    new_grid[i, j] = 1  # Become alive
    return new_grid


# Main loop
running = True
while running:
    # Handle events (like closing the window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the grid state
    grid = update_grid(grid)

    # Draw the grid: fill dead cells with black and live cells with white
    screen.fill(pygame.Color("black"))
    for i in range(grid_height):
        for j in range(grid_width):
            if grid[i, j] == 1:
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, pygame.Color("white"), rect)

    # Refresh the display
    pygame.display.flip()

    # Control the frame rate (adjust for desired simulation speed)
    clock.tick(10)

pygame.quit()
