import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Global constants you can override in run_ga():
# -------------------------------------------------
GA_POPSIZE       = 8000   # massive population size for better exploration
GA_MAXITER       = 500    # plenty of iterations to find solution
GA_ELITRATE      = 0.10   # keep the top 10% elite candidates
GA_MUTATIONRATE  = 0.55   # high mutation rate to avoid local optima
GA_TARGET        = "testing string123 diff_chars"  # target string we're evolving toward
GA_CROSSOVER_METHOD = "single"   # crossover type: "single", "two_point", or "uniform"
GA_LCS_BONUS     = 5     # weight factor for LCS in combined fitness
GA_FITNESS_MODE  = "ascii"  # fitness mode: "ascii", "lcs", or "combined"

# -------------------------------------------------
# Represents a single solution in our population
# -------------------------------------------------
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene   # candidate's genetic sequence
        self.fitness = fitness

# -------------------------------------------------
# Initialize random population and buffer
# -------------------------------------------------
def init_population():
    target_length = len(GA_TARGET)
    population = []

    for _ in range(GA_POPSIZE):
        # generate random ascii chars between space (32) and 'y' (121)
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))

    # buffer will hold the next generation
    buffer = [Candidate('', 0) for _ in range(GA_POPSIZE)]
    return population, buffer

# -------------------------------------------------
# Calculate the Longest Common Subsequence (LCS)
# -------------------------------------------------
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# -------------------------------------------------
# Calculate fitness for each candidate
# -------------------------------------------------
def calc_fitness(population):
    target = GA_TARGET
    target_length = len(target)

    for candidate in population:
        if GA_FITNESS_MODE == "ascii":
            # sum of absolute ascii differences
            fitness = sum(abs(ord(candidate.gene[i]) - ord(target[i])) for i in range(target_length))

        elif GA_FITNESS_MODE == "lcs":
            # LCS-based fitness (larger LCS => better, so invert by subtracting from target_length)
            lcs_len = longest_common_subsequence(candidate.gene, target)
            fitness = target_length - lcs_len

        elif GA_FITNESS_MODE == "combined":
            # combined ascii + LCS (weighted)
            ascii_fitness = sum(abs(ord(candidate.gene[i]) - ord(target[i])) for i in range(target_length))
            lcs_len = longest_common_subsequence(candidate.gene, target)
            # invert LCS, then apply bonus
            lcs_fitness = target_length - lcs_len
            fitness = ascii_fitness + GA_LCS_BONUS * lcs_fitness

        else:
            # default to ascii difference if mode is invalid
            fitness = sum(abs(ord(candidate.gene[i]) - ord(target[i])) for i in range(target_length))

        candidate.fitness = fitness

# -------------------------------------------------
# Sort by fitness (ascending); lower is better
# -------------------------------------------------
def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)

# -------------------------------------------------
# Preserve the top 'elite_size' solutions
# -------------------------------------------------
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        buffer[i] = Candidate(population[i].gene, population[i].fitness)

# -------------------------------------------------
# Randomly alter one character in the gene
# -------------------------------------------------
def mutate(candidate):
    target_length = len(GA_TARGET)
    pos = random.randint(0, target_length - 1)
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    new_val = 32 + ((old_val - 32 + delta) % (121 - 32 + 1))

    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)

# -------------------------------------------------
# Single-point crossover
# -------------------------------------------------
def single_point_crossover(parent1, parent2, length):
    crossover_point = random.randint(0, length - 1)
    child1 = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    child2 = parent2.gene[:crossover_point] + parent1.gene[crossover_point:]
    return child1, child2

# -------------------------------------------------
# Two-point crossover
# -------------------------------------------------
def two_point_crossover(parent1, parent2, length):
    point1, point2 = sorted(random.sample(range(length), 2))
    child1 = (parent1.gene[:point1] +
              parent2.gene[point1:point2] +
              parent1.gene[point2:])
    child2 = (parent2.gene[:point1] +
              parent1.gene[point1:point2] +
              parent2.gene[point2:])
    return child1, child2

# -------------------------------------------------
# Uniform crossover
# -------------------------------------------------
def uniform_crossover(parent1, parent2, length):
    child1_gene = []
    child2_gene = []
    for i in range(length):
        if random.random() < 0.5:
            child1_gene.append(parent1.gene[i])
            child2_gene.append(parent2.gene[i])
        else:
            child1_gene.append(parent2.gene[i])
            child2_gene.append(parent1.gene[i])
    return ''.join(child1_gene), ''.join(child2_gene)

# -------------------------------------------------
# Breed next generation
# -------------------------------------------------
def mate(population, buffer):
    elite_size = int(GA_POPSIZE * GA_ELITRATE)
    target_length = len(GA_TARGET)

    # Keep the top elites
    elitism(population, buffer, elite_size)

    # Fill the rest of the buffer by crossing pairs from top half
    for i in range(elite_size, GA_POPSIZE - 1, 2):
        parent1 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        parent2 = population[random.randint(0, GA_POPSIZE // 2 - 1)]

        if GA_CROSSOVER_METHOD == "single":
            child1, child2 = single_point_crossover(parent1, parent2, target_length)
        elif GA_CROSSOVER_METHOD == "two_point":
            child1, child2 = two_point_crossover(parent1, parent2, target_length)
        elif GA_CROSSOVER_METHOD == "uniform":
            child1, child2 = uniform_crossover(parent1, parent2, target_length)
        else:
            # default to single if invalid
            child1, child2 = single_point_crossover(parent1, parent2, target_length)

        buffer[i] = Candidate(child1)
        buffer[i + 1] = Candidate(child2)

        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i + 1])

    # Edge case for odd population
    if GA_POPSIZE % 2 == 1 and elite_size % 2 == 0:
        buffer[-1] = Candidate(buffer[-2].gene)
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[-1])

# -------------------------------------------------
# Utility: Print best candidate for debugging
# -------------------------------------------------
def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")

# -------------------------------------------------
# Swap population and buffer
# -------------------------------------------------
def swap(pop1, pop2):
    return pop2, pop1

# -------------------------------------------------
# (Optional) Plotting Functions
# -------------------------------------------------
def plot_fitness_evolution(best_history, mean_history, worst_history):
    generations = range(len(best_history))
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_history, label="best", linewidth=2)
    plt.plot(generations, mean_history, label="mean", linewidth=2)
    plt.plot(generations, worst_history, label="worst", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness evolution (crossover: {GA_CROSSOVER_METHOD})")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitness_boxplots(fitness_distributions):
    plt.figure(figsize=(14, 8))
    # Basic boxplot (example; advanced styling removed for brevity)
    plt.boxplot(fitness_distributions, patch_artist=True)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness distribution (crossover: {GA_CROSSOVER_METHOD})")
    plt.grid(True)
    plt.show()

# -------------------------------------------------
# New function to run GA with given parameters
# -------------------------------------------------
def run_ga(crossover_method, fitness_mode, lcs_bonus, mutation_rate):
    """
    Run the GA with the provided parameters. Returns a dict with:
      - best_fitness_history: list of best fitness per generation
      - converged_generation: which generation got best fitness==0
    """

    # Overwrite the global constants with the function arguments
    global GA_CROSSOVER_METHOD, GA_FITNESS_MODE, GA_LCS_BONUS, GA_MUTATIONRATE
    GA_CROSSOVER_METHOD = crossover_method
    GA_FITNESS_MODE = fitness_mode
    GA_LCS_BONUS = lcs_bonus
    GA_MUTATIONRATE = mutation_rate

    random.seed(time.time())
    population, buffer = init_population()

    best_fitness_history = []
    converged_generation = GA_MAXITER  # assume no convergence unless proven otherwise

    for iteration in range(GA_MAXITER):
        calc_fitness(population)
        sort_by_fitness(population)

        # Check best fitness
        best_fit = population[0].fitness
        best_fitness_history.append(best_fit)

        if best_fit == 0:
            # We converged on the target
            converged_generation = iteration
            break

        mate(population, buffer)
        population, buffer = swap(population, buffer)

    return {
        "best_fitness_history": best_fitness_history,
        "converged_generation": converged_generation
    }

# -------------------------------------------------
# Optional main() for single-run testing
# -------------------------------------------------
def main():
    """
    This main() runs the GA once, using global params
    (GA_*). It demonstrates a single run with the
    existing settings. If you want to test many combos,
    please use the separate test script.
    """
    results = run_ga(
        crossover_method=GA_CROSSOVER_METHOD,
        fitness_mode=GA_FITNESS_MODE,
        lcs_bonus=GA_LCS_BONUS,
        mutation_rate=GA_MUTATIONRATE
    )

    # Print final data
    print(f"Converged generation: {results['converged_generation']}")
    print(f"Best fitness history length: {len(results['best_fitness_history'])}")

    # If you want to do any single-run plotting, you can do it here:
    # e.g., just plot best fitness:
    import matplotlib.pyplot as plt
    plt.plot(results["best_fitness_history"])
    plt.title("Best Fitness Per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()


if __name__ == "__main__":
    main()
