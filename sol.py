import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

#constant params for genetic algorithm
GA_POPSIZE = 2*8192  #massive population size for better exploration
GA_MAXITER = 16384  #plenty of iterations to find solution
GA_ELITRATE = 0.10  #keep the top 10% elite candidates
GA_MUTATIONRATE = 0.55  #high mutation rate to avoid local optima
GA_TARGET = "impossible to converge, but ill try "  #target string we're evolving toward
GA_CROSSOVER_METHOD = "uniform"  #crossover type: "single", "two_point", or "uniform"
GA_LCS_BONUS = 10  #weight factor for LCS in combined fitness
GA_FITNESS_MODE = "combined"  #fitness mode: "ascii", "lcs", or "combined"

#represents a single solution in our population
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene  #candidate's genetic sequence
        self.fitness = fitness  #how close it is to the target (lower = better)

#create initial random population and buffer
def init_population():
    target_length = len(GA_TARGET)
    population = []
    #fill population with random candidates
    for _ in range(GA_POPSIZE):
        #generate random ascii chars between space and 'y'
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    #buffer holds next generation
    buffer = [Candidate('') for _ in range(GA_POPSIZE)]
    return population, buffer

# Calculate the longest common subsequence between two strings
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    # Create a table to store LCS lengths
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Return length of LCS
    return dp[m][n]

#calculate fitness score for each candidate
def calc_fitness(population):
    target = GA_TARGET
    target_length = len(target)
    
    for candidate in population:
        if GA_FITNESS_MODE == "ascii":
            # Original ASCII difference method
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
        
        elif GA_FITNESS_MODE == "lcs":
            # LCS method (higher LCS is better, so we invert)
            lcs_length = longest_common_subsequence(candidate.gene, target)
            fitness = target_length - lcs_length  # Invert so lower is better
        
        elif GA_FITNESS_MODE == "combined":
            # Combine both methods
            # ASCII difference part
            ascii_fitness = 0
            for i in range(target_length):
                ascii_fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
            
            # LCS part
            lcs_length = longest_common_subsequence(candidate.gene, target)
            lcs_fitness = target_length - lcs_length  # Invert so lower is better
            
            # Combine with weighted LCS component
            fitness = ascii_fitness + GA_LCS_BONUS * lcs_fitness
        
        else:
            # Default to ASCII difference if mode is invalid
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
            
        candidate.fitness = fitness

#sort candidates by fitness (lower is better)
def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)

#preserve best solutions for next generation
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        #direct copy of best genes to next gen
        buffer[i] = Candidate(population[i].gene, population[i].fitness)

#randomly alter one character in candidate's gene
def mutate(candidate):
    target_length = len(GA_TARGET)
    pos = random.randint(0, target_length - 1)  #pick random position to mutate
    #random shift in ascii value
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    #wrap around if we exceed printable chars
    new_val = 32 + ((old_val - 32 + delta) % (121 - 32 + 1))    #strings are immutable so convert to list first
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)

#different crossover implementations
def single_point_crossover(parent1, parent2, target_length):
    crossover_point = random.randint(0, target_length - 1)
    #child1 gets first part from parent1, second from parent2
    child1 = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    #child2 gets first part from parent2, second from parent1
    child2 = parent2.gene[:crossover_point] + parent1.gene[crossover_point:]
    return child1, child2


def two_point_crossover(parent1, parent2, target_length):
    #get two distinct crossover points
    point1, point2 = sorted(random.sample(range(target_length), 2))
    
    #child1: parent1's outer parts, parent2's middle
    child1 = (parent1.gene[:point1] + 
              parent2.gene[point1:point2] + 
              parent1.gene[point2:])
    
    #child2: parent2's outer parts, parent1's middle
    child2 = (parent2.gene[:point1] + 
              parent1.gene[point1:point2] + 
              parent2.gene[point2:])
    
    return child1, child2


def uniform_crossover(parent1, parent2, target_length):
    child1_gene = []
    child2_gene = []
    
    for i in range(target_length):
        if random.random() < 0.5:
            #flip coin for each position
            child1_gene.append(parent1.gene[i])
            child2_gene.append(parent2.gene[i])
        else:
            child1_gene.append(parent2.gene[i])
            child2_gene.append(parent1.gene[i])
    
    return ''.join(child1_gene), ''.join(child2_gene)

#create next generation through selection, crossover and mutation
def mate(population, buffer):
    elite_size = int(GA_POPSIZE * GA_ELITRATE)
    target_length = len(GA_TARGET)
    
    #transfer elite candidates directly
    elitism(population, buffer, elite_size)
    
    #breed remaining candidates
    for i in range(elite_size, GA_POPSIZE - 1, 2):  #step by 2 for pairs
        #tournament selection from top half
        parent1 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        parent2 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        
        #apply selected crossover method
        if GA_CROSSOVER_METHOD == "single":
            child1, child2 = single_point_crossover(parent1, parent2, target_length)
        elif GA_CROSSOVER_METHOD == "two_point":
            child1, child2 = two_point_crossover(parent1, parent2, target_length)
        elif GA_CROSSOVER_METHOD == "uniform":
            child1, child2 = uniform_crossover(parent1, parent2, target_length)
        else:
            child1, child2 = single_point_crossover(parent1, parent2, target_length)

        child1,child2=parent1.gene, parent2.gene
        #add children to next gen
        buffer[i] = Candidate(child1)
        buffer[i + 1] = Candidate(child2)
        
        #chance to mutate each child
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i + 1])
    
    #handle edge case for odd population size
    if GA_POPSIZE % 2 == 1 and elite_size % 2 == 0:
        buffer[-1] = Candidate(buffer[-2].gene)  #copy penultimate solution
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[-1])

#output the best solution so far
def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")

#swap population and buffer for next generation
def swap(population, buffer):
    return buffer, population

#calculate population stats for analysis
def compute_fitness_statistics(population):
    #extract all fitness values
    fitness_values = [cand.fitness for cand in population]
    #basic statistical measures
    mean_fitness = sum(fitness_values) / len(fitness_values)
    #variance calculation
    variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / len(fitness_values)
    std_fitness = math.sqrt(variance)
    #best and worst fitness
    best_fitness = population[0].fitness
    worst_fitness = population[-1].fitness
    #range shows convergence
    fitness_range = worst_fitness - best_fitness
    #pack everything in a dict
    return {
        "mean": mean_fitness,
        "std": std_fitness,
        "worst_fitness": worst_fitness,
        "fitness_range": fitness_range,
        "worst_candidate": population[-1]
    }

#track performance metrics
def compute_timing_metrics(generation_start_cpu, overall_start_wall):
    #get current timestamps
    current_cpu = time.process_time()
    current_wall = time.time()
    #calculate time deltas
    generation_cpu_time = current_cpu - generation_start_cpu
    elapsed_time = current_wall - overall_start_wall
    #return timing data
    return {
        "generation_cpu_time": generation_cpu_time,
        "elapsed_time": elapsed_time
    }

def plot_fitness_evolution(best_history, mean_history, worst_history):
    generations = list(range(len(best_history)))
    plt.figure(figsize=(12, 6))  #larger figure for clarity
    #plot fitness trends
    plt.plot(generations, best_history, label="best", linewidth=2)
    plt.plot(generations, mean_history, label="mean", linewidth=2)
    plt.plot(generations, worst_history, label="worst", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness evolution over generations (crossover: {GA_CROSSOVER_METHOD})")
    plt.legend()
    plt.grid(True)
    plt.show()  #display the plot

def plot_fitness_boxplots(fitness_distributions):
    plt.figure(figsize=(14, 8))  #bigger figure for detail
    #fancy markers for outliers
    flierprops = dict(marker='D', markersize=4, linestyle='none', markeredgecolor='blue')
    #styling for better viz
    boxprops = dict(facecolor='lightblue', color='blue', linewidth=1.5)
    whiskerprops = dict(color='blue', linewidth=1.5)
    capprops = dict(color='blue', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)
    total = len(fitness_distributions)
    if total > 10:
        #sample 10 generations evenly
        indices = [int(round(i * (total - 1) / 9)) for i in range(10)]
        sampled_distributions = [fitness_distributions[i] for i in indices]
        xtick_labels = [str(i) for i in indices]
    else:
        indices = list(range(total))
        sampled_distributions = fitness_distributions
        xtick_labels = [str(i) for i in indices]
    bp = plt.boxplot(sampled_distributions, flierprops=flierprops, boxprops=boxprops,
                     whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops, patch_artist=True)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness distribution per generation (crossover: {GA_CROSSOVER_METHOD})")
    #label each generation
    plt.xticks(range(1, len(indices) + 1), xtick_labels)
    plt.grid(True)

    #annotate boxplots with stats
    for i, data in enumerate(sampled_distributions):
        x = i + 1  # x position for current box
        #compute quartiles and whiskers
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_candidates = [v for v in data if v >= Q1 - 1.5 * IQR]
        lower_whisker = min(lower_candidates) if lower_candidates else np.min(data)
        upper_candidates = [v for v in data if v <= Q3 + 1.5 * IQR]
        upper_whisker = max(upper_candidates) if upper_candidates else np.max(data)
        median_val = np.median(data)

        #add labels for key metrics
        plt.text(x + 0.1, median_val, f"{median_val:.1f}", color="red", fontsize=8, verticalalignment='center')
        plt.text(x - 0.3, lower_whisker, f"{lower_whisker:.1f}", color="blue", fontsize=8, verticalalignment='center')
        plt.text(x + 0.1, upper_whisker, f"{upper_whisker:.1f}", color="blue", fontsize=8, verticalalignment='center')

    #create custom legend
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    median_line = mlines.Line2D([], [], color='red', linewidth=2, label='median')
    whisker_line = mlines.Line2D([], [], color='blue', linewidth=1.5, label='whiskers (min/max non-outliers)')
    box_patch = mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='iqr (25th-75th percentile)')
    outlier_marker = mlines.Line2D([], [], marker='D', color='blue', linestyle='none', markersize=4, label='outliers')
    plt.legend(handles=[box_patch, median_line, whisker_line, outlier_marker], loc='upper right')

    plt.show()  #display the plot

#main program execution
def main():
    random.seed(time.time())  #different results each run
    population, buffer = init_population()
    overall_start_wall = time.time()  #track total runtime

    #data for visualization
    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []
    if(GA_CROSSOVER_METHOD not in ["single","two_point","uniform"]):
        print("No crossover operator detected, using single-point crossover by default.")
    else:
        print(f"Starting genetic algorithm with {GA_CROSSOVER_METHOD} crossover...")
    
    print(f"Using fitness mode: {GA_FITNESS_MODE}")

    for iteration in range(GA_MAXITER):
        generation_start_cpu = time.process_time()  #track per-generation time
        calc_fitness(population)  #score each candidate
        sort_by_fitness(population)  #rank them
        print_best(population)  #show progress

        #get population stats
        stats = compute_fitness_statistics(population)
        print(f"generation {iteration}: mean fitness = {stats['mean']:.2f}, std = {stats['std']:.2f}, "
              f"worst fitness = {stats['worst_fitness']}, range = {stats['fitness_range']}, "
              f"worst candidate = {stats['worst_candidate'].gene}")

        #get timing info
        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        print(f"generation {iteration}: cpu time = {timing['generation_cpu_time']:.4f} s, "
              f"elapsed time = {timing['elapsed_time']:.4f} s")

        #record data for plots
        best_history.append(population[0].fitness)
        mean_history.append(stats['mean'])
        worst_history.append(stats['worst_fitness'])
        #full distribution for boxplots
        fitness_distributions.append([cand.fitness for cand in population])

        #check if we solved it
        if population[0].fitness == 0:
            print("target reached!")
            break

        mate(population, buffer)  #create next generation
        population, buffer = swap(population, buffer)  #swap buffers

    #visualize results
    plot_fitness_evolution(best_history, mean_history, worst_history)
    plot_fitness_boxplots(fitness_distributions)


if __name__ == "__main__":
    main()

