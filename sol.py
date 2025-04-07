import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# constant params for the genetic algorithm
ga_popsize = 400 # large population for better exploration
ga_maxiter = 150   # maximum iterations to find a good solution
ga_elitrate = 0.10 # top 10% of candidates are kept as elite
ga_mutationrate = 0.55  # higher mutation rate to escape local optima
ga_target = "testing string123 diff_chars"  # the string we want to evolve towards
ga_crossover_method = "two_point"  # crossover strategy: single, two_point, or uniform
ga_lcs_bonus = 5  # weight factor for lcs in combined fitness
ga_fitness_mode = "ascii"  # fitness mode: ascii, lcs, or combined
ga_max_runtime = 120  # maximum runtime in seconds (2 minutes)
ga_distance_metric = "ulam"  # distance metric: levenshtein or ulam

# represents one candidate solution in the population
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene  # holds the string
        self.fitness = fitness  # lower is better

# sets up a random population and a buffer for the next generation
def init_population():
    target_length = len(ga_target)
    population = []
    for _ in range(ga_popsize):
        # build a random gene from ascii 32 to 121
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    # buffer is empty at first, same size
    buffer = [Candidate('', 0) for _ in range(ga_popsize)]
    return population, buffer

# calculates longest common subsequence length (str1, str2)
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

def levenshtein_distance(str1, str2):
    # get lengths of both strings
    len_str1, len_str2 = len(str1), len(str2)
    # create dp table for edit distances
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
    # initialize dp table for deletions (empty second string)
    for i in range(len_str1 + 1):
        dp[i][0] = i
    # initialize dp table for insertions (empty first string)
    for j in range(len_str2 + 1):
        dp[0][j] = j
    # fill dp table with min edit operations
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            # cost is 0 if characters match, else 1
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # return computed levenshtein distance
    return dp[len_str1][len_str2]

def ulam_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("strings must be of equal length")
        
    # Check if both strings have the same set of characters
    if set(s1) != set(s2):
        return levenshtein_distance(s1, s2), True  # Return the distance and signal fallback
        
    #map each character in s2 to its index
    index_map = {char: idx for idx, char in enumerate(s2)}
    #convert s1 into a list of indices based on s2's ordering
    mapped_indices = [index_map[char] for char in s1]
    #helper function to perform binary search for first index in lst that is >= value
    def find_position(lst, value):
        lo = 0
        hi = len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        return lo
    #initialize list to hold the tails of increasing subsequences
    lis_tails = []
    #compute the longest increasing subsequence using manual binary search
    for num in mapped_indices:
        pos = find_position(lis_tails, num)
        if pos == len(lis_tails):
            lis_tails.append(num)  #extend current subsequence
        else:
            lis_tails[pos] = num   #replace element to maintain lowest tail
    return len(s1) - len(lis_tails), False  # Return the distance and no fallback


# calculates fitness for each candidate based on mode
def calc_fitness(population):
    target = ga_target
    target_length = len(target)
    for candidate in population:
        if ga_fitness_mode == "ascii":
            # sum of ascii differences between gene and target
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))

        elif ga_fitness_mode == "lcs":
            # lcs-based fitness (longer lcs -> better, so invert by subtracting)
            lcs_length = longest_common_subsequence(candidate.gene, target)
            fitness = target_length - lcs_length

        elif ga_fitness_mode == "combined":
            # adds ascii difference and a weighted lcs difference
            ascii_fitness = 0
            for i in range(target_length):
                ascii_fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
            lcs_length = longest_common_subsequence(candidate.gene, target)
            lcs_fitness = target_length - lcs_length
            fitness = ascii_fitness + ga_lcs_bonus * lcs_fitness

        else:
            # fallback to ascii difference
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))

        candidate.fitness = fitness

# sorts the population by fitness ascending
def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)

# copies the top elite_size candidates directly into the next gen
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        buffer[i] = Candidate(population[i].gene, population[i].fitness)

# randomly mutate a single character in the candidate's gene
def mutate(candidate):
    target_length = len(ga_target)
    pos = random.randint(0, target_length - 1)
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    new_val = 32 + ((old_val - 32 + delta) % (121 - 32 + 1))
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)

# single-point crossover
def single_point_crossover(parent1, parent2, target_length):
    crossover_point = random.randint(0, target_length - 1)
    child1 = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    child2 = parent2.gene[:crossover_point] + parent1.gene[crossover_point:]
    return child1, child2

# two-point crossover
def two_point_crossover(parent1, parent2, target_length):
    point1, point2 = sorted(random.sample(range(target_length), 2))
    child1 = (parent1.gene[:point1] +
              parent2.gene[point1:point2] +
              parent1.gene[point2:])
    child2 = (parent2.gene[:point1] +
              parent1.gene[point1:point2] +
              parent2.gene[point2:])
    return child1, child2

# uniform crossover (each char chosen from one parent at random)
def uniform_crossover(parent1, parent2, target_length):
    child1_gene = []
    child2_gene = []
    for i in range(target_length):
        if random.random() < 0.5:
            child1_gene.append(parent1.gene[i])
            child2_gene.append(parent2.gene[i])
        else:
            child1_gene.append(parent2.gene[i])
            child2_gene.append(parent1.gene[i])
    return ''.join(child1_gene), ''.join(child2_gene)

# line ~149
def roulette_wheel_select(candidates):
    inv_fitnesses = [1.0 / (1.0 + c.fitness) for c in candidates]
    total_inv = sum(inv_fitnesses)
    probabilities = [inv / total_inv for inv in inv_fitnesses]
    pick = random.random()
    running_sum = 0.0
    for i, prob in enumerate(probabilities):
        running_sum += prob
        if pick <= running_sum:
            return candidates[i]
    return candidates[-1]


# creates the next generation using selection, crossover, and mutation
def mate(population, buffer):
    elite_size = int(ga_popsize * ga_elitrate)
    target_length = len(ga_target)
    elitism(population, buffer, elite_size)
    for i in range(elite_size, ga_popsize - 1, 2):
        parent1 = roulette_wheel_select(population[:ga_popsize // 2])
        parent2 = roulette_wheel_select(population[:ga_popsize // 2])

        if ga_crossover_method == "single":
            child1, child2 = single_point_crossover(parent1, parent2, target_length)
        elif ga_crossover_method == "two_point":
            child1, child2 = two_point_crossover(parent1, parent2, target_length)
        elif ga_crossover_method == "uniform":
            child1, child2 = uniform_crossover(parent1, parent2, target_length)
        else:
            child1, child2 = single_point_crossover(parent1, parent2, target_length)

        buffer[i] = Candidate(child1)
        buffer[i + 1] = Candidate(child2)

        if random.random() < ga_mutationrate:
            mutate(buffer[i])
        if random.random() < ga_mutationrate:
            mutate(buffer[i + 1])

    # handle odd population edge case
    if ga_popsize % 2 == 1 and elite_size % 2 == 0:
        buffer[-1] = Candidate(buffer[-2].gene)
        if random.random() < ga_mutationrate:
            mutate(buffer[-1])

# prints the best candidate (lowest fitness)
def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")

# swaps population and buffer references
def swap(population, buffer):
    return buffer, population

# calculates some stats: mean, std, etc
def compute_fitness_statistics(population):
    fitness_values = [cand.fitness for cand in population]
    mean_fitness = sum(fitness_values) / len(fitness_values)
    variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / (len(fitness_values)-1)
    std_fitness = math.sqrt(variance)
    best_fitness = population[0].fitness
    worst_fitness = population[-1].fitness
    fitness_range = worst_fitness - best_fitness

    inv_fitnesses = [1.0 / (1.0 + c.fitness) for c in population]
    sum_inv_fitnesses = sum(inv_fitnesses)
    top_size = max(1, int(ga_elitrate * len(population)))
    selection_probs=[inv_fitness / sum_inv_fitnesses for inv_fitness in inv_fitnesses]
    selection_probs.sort(reverse=True)
    selection_variance=np.var(selection_probs) * 10 ** (-int(math.floor(math.log10(np.var(selection_probs)))))

    p_avg = 1 / len(population)
    top_probs = selection_probs[:top_size]
    p_top = sum(top_probs) / top_size
    top_avg_prob_ratio = p_top / p_avg

    stats = {
        "mean": mean_fitness,
        "std": std_fitness,
        "variance": variance,
        "selection_variance": selection_variance,
        "worst_fitness": worst_fitness,
        "fitness_range": fitness_range,
        "worst_candidate": population[-1],
        "top_avg_prob_ratio": top_avg_prob_ratio
    }
    return stats

# measures how long things take each generation
def compute_timing_metrics(generation_start_cpu, overall_start_wall):
    current_cpu = time.process_time()
    current_wall = time.time()
    generation_cpu_time = current_cpu - generation_start_cpu
    elapsed_time = current_wall - overall_start_wall

    # Get raw clock ticks for this generation
    raw_ticks = time.perf_counter_ns()  # Get raw nanosecond ticks
    ticks_per_second = time.get_clock_info('perf_counter').resolution

    return {
        "generation_cpu_time": generation_cpu_time,
        "elapsed_time": elapsed_time,
        "raw_ticks": raw_ticks,
        "ticks_per_second": ticks_per_second
    }

# New function to calculate average population distance
def calculate_avg_population_distance(population, distance_metric="levenshtein"):
    """
    Calculate the average pairwise distance between population members.
    
    Args:
        population: List of Candidate objects
        distance_metric: String, either "levenshtein" or "ulam"
        
    Returns:
        tuple: (float: Average pairwise distance, str: actual metric used)
    """
    global ga_distance_metric
    total_distance = 0
    count = 0
    used_levenshtein_fallback = False
    
    # Sample a subset if population is large to avoid excessive computation
    pop_size = len(population)
    sample_size = min(50, pop_size)  # Cap at 50 candidates to avoid O(n²) performance issues
    
    if pop_size <= sample_size:
        sample = population
    else:
        # Sample randomly but include some of the best candidates
        elite_size = max(1, int(0.1 * sample_size))  # Include top 10% of the sample
        sample = population[:elite_size] + random.sample(population[elite_size:], sample_size - elite_size)
    
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if distance_metric == "ulam":
                # For Ulam distance, ensure strings are of same length
                if len(sample[i].gene) == len(sample[j].gene):
                    try:
                        distance, fallback = ulam_distance(sample[i].gene, sample[j].gene)
                        if fallback:
                            used_levenshtein_fallback = True
                    except ValueError:
                        distance = levenshtein_distance(sample[i].gene, sample[j].gene)
                        used_levenshtein_fallback = True
                else:
                    distance = levenshtein_distance(sample[i].gene, sample[j].gene)
                    used_levenshtein_fallback = True
            else:
                # Default to levenshtein
                distance = levenshtein_distance(sample[i].gene, sample[j].gene)
            
            total_distance += distance
            count += 1
    
    # If we had to fall back to Levenshtein, update the global metric
    if used_levenshtein_fallback and distance_metric == "ulam":
        ga_distance_metric = "levenshtein"
        print("Warning: Had to fall back to Levenshtein distance because Ulam conditions were not met")
    
    if count == 0:
        return 0, ga_distance_metric
    
    return total_distance / count, ga_distance_metric

# plots best, mean, and worst fitness over time
def plot_fitness_evolution(best_history, mean_history, worst_history):
    generations = list(range(len(best_history)))
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_history, label="best", linewidth=2)
    plt.plot(generations, mean_history, label="mean", linewidth=2)
    plt.plot(generations, worst_history, label="worst", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness evolution over generations (crossover: {ga_crossover_method})")
    plt.legend()
    plt.grid(True)
    plt.show()

# plots a series of boxplots for fitness distributions across generations
def plot_fitness_boxplots(fitness_distributions):
    plt.figure(figsize=(14, 8))
    flierprops = dict(marker='D', markersize=4, linestyle='none', markeredgecolor='blue')
    boxprops = dict(facecolor='lightblue', color='blue', linewidth=1.5)
    whiskerprops = dict(color='blue', linewidth=1.5)
    capprops = dict(color='blue', linewidth=1.5)
    medianprops = dict(color='red', linewidth=2)

    total = len(fitness_distributions)
    if total > 10:
        # sample 10 points evenly
        indices = [int(round(i * (total - 1) / 9)) for i in range(10)]
        sampled_distributions = [fitness_distributions[i] for i in indices]
        xtick_labels = [str(i) for i in indices]
    else:
        indices = list(range(total))
        sampled_distributions = fitness_distributions
        xtick_labels = [str(i) for i in indices]

    bp = plt.boxplot(
        sampled_distributions,
        flierprops=flierprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
        patch_artist=True
    )
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness distribution per generation (crossover: {ga_crossover_method})")
    plt.xticks(range(1, len(indices) + 1), xtick_labels)
    plt.grid(True)

    # annotate boxplots with quartiles and whiskers
    for i, data in enumerate(sampled_distributions):
        x = i + 1
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_candidates = [v for v in data if v >= q1 - 1.5 * iqr]
        lower_whisker = min(lower_candidates) if lower_candidates else np.min(data)
        upper_candidates = [v for v in data if v <= q3 + 1.5 * iqr]
        upper_whisker = max(upper_candidates) if upper_candidates else np.max(data)
        median_val = np.median(data)

        plt.text(x + 0.1, median_val, f"{median_val:.1f}", color="red", fontsize=8, verticalalignment='center')
        plt.text(x - 0.3, lower_whisker, f"{lower_whisker:.1f}", color="blue", fontsize=8, verticalalignment='center')
        plt.text(x + 0.1, upper_whisker, f"{upper_whisker:.1f}", color="blue", fontsize=8, verticalalignment='center')

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    median_line = mlines.Line2D([], [], color='red', linewidth=2, label='median')
    whisker_line = mlines.Line2D([], [], color='blue', linewidth=1.5, label='whiskers')
    box_patch = mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='iqr')
    outlier_marker = mlines.Line2D([], [], marker='D', color='blue', linestyle='none', markersize=4, label='outliers')
    plt.legend(handles=[box_patch, median_line, whisker_line, outlier_marker], loc='upper right')

    plt.show()

# runs the ga with the given parameters and returns stats
def run_ga(crossover_method, fitness_mode, lcs_bonus, mutation_rate, 
           population_size=2000, max_runtime=120, distance_metric="levenshtein"):
    """
    runs the ga with the specified settings, returns:
      {
        "best_fitness_history": [...],
        "converged_generation": int,
        "termination_reason": str
      }
    """
    global ga_crossover_method, ga_fitness_mode, ga_lcs_bonus, ga_mutationrate, ga_popsize, ga_distance_metric
    ga_crossover_method = crossover_method
    ga_fitness_mode = fitness_mode
    ga_lcs_bonus = lcs_bonus
    ga_mutationrate = mutation_rate
    ga_popsize = population_size
    ga_distance_metric = distance_metric

    random.seed(time.time())
    population, buffer = init_population()
    overall_start_wall = time.time()

    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []

    converged_generation = ga_maxiter
    termination_reason = "max_iterations"

    for iteration in range(ga_maxiter):
        # Check if we've exceeded the time limit
        current_time = time.time()
        elapsed_time = current_time - overall_start_wall
        if elapsed_time >= max_runtime:
            print(f"Time limit of {max_runtime} seconds reached after {iteration} generations.")
            termination_reason = "time_limit"
            converged_generation = iteration
            break

        generation_start_cpu = time.process_time()
        generation_start_ticks = time.perf_counter_ns()

        calc_fitness(population)
        sort_by_fitness(population)

        best_f = population[0].fitness
        best_history.append(best_f)

        print_best(population)
        stats = compute_fitness_statistics(population)
        
        # Calculate average population distance
        avg_distance, actual_metric = calculate_avg_population_distance(population, ga_distance_metric)
        
        print(f"generation {iteration}: mean fitness = {stats['mean']:.2f}, selection variance = {stats['selection_variance']:.4f}, fitness std = {stats['std']:.2f}, worst fitness = {stats['worst_fitness']}, range = {stats['fitness_range']}, worst candidate = {stats['worst_candidate'].gene}")
        print(f"selection pressure -> top_avg_prob_ratio = {stats['top_avg_prob_ratio']:.2f}, avg {actual_metric} distance = {avg_distance:.2f}")

        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        gen_ticks = time.perf_counter_ns() - generation_start_ticks
        print(f"generation {iteration}: cpu time = {timing['generation_cpu_time']:.4f} s, "
              f"elapsed time = {timing['elapsed_time']:.4f} s, "
              f"raw ticks = {gen_ticks}, "
              f"tick time = {gen_ticks/1e9:.6f} s")

        mean_history.append(stats['mean'])
        worst_history.append(stats['worst_fitness'])
        fitness_distributions.append([cand.fitness for cand in population])

        if best_f == 0:
            print("target reached!")
            termination_reason = "solution_found"
            converged_generation = iteration
            break

        mate(population, buffer)
        population, buffer = swap(population, buffer)

    return {
        "best_fitness_history": best_history,
        "converged_generation": converged_generation,
        "termination_reason": termination_reason
    }

# main function for a single run; can comment out plots if you want
def main():
    random.seed(time.time())
    population, buffer = init_population()
    overall_start_wall = time.time()

    # Store the initial raw ticks
    initial_raw_ticks = time.perf_counter_ns()

    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []

    if ga_crossover_method not in ["single", "two_point", "uniform"]:
        print("no crossover operator detected, using single-point crossover by default.")
    else:
        print(f"starting genetic algorithm with {ga_crossover_method} crossover...")

    if ga_fitness_mode not in ["ascii", "lcs", "combined"]:
        print("no fitness mode selected, defaulting to ascii")
    else:
        print(f"using fitness mode: {ga_fitness_mode}")

    print(f"Maximum runtime set to {ga_max_runtime} seconds")
    print(f"Using {ga_distance_metric} distance metric for population diversity")

    for iteration in range(ga_maxiter):
        # Check if we've exceeded the time limit
        current_time = time.time()
        elapsed_time = current_time - overall_start_wall
        if elapsed_time >= ga_max_runtime:
            print(f"Time limit of {ga_max_runtime} seconds reached after {iteration} generations.")
            break

        generation_start_cpu = time.process_time()
        generation_start_ticks = time.perf_counter_ns()

        calc_fitness(population)
        sort_by_fitness(population)
        print_best(population)

        stats = compute_fitness_statistics(population)
        
        # Calculate average population distance
        avg_distance, actual_metric = calculate_avg_population_distance(population, ga_distance_metric)
        
        print(f"generation {iteration}: mean fitness = {stats['mean']:.2f}, selection variance = {stats['selection_variance']:.4f}, fitness std = {stats['std']:.2f}, worst fitness = {stats['worst_fitness']}, range = {stats['fitness_range']}, worst candidate = {stats['worst_candidate'].gene}")
        print(f"selection pressure -> top_avg_prob_ratio = {stats['top_avg_prob_ratio']:.2f}, avg {actual_metric} distance = {avg_distance:.2f}")

        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        gen_ticks = time.perf_counter_ns() - generation_start_ticks
        print(f"generation {iteration}: cpu time = {timing['generation_cpu_time']:.4f} s, "
              f"elapsed time = {timing['elapsed_time']:.4f} s, "
              f"raw ticks = {gen_ticks}, "
              f"tick time = {gen_ticks/1e9:.6f} s")

        best_history.append(population[0].fitness)
        mean_history.append(stats['mean'])
        worst_history.append(stats['worst_fitness'])
        fitness_distributions.append([cand.fitness for cand in population])

        if population[0].fitness == 0:
            print("target reached!")
            break

        mate(population, buffer)
        population, buffer = swap(population, buffer)

    final_time = time.time() - overall_start_wall
    print(f"Total runtime: {final_time:.2f} seconds")

    # comment out or uncomment these plots as you wish
    plot_fitness_evolution(best_history, mean_history, worst_history)
    plot_fitness_boxplots(fitness_distributions)

if __name__ == "__main__":
    main()

