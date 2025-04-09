import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS - NEW/UPDATED
# -----------------------------------------------------------------------------
ga_selection_method = "rws"      # "rws", "sus", "tournament_deterministic", "tournament_probabilistic"
ga_use_linear_scaling = False     # If True, apply linear scaling for RWS or SUS
ga_max_fitness_ratio = 2.0       # Maximum ratio (scaled_best / scaled_avg) to limit dominance
ga_tournament_k = 3              # 'k' for deterministic tournament
ga_tournament_k_prob = 3         # 'k' for probabilistic tournament
ga_tournament_p = 0.75           # 'p' for probabilistic tournament

ga_use_aging = False             # Enable or disable aging-based survival
ga_age_limit = 100               # Default age limit for individuals

# Existing parameters for GA:
ga_popsize = 500
ga_maxiter = 150
ga_elitrate = 0.10
ga_mutationrate = 0.55
ga_target = "testing string123 diff_chars"
ga_crossover_method = "two_point"  # "single", "two_point", or "uniform"
ga_lcs_bonus = 5
ga_fitness_mode = "ascii"       # "ascii", "lcs", "combined"
ga_max_runtime = 120
ga_distance_metric = "levenshtein" # "ulam" or "levenshtein"


# -----------------------------------------------------------------------------
# CANDIDATE CLASS WITH AGE
# -----------------------------------------------------------------------------
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene
        self.fitness = fitness
        self.age = 0  # new field to track aging


# -----------------------------------------------------------------------------
# POPULATION INITIALIZATION
# -----------------------------------------------------------------------------
def init_population():
    target_length = len(ga_target)
    population = []
    for _ in range(ga_popsize):
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    # create a buffer to hold next-gen individuals
    buffer = [Candidate('', 0) for _ in range(ga_popsize)]
    return population, buffer


# -----------------------------------------------------------------------------
# FITNESS UTILITIES
# -----------------------------------------------------------------------------
def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def levenshtein_distance(str1, str2):
    len_str1, len_str2 = len(str1), len(str2)
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
    for i in range(len_str1 + 1):
        dp[i][0] = i
    for j in range(len_str2 + 1):
        dp[0][j] = j
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len_str1][len_str2]


def ulam_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("strings must be of equal length")

    if set(s1) != set(s2):
        return levenshtein_distance(s1, s2), True  # fallback

    index_map = {char: idx for idx, char in enumerate(s2)}
    mapped_indices = [index_map[char] for char in s1]

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

    lis_tails = []
    for num in mapped_indices:
        pos = find_position(lis_tails, num)
        if pos == len(lis_tails):
            lis_tails.append(num)
        else:
            lis_tails[pos] = num

    return len(s1) - len(lis_tails), False


def different_alleles(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("strings must have equal length for different_alleles calculation")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def calc_fitness(population):
    target = ga_target
    target_length = len(target)
    for candidate in population:
        if ga_fitness_mode == "ascii":
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))

        elif ga_fitness_mode == "lcs":
            lcs_len = longest_common_subsequence(candidate.gene, target)
            fitness = target_length - lcs_len

        elif ga_fitness_mode == "combined":
            ascii_fit = 0
            for i in range(target_length):
                ascii_fit += abs(ord(candidate.gene[i]) - ord(target[i]))
            lcs_len = longest_common_subsequence(candidate.gene, target)
            lcs_fit = target_length - lcs_len
            fitness = ascii_fit + ga_lcs_bonus * lcs_fit

        else:
            # default fallback
            fitness = 0
            for i in range(target_length):
                fitness += abs(ord(candidate.gene[i]) - ord(target[i]))

        candidate.fitness = fitness


def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)


# -----------------------------------------------------------------------------
# SELECTION: LINEAR SCALING FOR RWS / SUS
# -----------------------------------------------------------------------------
def linear_scale_fitness(population, max_ratio=2.0):
    """
    Perform linear scaling of each candidate's fitness to keep the ratio
    (max_scaled / avg_scaled) <= max_ratio. We'll store the scaled fitness
    in candidate.scaled_fitness to use in RWS or SUS.
    """
    # First, compute raw stats
    raw_fitnesses = [c.fitness for c in population]
    f_min = min(raw_fitnesses)
    f_max = max(raw_fitnesses)
    f_avg = sum(raw_fitnesses) / len(raw_fitnesses)

    # If the entire population has the same fitness, no scaling needed
    if abs(f_max - f_min) < 1e-9:
        for c in population:
            c.scaled_fitness = 1.0  # constant
        return

    # Basic linear scaling terms: s_i = a * f_i + b
    # We'll adjust a and b so that:
    #  - the best individual is not more than 'max_ratio' times above the average.
    #
    # Standard linear scaling formula from Goldberg:
    #   s_i = a * f_i + b
    #
    # We can also enforce: scaled_min >= 0

    # By default, a = (desired_avg - min_scaled) / (f_avg - f_min)
    # but we also want scaled_max / scaled_avg <= max_ratio
    # We'll do a quick approach:
    #  1) Set s_i' = f_max - f_i so "lower fitness = bigger selection chance" if you prefer.
    #     However, typically for RWS we want "bigger fitness => bigger probability".
    #     We'll do "bigger fitness => bigger selection chance". So let's invert:
    #     If your GA is currently "lower is better" you might invert. But let's keep it consistent
    #     with your current code. Right now, "candidate.fitness = lower is better". We'll invert
    #     to turn that into "scaled_fitness is bigger is better".
    #
    # So we define: raw_score = (f_max - c.fitness).
    # Then we do the standard linear scaling on raw_score to limit ratio.
    #
    # We'll rename them for clarity:
    scores = [f_max - f for f in raw_fitnesses]  # now bigger = better
    score_min, score_max = min(scores), max(scores)
    score_avg = sum(scores) / len(scores)

    if abs(score_max - score_min) < 1e-9:
        # Everyone is effectively the same after inversion
        for c in population:
            c.scaled_fitness = 1.0
        return

    # If we do standard linear scaling to ensure scaled_min ~ 0
    #   s_i = (scores[i] - score_min)
    # Then we can shift or scale up so that the average is e.g. some value. Then we clamp ratio.
    # We'll do a simpler approach:
    #   s_i_base = (scores[i] - score_min)
    #   base_max = score_max - score_min
    #   base_avg = sum of s_i_base / n
    # We want: scaled_max / scaled_avg <= max_ratio
    # scaled_max = a * base_max
    # scaled_avg = a * base_avg
    # => scaled_max / scaled_avg = base_max / base_avg
    # so we must choose a so that
    #   (a * base_max) / (a * base_avg) = base_max / base_avg <= max_ratio
    # If base_max/base_avg is bigger than max_ratio, we reduce a; otherwise we do a=1
    base_values = [s - score_min for s in scores]
    base_max = score_max - score_min
    base_avg = sum(base_values) / len(base_values)

    ratio = (base_max / base_avg) if base_avg > 1e-9 else 1.0
    if ratio > max_ratio:
        # scale down
        a = max_ratio / ratio
    else:
        a = 1.0

    # Then s_i = a * base_values[i]
    # This ensures the ratio <= max_ratio
    # We'll just store scaled_fitness = max(0, a*base_val)
    # If that ends up being zero for some, that's okay, as long as total > 0
    for i, c in enumerate(population):
        base_val = base_values[i]
        c.scaled_fitness = a * base_val

    # If entire population is extremely uniform, we might end up with all zeroes.
    # As a fallback, ensure they sum to something:
    total_scaled = sum(c.scaled_fitness for c in population)
    if total_scaled < 1e-9:
        # if all zero, just assign uniform
        for c in population:
            c.scaled_fitness = 1.0


# -----------------------------------------------------------------------------
# SELECTION METHODS
# -----------------------------------------------------------------------------
def rws_select_one(population):
    """Roulette Wheel Selection for ONE parent, assuming we have .scaled_fitness available."""
    total = sum(c.scaled_fitness for c in population)
    if total < 1e-9:
        return random.choice(population)

    pick = random.random() * total
    running = 0.0
    for c in population:
        running += c.scaled_fitness
        if running >= pick:
            return c
    return population[-1]


def sus_select_parents(population, num_parents):
    """
    Stochastic Universal Sampling to pick 'num_parents' individuals at once.
    We assume .scaled_fitness is set, we do a single pass with equally spaced pointers.
    Return the list of chosen individuals.
    """
    total = sum(c.scaled_fitness for c in population)
    if total < 1e-9:
        # fallback: random picks
        return [random.choice(population) for _ in range(num_parents)]

    distance = total / num_parents
    start = random.random() * distance
    chosen = []
    running_sum = 0.0
    idx = 0
    for i in range(num_parents):
        pointer = start + i * distance
        while running_sum < pointer and idx < len(population) - 1:
            running_sum += population[idx].scaled_fitness
            idx += 1
        chosen.append(population[idx - 1])
    return chosen


def tournament_deterministic_select_one(population, k=2):
    """Pick the best out of k randomly chosen candidates."""
    contenders = random.sample(population, k)
    # The one with the lowest .fitness is best in your system
    # so we sort by .fitness ascending
    contenders.sort(key=lambda c: c.fitness)
    return contenders[0]


def tournament_probabilistic_select_one(population, k=2, p=0.75):
    """
    Non-deterministic (probabilistic) tournament selection:
      1) Randomly choose k individuals
      2) Sort them best->worst
      3) Pick the best with prob p, second-best with prob p*(1-p), etc.
    """
    contenders = random.sample(population, k)
    contenders.sort(key=lambda c: c.fitness)  # best first
    # best has index 0
    # probability for contender[i] = p * (1-p)^i
    # but we need to do a stepwise draw
    # We'll do a standard approach: generate r, walk down cumulative probabilities
    r = random.random()
    cumulative = 0.0
    for i, cand in enumerate(contenders):
        prob_i = p * ((1 - p) ** i)
        cumulative += prob_i
        if r <= cumulative:
            return cand
    # if not found, return the last
    return contenders[-1]


def select_one_parent(population):
    """
    Master selection function that picks ONE parent
    using ga_selection_method. If using RWS/SUS,
    ensure linear scaling has already been called.
    """
    method = ga_selection_method.lower()
    if method == "rws":
        # we assume that if ga_use_linear_scaling is True, we've done linear_scale_fitness
        return rws_select_one(population)
    elif method == "sus":
        # For SUS, we typically do one big pass for multiple parents,
        # but for simplicity we can pick one parent at a time:
        chosen = sus_select_parents(population, 1)
        return chosen[0]

    elif method == "tournament_deterministic":
        return tournament_deterministic_select_one(population, ga_tournament_k)

    elif method == "tournament_probabilistic":
        return tournament_probabilistic_select_one(population, ga_tournament_k_prob, ga_tournament_p)

    else:
        # default to the old approach for backward compatibility
        return old_roulette_wheel_select(population)


def old_roulette_wheel_select(candidates):
    """
    This is your original random-based selection approach from the code
    (in the snippet you had a 'roulette_wheel_select' that used 1/(1+fitness)).
    We keep it here so that if user does not set ga_selection_method,
    the code remains backward compatible.
    """
    inv_fitnesses = [1.0 / (1.0 + c.fitness) for c in candidates]
    total_inv = sum(inv_fitnesses)
    pick = random.random()
    running_sum = 0.0
    for i, inv in enumerate(inv_fitnesses):
        running_sum += inv / total_inv
        if pick <= running_sum:
            return candidates[i]
    return candidates[-1]


# -----------------------------------------------------------------------------
# CROSSOVER + MUTATION
# -----------------------------------------------------------------------------
def single_point_crossover(parent1, parent2, target_length):
    crossover_point = random.randint(0, target_length - 1)
    child1 = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    child2 = parent2.gene[:crossover_point] + parent1.gene[crossover_point:]
    return child1, child2


def two_point_crossover(parent1, parent2, target_length):
    point1, point2 = sorted(random.sample(range(target_length), 2))
    child1 = (parent1.gene[:point1] +
              parent2.gene[point1:point2] +
              parent1.gene[point2:])
    child2 = (parent2.gene[:point1] +
              parent1.gene[point1:point2] +
              parent2.gene[point2:])
    return child1, child2


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


def mutate(candidate):
    target_length = len(ga_target)
    pos = random.randint(0, target_length - 1)
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    new_val = 32 + ((old_val - 32 + delta) % (121 - 32 + 1))
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)


# -----------------------------------------------------------------------------
# ELITISM
# -----------------------------------------------------------------------------
def elitism(population, buffer, elite_size):
    """
    Copy top 'elite_size' from population -> buffer,
    preserving their ages (and incrementing them).
    """
    for i in range(elite_size):
        buffer[i].gene = population[i].gene
        buffer[i].fitness = population[i].fitness
        # Surviving to next gen => increment age
        buffer[i].age = population[i].age + 1


# -----------------------------------------------------------------------------
# AGING-BASED SURVIVAL
# -----------------------------------------------------------------------------
def apply_aging_replacement(population):
    """
    For each individual whose age > ga_age_limit, replace it with a new child
    from crossover of two selected parents (same selection method).
    Age of new child = 0.
    """
    if not ga_use_aging:
        return

    target_length = len(ga_target)
    for i in range(len(population)):
        if population[i].age > ga_age_limit:
            # pick two parents
            parent1 = select_one_parent(population)
            parent2 = select_one_parent(population)

            # do crossover
            if ga_crossover_method == "single":
                c1, _ = single_point_crossover(parent1, parent2, target_length)
            elif ga_crossover_method == "two_point":
                c1, _ = two_point_crossover(parent1, parent2, target_length)
            elif ga_crossover_method == "uniform":
                c1, _ = uniform_crossover(parent1, parent2, target_length)
            else:
                c1, _ = single_point_crossover(parent1, parent2, target_length)

            # form new candidate
            new_cand = Candidate(c1, fitness=0)
            new_cand.age = 0
            population[i] = new_cand


# -----------------------------------------------------------------------------
# MATE FUNCTION (BREED NEXT GENERATION)
# -----------------------------------------------------------------------------
def mate(population, buffer):
    elite_size = int(ga_popsize * ga_elitrate)
    target_length = len(ga_target)

    # 1) Elitism
    elitism(population, buffer, elite_size)

    # 2) If using RWS or SUS, perform linear scaling if requested
    method = ga_selection_method.lower()
    if method in ["rws", "sus"] and ga_use_linear_scaling:
        linear_scale_fitness(population, ga_max_fitness_ratio)
    # If method is something else, we skip linear scaling

    # 3) Fill the remainder of the buffer
    i = elite_size
    while i < ga_popsize - 1:
        # pick parents
        parent1 = select_one_parent(population)
        parent2 = select_one_parent(population)

        # do crossover
        if ga_crossover_method == "single":
            child1, child2 = single_point_crossover(parent1, parent2, target_length)
        elif ga_crossover_method == "two_point":
            child1, child2 = two_point_crossover(parent1, parent2, target_length)
        elif ga_crossover_method == "uniform":
            child1, child2 = uniform_crossover(parent1, parent2, target_length)
        else:
            child1, child2 = single_point_crossover(parent1, parent2, target_length)

        buffer[i].gene = child1
        buffer[i].fitness = 0
        buffer[i].age = 0  # new child => age=0
        buffer[i + 1].gene = child2
        buffer[i + 1].fitness = 0
        buffer[i + 1].age = 0

        # apply mutation
        if random.random() < ga_mutationrate:
            mutate(buffer[i])
        if random.random() < ga_mutationrate:
            mutate(buffer[i + 1])

        i += 2

    # If popsize is odd, handle last one
    if ga_popsize % 2 == 1 and i < ga_popsize:
        buffer[i].gene = buffer[i - 1].gene
        buffer[i].fitness = buffer[i - 1].fitness
        buffer[i].age = buffer[i - 1].age
        if random.random() < ga_mutationrate:
            mutate(buffer[i])

    # 4) Aging replacement step (on the newly formed generation)
    apply_aging_replacement(buffer)


# -----------------------------------------------------------------------------
# SWAP, PRINT, AND STATISTICS
# -----------------------------------------------------------------------------
def swap(population, buffer):
    return buffer, population


def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness}) age={best.age}")


def compute_fitness_statistics(population):
    fitness_values = [cand.fitness for cand in population]
    mean_fitness = sum(fitness_values) / len(fitness_values)
    if len(fitness_values) > 1:
        variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / (len(fitness_values) - 1)
    else:
        variance = 0.0
    std_fitness = math.sqrt(variance)
    best_fitness = population[0].fitness
    worst_fitness = population[-1].fitness
    fitness_range = worst_fitness - best_fitness

    inv_fitnesses = [1.0 / (1.0 + c.fitness) for c in population]
    sum_inv_fitnesses = sum(inv_fitnesses)
    top_size = max(1, int(ga_elitrate * len(population)))
    selection_probs = [inv_fitness / sum_inv_fitnesses for inv_fitness in inv_fitnesses]
    selection_probs.sort(reverse=True)
    selection_variance = np.var(selection_probs)
    if selection_variance < 1e-12:
        selection_variance = 0.0
    p_avg = 1 / len(population)
    top_probs = selection_probs[:top_size]
    p_top = sum(top_probs) / top_size
    top_avg_prob_ratio = p_top / p_avg if p_avg > 1e-9 else 1.0

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


def compute_timing_metrics(generation_start_cpu, overall_start_wall):
    current_cpu = time.process_time()
    current_wall = time.time()
    generation_cpu_time = current_cpu - generation_start_cpu
    elapsed_time = current_wall - overall_start_wall
    raw_ticks = time.perf_counter_ns()
    ticks_per_second = time.get_clock_info('perf_counter').resolution

    return {
        "generation_cpu_time": generation_cpu_time,
        "elapsed_time": elapsed_time,
        "raw_ticks": raw_ticks,
        "ticks_per_second": ticks_per_second
    }


# -----------------------------------------------------------------------------
# DIVERSITY METRICS
# -----------------------------------------------------------------------------
def calculate_avg_different_alleles(population):
    total_diff = 0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total_diff += different_alleles(population[i].gene, population[j].gene)
            count += 1
    if count == 0:
        return 0
    return total_diff / count


def calculate_avg_population_distance(population, distance_metric="levenshtein"):
    global ga_distance_metric
    total_distance = 0
    count = 0
    used_levenshtein_fallback = False

    pop_size = len(population)
    sample_size = min(50, pop_size)
    if pop_size <= sample_size:
        sample = population
    else:
        elite_size = max(1, int(0.1 * sample_size))
        sample = population[:elite_size] + random.sample(population[elite_size:], sample_size - elite_size)

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if distance_metric == "ulam":
                if len(sample[i].gene) == len(sample[j].gene):
                    try:
                        dist, fallback = ulam_distance(sample[i].gene, sample[j].gene)
                        if fallback:
                            used_levenshtein_fallback = True
                    except ValueError:
                        dist = levenshtein_distance(sample[i].gene, sample[j].gene)
                        used_levenshtein_fallback = True
                else:
                    dist = levenshtein_distance(sample[i].gene, sample[j].gene)
                    used_levenshtein_fallback = True
            else:
                dist = levenshtein_distance(sample[i].gene, sample[j].gene)
            total_distance += dist
            count += 1

    if used_levenshtein_fallback and distance_metric == "ulam":
        ga_distance_metric = "levenshtein"
        print("Warning: Had to fall back to Levenshtein distance because Ulam conditions were not met")

    if count == 0:
        return 0, ga_distance_metric
    return total_distance / count, ga_distance_metric


def calculate_avg_shannon_entropy(population):
    if not population or len(population) == 0:
        return 0.0
    gene_length = len(population[0].gene)
    position_entropies = []
    for position in range(gene_length):
        position_chars = [candidate.gene[position] for candidate in population]
        char_count = {}
        for char in position_chars:
            char_count[char] = char_count.get(char, 0) + 1

        entropy = 0.0
        pop_size = len(population)
        for count in char_count.values():
            probability = count / pop_size
            entropy -= probability * math.log2(probability)
        position_entropies.append(entropy)
    return sum(position_entropies) / gene_length


# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------
def plot_fitness_evolution(best_history, mean_history, worst_history):
    generations = list(range(len(best_history)))
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_history, label="best", linewidth=2)
    plt.plot(generations, mean_history, label="mean", linewidth=2)
    plt.plot(generations, worst_history, label="worst", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness evolution (crossover: {ga_crossover_method})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitness_boxplots(fitness_distributions):
    plt.figure(figsize=(14, 8))
    flierprops = dict(marker='D', markersize=4, linestyle='none', markeredgecolor='blue')
    boxprops = dict(facecolor='lightblue', color='blue', linewidth=1.5)
    whiskerprops = dict(color='blue', linewidth=1.5)
    capprops = dict(color='blue', linewidth=1.5)
    medianprops = dict(color='red', linewidth=1)

    total = len(fitness_distributions)
    if total > 10:
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
    plt.title(f"fitness distribution (crossover: {ga_crossover_method})")
    plt.xticks(range(1, len(indices) + 1), xtick_labels)
    plt.grid(True)

    # annotate for readability
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


def plot_entropy_evolution(entropy_history, allele_diff_history, distance_history):
    generations = list(range(len(entropy_history)))
    plt.figure(figsize=(14, 8))
    plt.plot(generations, entropy_history, label="Shannon Entropy", linewidth=2)
    plt.plot(generations, allele_diff_history, label="Avg Different Alleles", linewidth=2)
    plt.plot(generations, distance_history, label="Avg Distance", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Diversity Metrics")
    plt.title("Population Diversity Metrics")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# GA EXECUTION (PROGRAMMATIC ENTRY POINT)
# -----------------------------------------------------------------------------
def run_ga(
    crossover_method="two_point",
    fitness_mode="combined",
    lcs_bonus=5,
    mutation_rate=0.55,
    population_size=500,
    max_runtime=120,
    distance_metric="levenshtein",
    selection_method="rws",
    use_linear_scaling=False,
    max_fitness_ratio=2.0,
    use_aging=False,
    age_limit=100,
    tournament_k=3,
    tournament_k_prob=3,
    tournament_p=0.75
):
    """
    Run the GA with the specified settings, returning a dict of stats.
    All parameters have defaults for backward-compatibility.
    """
    global ga_crossover_method, ga_fitness_mode, ga_lcs_bonus, ga_mutationrate
    global ga_popsize, ga_distance_metric, ga_max_runtime
    global ga_selection_method, ga_use_linear_scaling, ga_max_fitness_ratio
    global ga_use_aging, ga_age_limit, ga_tournament_k, ga_tournament_k_prob, ga_tournament_p

    # set all
    ga_crossover_method = crossover_method
    ga_fitness_mode = fitness_mode
    ga_lcs_bonus = lcs_bonus
    ga_mutationrate = mutation_rate
    ga_popsize = population_size
    ga_distance_metric = distance_metric
    ga_max_runtime = max_runtime

    ga_selection_method = selection_method
    ga_use_linear_scaling = use_linear_scaling
    ga_max_fitness_ratio = max_fitness_ratio

    ga_use_aging = use_aging
    ga_age_limit = age_limit

    ga_tournament_k = tournament_k
    ga_tournament_k_prob = tournament_k_prob
    ga_tournament_p = tournament_p

    random.seed(time.time())
    population, buffer = init_population()
    overall_start_wall = time.time()

    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []
    entropy_history = []
    allele_diff_history = []
    distance_history = []

    converged_generation = ga_maxiter
    termination_reason = "max_iterations"

    for iteration in range(ga_maxiter):
        if (time.time() - overall_start_wall) >= ga_max_runtime:
            print(f"Time limit of {ga_max_runtime} seconds reached after {iteration} generations.")
            termination_reason = "time_limit"
            converged_generation = iteration
            break

        generation_start_cpu = time.process_time()
        generation_start_ticks = time.perf_counter_ns()

        calc_fitness(population)
        sort_by_fitness(population)

        best_history.append(population[0].fitness)
        print_best(population)
        stats = compute_fitness_statistics(population)

        # diversity
        avg_distance, actual_metric = calculate_avg_population_distance(population, ga_distance_metric)
        distance_history.append(avg_distance)
        avg_diff_alleles = calculate_avg_different_alleles(population)
        allele_diff_history.append(avg_diff_alleles)
        avg_shannon_entropy = calculate_avg_shannon_entropy(population)
        entropy_history.append(avg_shannon_entropy)

        print(
            f"Gen {iteration}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, worst={stats['worst_fitness']}, "
            f"range={stats['fitness_range']}, selection_var={stats['selection_variance']:.4f}"
        )
        print(
            f"   top_avg_prob_ratio={stats['top_avg_prob_ratio']:.2f}, "
            f"   distance={avg_distance:.2f}, diff_alleles={avg_diff_alleles:.2f}, entropy={avg_shannon_entropy:.2f}"
        )

        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        gen_ticks = time.perf_counter_ns() - generation_start_ticks
        print(
            f"   CPU time={timing['generation_cpu_time']:.4f}s, elapsed={timing['elapsed_time']:.2f}s, "
            f"raw ticks={gen_ticks}, tick_time={gen_ticks/1e9:.6f}s"
        )

        mean_history.append(stats['mean'])
        worst_history.append(stats['worst_fitness'])
        fitness_distributions.append([cand.fitness for cand in population])

        if population[0].fitness == 0:
            print("Target reached!")
            termination_reason = "solution_found"
            converged_generation = iteration
            break

        mate(population, buffer)
        population, buffer = swap(population, buffer)

    return {
        "best_fitness_history": best_history,
        "mean_fitness_history": mean_history,
        "worst_fitness_history": worst_history,
        "fitness_distributions": fitness_distributions,
        "entropy_history": entropy_history,
        "allele_diff_history": allele_diff_history,
        "distance_history": distance_history,
        "converged_generation": converged_generation,
        "termination_reason": termination_reason
    }


# -----------------------------------------------------------------------------
# MAIN (STANDALONE EXECUTION)
# -----------------------------------------------------------------------------
def main():
    random.seed(time.time())
    population, buffer = init_population()
    overall_start_wall = time.time()

    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []
    entropy_history = []
    allele_diff_history = []
    distance_history = []

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
    print(f"Selection method: {ga_selection_method}, aging={ga_use_aging} (limit={ga_age_limit})")

    for iteration in range(ga_maxiter):
        if (time.time() - overall_start_wall) >= ga_max_runtime:
            print(f"Time limit of {ga_max_runtime} seconds reached after {iteration} generations.")
            break

        generation_start_cpu = time.process_time()
        generation_start_ticks = time.perf_counter_ns()

        calc_fitness(population)
        sort_by_fitness(population)
        print_best(population)

        stats = compute_fitness_statistics(population)

        avg_distance, actual_metric = calculate_avg_population_distance(population, ga_distance_metric)
        distance_history.append(avg_distance)
        avg_diff_alleles = calculate_avg_different_alleles(population)
        allele_diff_history.append(avg_diff_alleles)
        avg_shannon_entropy = calculate_avg_shannon_entropy(population)
        entropy_history.append(avg_shannon_entropy)

        print(
            f"generation {iteration}: mean fitness = {stats['mean']:.2f}, "
            f"std = {stats['std']:.2f}, worst = {stats['worst_fitness']}, "
            f"range = {stats['fitness_range']}"
        )
        print(
            f"   selection_var = {stats['selection_variance']:.4f}, top_avg_prob_ratio = {stats['top_avg_prob_ratio']:.2f}, "
            f"distance = {avg_distance:.2f}, diff_alleles = {avg_diff_alleles:.2f}, "
            f"entropy = {avg_shannon_entropy:.2f}"
        )

        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        gen_ticks = time.perf_counter_ns() - generation_start_ticks
        print(
            f"   CPU time = {timing['generation_cpu_time']:.4f}s, elapsed = {timing['elapsed_time']:.2f}s, "
            f"raw ticks = {gen_ticks}, tick_time = {gen_ticks/1e9:.6f}s"
        )

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

    plot_fitness_evolution(best_history, mean_history, worst_history)
    plot_fitness_boxplots(fitness_distributions)
    plot_entropy_evolution(entropy_history, allele_diff_history, distance_history)


if __name__ == "__main__":
    main()
