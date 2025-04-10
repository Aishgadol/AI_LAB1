import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# GLOBAL PARAMETERS - NEW/UPDATED
# -----------------------------------------------------------------------------
ga_selection_method = "tournament_probabilistic"      # "rws", "sus", "tournament_deterministic", "tournament_probabilistic"
ga_use_linear_scaling = True     # if True, apply linear scaling for RWS or SUS
ga_max_fitness_ratio = 2.0       #maximum ratio (scaled_best / scaled_avg) to limit dominance
ga_tournament_k = 3              #'k' for deterministic tournament
ga_tournament_k_prob = 3         #'k' for probabilistic (non-determinstic) tournament
ga_tournament_p = 0.75           #'p' for probabilistic (non-determinstic) tournament

ga_use_aging = True              #enable or disable aging-based survival
ga_age_limit = 100               #default age limit for individuals

# Existing parameters for GA (string matching defaults):
ga_popsize = 1000
ga_maxiter = 250
ga_elitrate = 0.10
ga_mutationrate = 0.55
ga_target = "testing string123 diff_chars"
ga_crossover_method = "two_point"  # "single", "two_point", or "uniform" (for strings)
ga_lcs_bonus = 5
ga_fitness_mode = "ascii"       # "ascii", "lcs", "combined"
ga_max_runtime = 600
ga_distance_metric = "levenshtein" # "ulam" or "levenshtein"

# -----------------------------------------------------------------------------
# NEW: ADD PROBLEM TYPE + PARAMETERS FOR BIN PACKING OR FUTURE EXPANSIONS
# -----------------------------------------------------------------------------
ga_problem_type = "string"  # "string" or "bin_packing" or "arc" (for future extension)

#for bin packing
ga_binpacking_items = [10, 8, 3, 7, 5, 9, 1, 2]  #example item sizes
ga_binpacking_bin_capacity = 15
ga_binpacking_alpha = 0.4  #weight for wasted space in fitness formula

# You may add additional parameters for future expansions (ARC, etc.) here.


# -----------------------------------------------------------------------------
# CANDIDATE CLASS WITH AGE
# -----------------------------------------------------------------------------
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene
        self.fitness = fitness
        self.age = 0  # track aging


# -----------------------------------------------------------------------------
# PROBLEM-SPECIFIC INITIALIZATIONS
# -----------------------------------------------------------------------------
def init_population_string():

    target_length = len(ga_target)
    population = []
    for _ in range(ga_popsize):
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    buffer = [Candidate('', 0) for _ in range(ga_popsize)]
    return population, buffer

#bin-packing population init (permutation)
def init_population_bin_packing():
    num_items = len(ga_binpacking_items)
    population = []
    for _ in range(ga_popsize):
        perm = list(range(num_items))
        random.shuffle(perm)
        population.append(Candidate(perm))
    # Make a buffer of the same shape
    buffer = [Candidate([0]*num_items) for _ in range(ga_popsize)]
    return population, buffer

# Placeholder for future ARC or other problems
def init_population_arc():
    """
    Stub for a future problem (Kaggle’s ARC).
    This function should create valid candidate representations for ARC.
    """
    raise NotImplementedError("ARC population initialization not yet implemented.")


def init_population():
    if ga_problem_type == "string":
        return init_population_string()
    elif ga_problem_type == "bin_packing":
        return init_population_bin_packing()
    elif ga_problem_type == "arc":
        return init_population_arc()
    else:
        # fallback to string
        return init_population_string()


# -----------------------------------------------------------------------------
# FITNESS UTILITIES FOR STRING MATCHING
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
        # fallback to Levenshtein
        return levenshtein_distance(s1, s2), True

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


# -----------------------------------------------------------------------------
# FITNESS FOR BIN PACKING (1D)
# -----------------------------------------------------------------------------
def calc_fitness_bin_packing(candidate_perm):
    """
    Each candidate.gene is a permutation of item indices.
    We'll place items in bins in the order given by the permutation.

    fitness = (number_of_bins) + alpha * (total_wasted_space / bin_capacity)

    Lower fitness is better.
    """
    bin_capacity = ga_binpacking_bin_capacity
    alpha = ga_binpacking_alpha
    items = ga_binpacking_items

    current_bin_space = bin_capacity
    num_bins = 1
    wasted = 0

    for idx in candidate_perm:
        size = items[idx]
        if size <= current_bin_space:
            # fits in current bin
            current_bin_space -= size
        else:
            # open a new bin
            wasted += current_bin_space
            num_bins += 1
            current_bin_space = bin_capacity - size

    # account for leftover in the final bin
    wasted += current_bin_space
    return num_bins + alpha * (wasted / bin_capacity)


# -----------------------------------------------------------------------------
# MASTER FITNESS FUNCTION DISPATCH
# -----------------------------------------------------------------------------
def calc_fitness(population):
    """
    If problem_type == 'string', use old logic: ASCII, LCS, combined, etc.
    If problem_type == 'bin_packing', use bin packing formula.
    (For ARC or other expansions, add more clauses.)
    """
    if ga_problem_type == "bin_packing":
        # BIN PACKING
        for candidate in population:
            candidate.fitness = calc_fitness_bin_packing(candidate.gene)
        return

    elif ga_problem_type == "string":
        target = ga_target
        target_length = len(target)
        for candidate in population:
            if ga_fitness_mode == "ascii":
                fitness = 0
                for i in range(target_length):
                    fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
                candidate.fitness = fitness

            elif ga_fitness_mode == "lcs":
                lcs_len = longest_common_subsequence(candidate.gene, target)
                candidate.fitness = target_length - lcs_len

            elif ga_fitness_mode == "combined":
                ascii_fit = 0
                for i in range(target_length):
                    ascii_fit += abs(ord(candidate.gene[i]) - ord(target[i]))
                lcs_len = longest_common_subsequence(candidate.gene, target)
                lcs_fit = target_length - lcs_len
                candidate.fitness = ascii_fit + ga_lcs_bonus * lcs_fit

            else:
                # default fallback is ASCII
                fitness = 0
                for i in range(target_length):
                    fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
                candidate.fitness = fitness

    else:
        # Fallback or future extension
        # For now, just do a simple "Hamming distance" if we get an unknown problem
        for candidate in population:
            # fallback if we can't interpret candidate.gene
            if isinstance(candidate.gene, str) and len(candidate.gene) == len(ga_target):
                # Hamming distance against ga_target
                candidate.fitness = sum(
                    1 for i in range(len(ga_target))
                    if candidate.gene[i] != ga_target[i]
                )
            else:
                # arbitrary fallback
                candidate.fitness = random.random() * 1000


def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)


# -----------------------------------------------------------------------------
# SELECTION: LINEAR SCALING FOR RWS / SUS
# -----------------------------------------------------------------------------
def linear_scale_fitness(population, max_ratio=2.0):
    raw_fitnesses = [c.fitness for c in population]
    f_min = min(raw_fitnesses)
    f_max = max(raw_fitnesses)
    if abs(f_max - f_min) < 1e-9:
        for c in population:
            c.scaled_fitness = 1.0
        return
    scores = [f_max - f for f in raw_fitnesses]  # invert so bigger -> better
    score_min, score_max = min(scores), max(scores)
    base_values = [s - score_min for s in scores]
    base_max = score_max - score_min
    base_avg = sum(base_values) / len(base_values) if len(base_values) > 0 else 0.0
    if abs(base_max) < 1e-9:
        for c in population:
            c.scaled_fitness = 1.0
        return

    ratio = (base_max / base_avg) if base_avg > 1e-9 else 1.0
    if ratio > max_ratio:
        a = max_ratio / ratio
    else:
        a = 1.0

    for i, c in enumerate(population):
        c.scaled_fitness = a * base_values[i]

    total_scaled = sum(c.scaled_fitness for c in population)
    if total_scaled < 1e-9:
        for c in population:
            c.scaled_fitness = 1.0


# -----------------------------------------------------------------------------
# SELECTION METHODS
# -----------------------------------------------------------------------------
def rws_select_one(population):
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
    total = sum(c.scaled_fitness for c in population)
    if total < 1e-9:
        return [random.choice(population) for _ in range(num_parents)]

    distance = total / num_parents
    start = random.random() * distance
    chosen = []
    running_sum = 0.0
    idx = 0
    for _ in range(num_parents):
        pointer = start + _ * distance
        while running_sum < pointer and idx < len(population) - 1:
            running_sum += population[idx].scaled_fitness
            idx += 1
        chosen.append(population[idx - 1])
    return chosen

def tournament_deterministic_select_one(population, k=2):
    contenders = random.sample(population, k)
    contenders.sort(key=lambda c: c.fitness)
    return contenders[0]

def tournament_probabilistic_select_one(population, k=2, p=0.75):
    contenders = random.sample(population, k)
    contenders.sort(key=lambda c: c.fitness)
    r = random.random()
    cumulative = 0.0
    for i, cand in enumerate(contenders):
        prob_i = p * ((1 - p) ** i)
        cumulative += prob_i
        if r <= cumulative:
            return cand
    return contenders[-1]

def old_roulette_wheel_select(candidates):
    inv_fitnesses = [1.0 / (1.0 + c.fitness) for c in candidates]
    total_inv = sum(inv_fitnesses)
    pick = random.random()
    running_sum = 0.0
    for i, inv in enumerate(inv_fitnesses):
        running_sum += inv / total_inv
        if pick <= running_sum:
            return candidates[i]
    return candidates[-1]

def select_one_parent(population):
    method = ga_selection_method.lower()
    if method == "rws":
        return rws_select_one(population)
    elif method == "sus":
        return sus_select_parents(population, 1)[0]
    elif method == "tournament_deterministic":
        return tournament_deterministic_select_one(population, ga_tournament_k)
    elif method == "tournament_probabilistic":
        return tournament_probabilistic_select_one(population, ga_tournament_k_prob, ga_tournament_p)
    else:
        return old_roulette_wheel_select(population)


# -----------------------------------------------------------------------------
# CROSSOVER OPERATORS (STRING-BASED)
# -----------------------------------------------------------------------------
def single_point_crossover_str(parent1, parent2, length):
    point = random.randint(0, length - 1)
    c1 = parent1.gene[:point] + parent2.gene[point:]
    c2 = parent2.gene[:point] + parent1.gene[point:]
    return c1, c2

def two_point_crossover_str(parent1, parent2, length):
    p1, p2 = sorted(random.sample(range(length), 2))
    c1 = (parent1.gene[:p1] + parent2.gene[p1:p2] + parent1.gene[p2:])
    c2 = (parent2.gene[:p1] + parent1.gene[p1:p2] + parent2.gene[p2:])
    return c1, c2

def uniform_crossover_str(parent1, parent2, length):
    c1 = []
    c2 = []
    for i in range(length):
        if random.random() < 0.5:
            c1.append(parent1.gene[i])
            c2.append(parent2.gene[i])
        else:
            c1.append(parent2.gene[i])
            c2.append(parent1.gene[i])
    return ''.join(c1), ''.join(c2)


# -----------------------------------------------------------------------------
# CROSSOVER OPERATORS (PERMUTATION-BASED)
# -----------------------------------------------------------------------------
def pmx_crossover_perm(p1, p2):
    """
    PMX crossover for permutations. p1, p2 are lists of ints.
    Returns two children as lists of ints.
    """
    length = len(p1)
    c1, c2 = [None]*length, [None]*length
    cxpoint1, cxpoint2 = sorted(random.sample(range(length), 2))

    # copy the slice
    for i in range(cxpoint1, cxpoint2):
        c1[i] = p1[i]
        c2[i] = p2[i]

    # fill the rest
    def pmx_fill(c, donor, start, end):
        for i in range(start, end):
            if donor[i] not in c:
                pos = i
                val = donor[i]
                while c[pos] is not None:
                    pos = donor.index(p1[pos])  # or p2 if c==c2
                c[pos] = val

    pmx_fill(c1, p2, cxpoint1, cxpoint2)
    pmx_fill(c2, p1, cxpoint1, cxpoint2)

    # fill remaining None with direct copy
    for i in range(length):
        if c1[i] is None:
            c1[i] = p2[i]
        if c2[i] is None:
            c2[i] = p1[i]
    return c1, c2

def ox_crossover_perm(p1, p2):
    """
    Ordered Crossover (OX) for permutations.
    """
    length = len(p1)
    c1, c2 = [None]*length, [None]*length
    start, end = sorted(random.sample(range(length), 2))

    # copy slice
    c1[start:end] = p1[start:end]
    c2[start:end] = p2[start:end]

    # fill the rest in order
    def fill_ox(child, parent, start, end):
        pos = end
        if pos >= length:
            pos = 0
        for x in parent:
            if x not in child:
                child[pos] = x
                pos += 1
                if pos >= length:
                    pos = 0

    fill_ox(c1, p2, start, end)
    fill_ox(c2, p1, start, end)
    return c1, c2

def cx_crossover_perm(p1, p2):
    """
    Cycle Crossover (CX) for permutations.
    """
    length = len(p1)
    c1, c2 = [None]*length, [None]*length
    # start cycle from index 0
    index = 0
    cycle = 1
    used = set()

    while True:
        if index in used:
            # find next free index
            free_positions = [i for i in range(length) if i not in used]
            if not free_positions:
                break
            index = free_positions[0]
            cycle += 1
        start = index
        val1 = p1[index]
        while True:
            c1[index] = p1[index]
            c2[index] = p2[index]
            used.add(index)
            index = p1.index(p2[index])
            if p1[index] == val1:
                c1[index] = p1[index]
                c2[index] = p2[index]
                used.add(index)
                break

    # fill any None with parent's genes
    for i in range(length):
        if c1[i] is None:
            c1[i] = p2[i]
        if c2[i] is None:
            c2[i] = p1[i]
    return c1, c2


# -----------------------------------------------------------------------------
# MASTER CROSSOVER DISPATCH
# -----------------------------------------------------------------------------
def crossover_operator(parent1, parent2):
    """
    If ga_problem_type = 'string', we use the string crossovers.
    If ga_problem_type = 'bin_packing', we use PMX/OX/CX (permutation).
    Fallbacks: 'single' for string-based if invalid, 'cx' for permutations if invalid.
    """
    if ga_problem_type == "string":
        length = len(parent1.gene)
        if ga_crossover_method == "single":
            return single_point_crossover_str(parent1, parent2, length)
        elif ga_crossover_method == "two_point":
            return two_point_crossover_str(parent1, parent2, length)
        elif ga_crossover_method == "uniform":
            return uniform_crossover_str(parent1, parent2, length)
        else:
            # fallback
            return single_point_crossover_str(parent1, parent2, length)

    elif ga_problem_type == "bin_packing":
        p1 = parent1.gene
        p2 = parent2.gene
        if ga_crossover_method == "pmx":
            return pmx_crossover_perm(p1, p2)
        elif ga_crossover_method == "ox":
            return ox_crossover_perm(p1, p2)
        elif ga_crossover_method == "cx":
            return cx_crossover_perm(p1, p2)
        else:
            # fallback to cycle
            return cx_crossover_perm(p1, p2)

    else:
        # fallback
        # if it's a string in unknown domain, do single-point
        # if it's a list, do cycle
        if isinstance(parent1.gene, str):
            return single_point_crossover_str(parent1, parent2, len(parent1.gene))
        elif isinstance(parent1.gene, list):
            return cx_crossover_perm(parent1.gene, parent2.gene)
        # otherwise no idea
        return parent1.gene, parent2.gene


# -----------------------------------------------------------------------------
# MUTATION OPERATORS (STRING-BASED)
# -----------------------------------------------------------------------------
def mutate_string(candidate):
    target_length = len(ga_target)
    pos = random.randint(0, target_length - 1)
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    new_val = 32 + ((old_val - 32 + delta) % (121 - 32 + 1))
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)


# -----------------------------------------------------------------------------
# MUTATION OPERATORS (PERMUTATION-BASED)
# -----------------------------------------------------------------------------
def exchange_mutation(candidate):
    """
    Simple swap mutation. Fallback that works for any permutation length >= 2.
    """
    gene = candidate.gene
    if len(gene) < 2:
        return
    i, j = random.sample(range(len(gene)), 2)
    gene[i], gene[j] = gene[j], gene[i]

def displacement_mutation(candidate):
    """
    Removes one element and inserts it at a random position.
    """
    gene = candidate.gene
    i = random.randint(0, len(gene)-1)
    elem = gene.pop(i)
    j = random.randint(0, len(gene))
    gene.insert(j, elem)

def insertion_mutation(candidate):
    """
    Also known as 'insert mutation':
    pick two positions, remove the element at i and insert before j.
    """
    gene = candidate.gene
    i = random.randint(0, len(gene)-1)
    elem = gene.pop(i)
    j = random.randint(0, len(gene))
    gene.insert(j, elem)

def simple_inversion_mutation(candidate):
    """
    Reverse a substring of the permutation.
    """
    gene = candidate.gene
    start, end = sorted(random.sample(range(len(gene)), 2))
    gene[start:end] = reversed(gene[start:end])

def scramble_mutation(candidate):
    """
    Scramble the elements in a random substring.
    """
    gene = candidate.gene
    start, end = sorted(random.sample(range(len(gene)), 2))
    subset = gene[start:end]
    random.shuffle(subset)
    gene[start:end] = subset


# -----------------------------------------------------------------------------
# MASTER MUTATION DISPATCH
# -----------------------------------------------------------------------------
# Let’s define a new global or “semi-global” for mutation operator if you like
ga_mutation_operator = "exchange"  # fallback operator for permutations

def mutate(candidate):
    """
    If problem_type = 'string', do the original ASCII-based mutation.
    If problem_type = 'bin_packing', choose among displacement, exchange,
    insertion, simple_inversion, scramble, etc. Fallback = exchange.
    """
    if ga_problem_type == "string":
        mutate_string(candidate)
    elif ga_problem_type == "bin_packing":
        # pick mutation operator from ga_mutation_operator
        if ga_mutation_operator == "displacement":
            displacement_mutation(candidate)
        elif ga_mutation_operator == "insertion":
            insertion_mutation(candidate)
        elif ga_mutation_operator == "simple_inversion":
            simple_inversion_mutation(candidate)
        elif ga_mutation_operator == "scramble":
            scramble_mutation(candidate)
        elif ga_mutation_operator == "exchange":
            exchange_mutation(candidate)
        else:
            # fallback
            exchange_mutation(candidate)
    else:
        # fallback
        if isinstance(candidate.gene, str):
            mutate_string(candidate)
        elif isinstance(candidate.gene, list):
            exchange_mutation(candidate)
        else:
            # do nothing
            pass


# -----------------------------------------------------------------------------
# ELITISM
# -----------------------------------------------------------------------------
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        buffer[i].gene = population[i].gene[:]
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age + 1


# -----------------------------------------------------------------------------
# AGING-BASED SURVIVAL
# -----------------------------------------------------------------------------
def apply_aging_replacement(population):
    if not ga_use_aging:
        return
    for i in range(len(population)):
        if population[i].age > ga_age_limit:
            parent1 = select_one_parent(population)
            parent2 = select_one_parent(population)
            c1, _ = crossover_operator(parent1, parent2)
            new_cand = Candidate(c1, fitness=0)
            new_cand.age = 0
            population[i] = new_cand


# -----------------------------------------------------------------------------
# MATE FUNCTION (BREED NEXT GENERATION)
# -----------------------------------------------------------------------------
def mate(population, buffer):
    elite_size = int(ga_popsize * ga_elitrate)

    # 1) Elitism
    elitism(population, buffer, elite_size)

    # 2) If using RWS or SUS:
    method = ga_selection_method.lower()
    if method in ["rws", "sus"]:
        if ga_use_linear_scaling:
            linear_scale_fitness(population, ga_max_fitness_ratio)
        else:
            for c in population:
                c.scaled_fitness = 1.0 / (1.0 + c.fitness)

    # 3) Fill the remainder
    i = elite_size
    while i < ga_popsize - 1:
        parent1 = select_one_parent(population)
        parent2 = select_one_parent(population)

        child1, child2 = crossover_operator(parent1, parent2)

        # Copy children into buffer
        if isinstance(child1, list):
            buffer[i].gene = child1[:]
        else:
            buffer[i].gene = child1
        buffer[i].fitness = 0
        buffer[i].age = 0

        if isinstance(child2, list):
            buffer[i+1].gene = child2[:]
        else:
            buffer[i+1].gene = child2
        buffer[i+1].fitness = 0
        buffer[i+1].age = 0

        if random.random() < ga_mutationrate:
            mutate(buffer[i])
        if random.random() < ga_mutationrate:
            mutate(buffer[i + 1])

        i += 2

    # 4) If population size is odd, copy the last one
    if ga_popsize % 2 == 1 and i < ga_popsize:
        buffer[i].gene = buffer[i - 1].gene
        buffer[i].fitness = buffer[i - 1].fitness
        buffer[i].age = buffer[i - 1].age
        if random.random() < ga_mutationrate:
            mutate(buffer[i])

    # 5) Aging-based replacement
    apply_aging_replacement(buffer)


def swap(population, buffer):
    return buffer, population


def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness}) age={best.age}")


# -----------------------------------------------------------------------------
# STATISTICS, DIVERSITY, TIMING
# -----------------------------------------------------------------------------
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


def calculate_avg_different_alleles(population):
    """
    For strings, measure allele differences. For permutations, fallback to partial measure, or skip.
    If all are the same representation, proceed. If mixing, just skip.
    """
    # If string-based:
    if ga_problem_type == "string" and len(population) >= 2:
        total_diff = 0
        count = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_diff += different_alleles(population[i].gene, population[j].gene)
                count += 1
        if count == 0:
            return 0
        return total_diff / count
    else:
        # You may define a measure for permutations or skip if not needed
        # e.g., the average # of positions that differ among permutations
        return 0

def calculate_avg_population_distance(population, distance_metric="levenshtein"):
    global ga_distance_metric
    if ga_problem_type == "string":
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
            print("Warning: Had to fall back to Levenshtein distance (Ulam conditions not met).")

        if count == 0:
            return 0, ga_distance_metric
        return total_distance / count, ga_distance_metric

    else:
        # For bin_packing or other problem, you may define a distance measure or skip
        return 0, distance_metric


def calculate_avg_shannon_entropy(population):
    """
    For string-based GA, we measure position-by-position entropy.
    For permutations, skip or define a separate measure if desired.
    """
    if not population:
        return 0.0
    if ga_problem_type == "string":
        gene_length = len(population[0].gene)
        position_entropies = []
        for position in range(gene_length):
            position_chars = [cand.gene[position] for cand in population]
            char_count = {}
            for ch in position_chars:
                char_count[ch] = char_count.get(ch, 0) + 1
            entropy = 0.0
            pop_size = len(population)
            for c_count in char_count.values():
                p = c_count / pop_size
                entropy -= p * math.log2(p)
            position_entropies.append(entropy)
        return sum(position_entropies) / gene_length
    else:
        # For permutations or other domains, we can define a custom measure or skip
        return 0.0


# -----------------------------------------------------------------------------
# VISUALIZATIONS (UNCHANGED)
# -----------------------------------------------------------------------------
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
    plt.title(f"fitness distribution per generation (crossover: {ga_crossover_method})")
    plt.xticks(range(1, len(indices) + 1), xtick_labels)
    plt.grid(True)

    # annotate data
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

    plt.plot(generations, entropy_history, label="Shannon Entropy", linewidth=2, color='purple')
    plt.plot(generations, allele_diff_history, label="Avg Different Alleles", linewidth=2, color='red')
    plt.plot(generations, distance_history, label="Avg Levenshtein Distance", linewidth=2, color='blue')

    if len(generations) > 1:
        label_points = [int(i * (len(generations) - 1) / 9) for i in range(10)]
        for idx in label_points:
            plt.annotate(f"Entropy: {entropy_history[idx]:.2f}",
                         (idx, entropy_history[idx]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         color='purple',
                         fontsize=8)
            plt.annotate(f"Alleles: {allele_diff_history[idx]:.2f}",
                         (idx, allele_diff_history[idx]),
                         textcoords="offset points",
                         xytext=(0, -15),
                         ha='center',
                         color='red',
                         fontsize=8)
            plt.annotate(f"Levenshtein: {distance_history[idx]:.2f}",
                         (idx, distance_history[idx]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         color='blue',
                         fontsize=8)

    plt.xlabel("Generation")
    plt.ylabel("Diversity Metrics")
    plt.title("Population Diversity Metrics per Generation")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# GA EXECUTION (PROGRAMMATIC ENTRY POINT)
# -----------------------------------------------------------------------------
def run_ga(
        problem_type="string",          # NEW: "string" or "bin_packing" or "arc"
        crossover_method="two_point",   # for strings: "single", "two_point", "uniform"
        # for permutations: "pmx", "ox", "cx"
        fitness_mode="combined",        # string-based: "ascii", "lcs", "combined"
        lcs_bonus=5,
        mutation_rate=0.55,
        population_size=500,
        max_runtime=120,
        distance_metric="levenshtein",  # string-based only
        selection_method="rws",
        use_linear_scaling=True,
        max_fitness_ratio=2.0,
        use_aging=False,
        age_limit=100,
        tournament_k=3,
        tournament_k_prob=3,
        tournament_p=0.75,
        # Extra bin-packing config if needed:
        binpacking_items=None,
        binpacking_bin_capacity=15,
        binpacking_alpha=0.5,
        # Which mutation operator for permutations
        mutation_operator="exchange"
):
    """
    Run the GA with the specified settings, returning a dict of stats.
    This includes a new 'problem_type' argument that selects domain-specific logic.

    For 1D bin packing:
      - problem_type="bin_packing"
      - ga_binpacking_items, ga_binpacking_bin_capacity, ga_binpacking_alpha can be set here
      - crossover_method can be "pmx", "ox", or "cx"
      - mutation_operator can be "exchange", "displacement", "insertion", "simple_inversion", "scramble"

    For future expansions (ARC, etc.), define new problem logic and pass problem_type="arc".
    """
    global ga_problem_type, ga_crossover_method, ga_fitness_mode, ga_lcs_bonus
    global ga_mutationrate, ga_popsize, ga_distance_metric, ga_max_runtime
    global ga_selection_method, ga_use_linear_scaling, ga_max_fitness_ratio
    global ga_use_aging, ga_age_limit, ga_tournament_k, ga_tournament_k_prob, ga_tournament_p
    global ga_binpacking_items, ga_binpacking_bin_capacity, ga_binpacking_alpha
    global ga_mutation_operator

    # Update global settings
    ga_problem_type = problem_type
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

    # Bin-packing domain parameters
    if binpacking_items is not None:
        ga_binpacking_items = binpacking_items
    ga_binpacking_bin_capacity = binpacking_bin_capacity
    ga_binpacking_alpha = binpacking_alpha

    # Mutation operator for permutations
    ga_mutation_operator = mutation_operator

    # Initialize RNG and population
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

        avg_distance, actual_metric = calculate_avg_population_distance(population, ga_distance_metric)
        distance_history.append(avg_distance)
        avg_diff_alleles = calculate_avg_different_alleles(population)
        allele_diff_history.append(avg_diff_alleles)
        avg_shannon_entropy = calculate_avg_shannon_entropy(population)
        entropy_history.append(avg_shannon_entropy)

        # minor fix for printing selection_variance
        if stats['selection_variance'] > 0:
            # scale for printing convenience
            factor = 10 ** (-math.floor(math.log10(stats['selection_variance'])))
            stats['selection_variance'] = stats['selection_variance'] * factor

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

        # If solution found
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
# MAIN (STANDALONE EXECUTION) - STILL FOR STRING DEMO
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

    if ga_problem_type == "string":
        # For strings, we might want single/two_point/uniform crossovers
        print(f"starting genetic algorithm (string mode) with {ga_crossover_method} crossover...")
        if ga_fitness_mode not in ["ascii", "lcs", "combined"]:
            print("no fitness mode selected, defaulting to ascii")
        else:
            print(f"using fitness mode: {ga_fitness_mode}")
    elif ga_problem_type == "bin_packing":
        print("starting genetic algorithm (bin packing mode) ...")
        print(f"crossover method = {ga_crossover_method}, mutation operator = {ga_mutation_operator}")

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
        if stats['selection_variance'] > 0:
            stats['selection_variance'] = stats['selection_variance'] * 100000000
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

    # Plot results
    plot_fitness_evolution(best_history, mean_history, worst_history)
    plot_fitness_boxplots(fitness_distributions)
    plot_entropy_evolution(entropy_history, allele_diff_history, distance_history)


if __name__ == "__main__":
    main()
