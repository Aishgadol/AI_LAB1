import random
import time
import math

# genetic algorithm parameters
ga_popsize = 1000  # population size
ga_maxiter = 250   # max number of generations
ga_elitrate = 0.10 # fraction of population to carry forward unchanged
ga_mutationrate = 0.55 # chance that a child mutates
ga_max_runtime = 600   # max runtime in seconds

# problem selection: "string" or "binpacking" or "arc" (future)
ga_problem_type = "string"

# string-matching parameters
ga_target = "testing string123 diff_chars"
ga_fitness_mode = "ascii"     # "ascii", "lcs", "combined"
ga_crossover_method = "two_point"  # "single", "two_point", "uniform" for strings
ga_lcs_bonus = 5
ga_distance_metric = "levenshtein" # "ulam" or "levenshtein"

# bin-packing parameters
ga_bin_capacity = 15   # capacity of each bin
ga_bin_items = [10,8,3,7,5,9,1,2] # example item sizes
ga_bin_alpha = 0.4     # weight for leftover space
ga_bin_max_bins = 20   # an upper bound on how many bins we allow in bit-string

# selection parameters
ga_selection_method = "tournament_probabilistic" # "rws","sus","tournament_deterministic","tournament_probabilistic"
ga_use_linear_scaling = True
ga_max_fitness_ratio = 2.0
ga_tournament_k = 3
ga_tournament_k_prob = 3
ga_tournament_p = 0.75

# aging-based survival
ga_use_aging = True
ga_age_limit = 100

# large penalty for invalid solutions
INVALID_PENALTY = 9999999

# candidate class
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene
        self.fitness = fitness
        self.age = 0

###############################################################################
# population initialization
###############################################################################

def init_population():
    # dispatch to the correct init based on problem type
    if ga_problem_type == "string":
        return init_population_string()
    elif ga_problem_type == "binpacking":
        return init_population_binpacking()
    elif ga_problem_type == "arc":
        return init_population_arc()  # placeholder
    else:
        # default to string init
        return init_population_string()

def init_population_string():
    # create random strings matching target length
    length = len(ga_target)
    population = []
    for _ in range(ga_popsize):
        gene = ''.join(chr(random.randint(32, 121)) for __ in range(length))
        population.append(Candidate(gene))
    buffer = [Candidate('') for __ in range(ga_popsize)]
    return population, buffer

def init_population_binpacking():
    """
    each candidate is a bit-string: gene[i] = bin_index for item i
    we place all items by basic first fit, then replicate that solution
    """
    # create one first-fit solution
    best_fit_gene = create_first_fit_assignment(ga_bin_items, ga_bin_capacity, ga_bin_max_bins)

    # replicate it across population
    population = []
    for _ in range(ga_popsize):
        # copy the same gene
        population.append(Candidate(best_fit_gene[:]))
    buffer = [Candidate([0]*len(ga_bin_items)) for __ in range(ga_popsize)]
    return population, buffer

def init_population_arc():
    # placeholder for future expansions
    raise NotImplementedError("ARC population init not yet implemented.")

def create_first_fit_assignment(items, capacity, max_bins):
    """
    items: list of item sizes
    capacity: capacity of each bin
    max_bins: upper bound of bin indexes
    returns a list 'assignment' where assignment[i] is bin index of item i
    """
    assignment = [0]*len(items)
    bin_remaining = [capacity]*max_bins
    next_bin = 0

    # place each item in first bin where it fits
    for i, size in enumerate(items):
        placed = False
        for b in range(next_bin+1):
            if bin_remaining[b] >= size:
                assignment[i] = b
                bin_remaining[b] -= size
                placed = True
                break
        if not placed:
            # open a new bin if possible
            next_bin += 1
            if next_bin >= max_bins:
                # we are out of bins, degrade gracefully
                assignment[i] = max_bins - 1
            else:
                assignment[i] = next_bin
                bin_remaining[next_bin] -= size
    return assignment

###############################################################################
# fitness calculation
###############################################################################

def calc_fitness(population):
    # dispatch
    if ga_problem_type == "string":
        calc_fitness_string(population)
    elif ga_problem_type == "binpacking":
        calc_fitness_binpacking(population)
    elif ga_problem_type == "arc":
        # placeholder
        for c in population:
            c.fitness = random.random() * 999
    else:
        calc_fitness_string(population)

def calc_fitness_string(population):
    # compute the selected fitness mode
    length = len(ga_target)
    for cand in population:
        if ga_fitness_mode == "ascii":
            fit = 0
            for i in range(length):
                fit += abs(ord(cand.gene[i]) - ord(ga_target[i]))
            cand.fitness = fit

        elif ga_fitness_mode == "lcs":
            lcs_len = longest_common_subsequence(cand.gene, ga_target)
            cand.fitness = length - lcs_len

        elif ga_fitness_mode == "combined":
            ascii_fit = 0
            for i in range(length):
                ascii_fit += abs(ord(cand.gene[i]) - ord(ga_target[i]))
            lcs_len = longest_common_subsequence(cand.gene, ga_target)
            lcs_fit = length - lcs_len
            cand.fitness = ascii_fit + ga_lcs_bonus * lcs_fit

        else:
            # fallback
            fit = 0
            for i in range(length):
                fit += abs(ord(cand.gene[i]) - ord(ga_target[i]))
            cand.fitness = fit

def calc_fitness_binpacking(population):
    """
    gene is a list: gene[i] = bin index for item i
    use the bin indexes to compute total bins used, leftover, etc.
    if any bin is over capacity -> assign large penalty
    also incorporate alpha * leftover/capacity
    """
    for cand in population:
        assignment = cand.gene
        # track bin usage
        bin_usage = {}
        for i, bin_idx in enumerate(assignment):
            # penalize out-of-range bin index
            if bin_idx < 0 or bin_idx >= ga_bin_max_bins:
                cand.fitness = INVALID_PENALTY
                break
            bin_usage[bin_idx] = bin_usage.get(bin_idx, 0) + ga_bin_items[i]
        else:
            # if we didn't break:
            # check capacity
            used_bins = 0
            leftover = 0
            for b, usage in bin_usage.items():
                used_bins += 1
                if usage > ga_bin_capacity:
                    # penalize invalid arrangement
                    cand.fitness = INVALID_PENALTY
                    break
                leftover += (ga_bin_capacity - usage)
            else:
                # if still no break
                if used_bins == 0:
                    # fallback if no items
                    cand.fitness = 0
                else:
                    cand.fitness = used_bins + ga_bin_alpha * (leftover / ga_bin_capacity)
                continue
        # if we get here from a break, do nothing. fitness already set to penalty or 0.

###############################################################################
# string utility
###############################################################################

def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

###############################################################################
# selection
###############################################################################

def select_one_parent(pop):
    # check method
    method = ga_selection_method.lower()
    if method == "rws":
        return select_rws(pop)
    elif method == "sus":
        return select_sus(pop, 1)[0]
    elif method == "tournament_deterministic":
        return select_tournament_deterministic(pop, ga_tournament_k)
    elif method == "tournament_probabilistic":
        return select_tournament_probabilistic(pop, ga_tournament_k_prob, ga_tournament_p)
    else:
        return select_old_roulette(pop)

def select_rws(pop):
    total = sum(c.scaled_fitness for c in pop)
    if total <= 0:
        return random.choice(pop)
    pick = random.random() * total
    running = 0
    for c in pop:
        running += c.scaled_fitness
        if running >= pick:
            return c
    return pop[-1]

def select_sus(pop, n):
    total = sum(c.scaled_fitness for c in pop)
    if total <= 0:
        return [random.choice(pop) for _ in range(n)]
    distance = total / n
    start = random.random() * distance
    chosen = []
    running_sum = 0
    idx = 0
    for i in range(n):
        pointer = start + i*distance
        while running_sum < pointer and idx < len(pop)-1:
            running_sum += pop[idx].scaled_fitness
            idx += 1
        chosen.append(pop[idx-1])
    return chosen

def select_tournament_deterministic(pop, k=2):
    contenders = random.sample(pop, k)
    contenders.sort(key=lambda c: c.fitness)
    return contenders[0]

def select_tournament_probabilistic(pop, k=2, p=0.75):
    contenders = random.sample(pop, k)
    contenders.sort(key=lambda c: c.fitness)
    r = random.random()
    cumulative = 0
    for i, cand in enumerate(contenders):
        prob_i = p * ((1-p)**i)
        cumulative += prob_i
        if r <= cumulative:
            return cand
    return contenders[-1]

def select_old_roulette(pop):
    # fallback
    inv_fits = [1/(1 + c.fitness) if c.fitness >= 0 else 1 for c in pop]
    s = sum(inv_fits)
    if s == 0:
        return random.choice(pop)
    pick = random.random() * s
    running = 0
    for i, inv in enumerate(inv_fits):
        running += inv
        if running >= pick:
            return pop[i]
    return pop[-1]

###############################################################################
# fitness scaling
###############################################################################

def linear_scale(pop, max_ratio):
    raw = [c.fitness for c in pop]
    f_min = min(raw)
    f_max = max(raw)
    if abs(f_max - f_min) < 1e-9:
        for c in pop:
            c.scaled_fitness = 1.0
        return
    # invert
    scores = [f_max - x for x in raw]
    s_min = min(scores)
    s_max = max(scores)
    base = [s - s_min for s in scores]
    base_max = s_max - s_min
    base_avg = sum(base)/len(base)
    if base_max < 1e-9 or base_avg < 1e-9:
        for c in pop:
            c.scaled_fitness = 1.0
        return
    ratio = base_max / base_avg
    if ratio > max_ratio:
        a = max_ratio / ratio
    else:
        a = 1.0
    for i, c in enumerate(pop):
        c.scaled_fitness = a * base[i]
    tot = sum(c.scaled_fitness for c in pop)
    if tot < 1e-9:
        for c in pop:
            c.scaled_fitness = 1.0

###############################################################################
# crossover
###############################################################################

def crossover(parent1, parent2):
    # dispatch
    if ga_problem_type == "string":
        return crossover_string(parent1, parent2)
    elif ga_problem_type == "binpacking":
        return crossover_binpacking(parent1, parent2)
    elif ga_problem_type == "arc":
        return (parent1.gene, parent2.gene)  # placeholder
    else:
        return crossover_string(parent1, parent2)

def crossover_string(p1, p2):
    length = len(p1.gene)
    if ga_crossover_method == "single":
        return single_point_crossover_str(p1, p2, length)
    elif ga_crossover_method == "two_point":
        return two_point_crossover_str(p1, p2, length)
    elif ga_crossover_method == "uniform":
        return uniform_crossover_str(p1, p2, length)
    else:
        return single_point_crossover_str(p1, p2, length)

def single_point_crossover_str(p1, p2, n):
    point = random.randint(0, n-1)
    c1 = p1.gene[:point] + p2.gene[point:]
    c2 = p2.gene[:point] + p1.gene[point:]
    return c1, c2

def two_point_crossover_str(p1, p2, n):
    a, b = sorted(random.sample(range(n),2))
    c1 = p1.gene[:a] + p2.gene[a:b] + p1.gene[b:]
    c2 = p2.gene[:a] + p1.gene[a:b] + p2.gene[b:]
    return c1, c2

def uniform_crossover_str(p1, p2, n):
    c1, c2 = [], []
    for i in range(n):
        if random.random()<0.5:
            c1.append(p1.gene[i])
            c2.append(p2.gene[i])
        else:
            c1.append(p2.gene[i])
            c2.append(p1.gene[i])
    return ''.join(c1), ''.join(c2)

def crossover_binpacking(p1, p2):
    """
    both genes are lists of bin indexes of equal length
    allow user to pick from single, two, uniform as well
    """
    n = len(p1.gene)
    if ga_crossover_method == "single":
        return single_point_crossover_list(p1.gene, p2.gene, n)
    elif ga_crossover_method == "two_point":
        return two_point_crossover_list(p1.gene, p2.gene, n)
    elif ga_crossover_method == "uniform":
        return uniform_crossover_list(p1.gene, p2.gene, n)
    else:
        return single_point_crossover_list(p1.gene, p2.gene, n)

def single_point_crossover_list(g1, g2, n):
    pt = random.randint(0, n-1)
    c1 = g1[:pt] + g2[pt:]
    c2 = g2[:pt] + g1[pt:]
    return c1, c2

def two_point_crossover_list(g1, g2, n):
    a, b = sorted(random.sample(range(n), 2))
    c1 = g1[:a] + g2[a:b] + g1[b:]
    c2 = g2[:a] + g1[a:b] + g2[b:]
    return c1, c2

def uniform_crossover_list(g1, g2, n):
    c1, c2 = [], []
    for i in range(n):
        if random.random()<0.5:
            c1.append(g1[i])
            c2.append(g2[i])
        else:
            c1.append(g2[i])
            c2.append(g1[i])
    return c1, c2

###############################################################################
# mutation
###############################################################################

def mutate(candidate):
    # dispatch
    if ga_problem_type == "string":
        mutate_string(candidate)
    elif ga_problem_type == "binpacking":
        mutate_binpacking(candidate)
    elif ga_problem_type == "arc":
        pass  # placeholder
    else:
        mutate_string(candidate)

def mutate_string(cand):
    n = len(ga_target)
    pos = random.randint(0, n-1)
    old_val = ord(cand.gene[pos])
    delta = random.randint(32,121)
    new_val = 32 + ((old_val - 32 + delta) % (121-32+1))
    gene_list = list(cand.gene)
    gene_list[pos] = chr(new_val)
    cand.gene = ''.join(gene_list)

def mutate_binpacking(cand):
    """
    picks one item at random, reassigns it to a random bin index
    we do not fix capacity here - we let fitness penalize if invalid
    """
    gene = cand.gene
    idx = random.randint(0, len(gene)-1)
    new_bin = random.randint(0, ga_bin_max_bins-1)
    gene[idx] = new_bin

###############################################################################
# elitism and aging
###############################################################################

def elitism(pop, buf, elite_size):
    for i in range(elite_size):
        buf[i].gene = pop[i].gene[:] if isinstance(pop[i].gene, list) else pop[i].gene
        buf[i].fitness = pop[i].fitness
        buf[i].age = pop[i].age + 1

def apply_aging_replacement(pop):
    if not ga_use_aging:
        return
    for i in range(len(pop)):
        if pop[i].age > ga_age_limit:
            # kill it off and generate a new child from random parents
            p1 = select_one_parent(pop)
            p2 = select_one_parent(pop)
            c1, _ = crossover(p1, p2)
            pop[i] = Candidate(c1, 0)

###############################################################################
# mate: builds next generation
###############################################################################

def mate(pop, buf):
    elite_size = int(ga_popsize * ga_elitrate)

    # elitism
    elitism(pop, buf, elite_size)

    # if using rws or sus, we do fitness scaling
    if ga_selection_method.lower() in ["rws","sus"]:
        if ga_use_linear_scaling:
            linear_scale(pop, ga_max_fitness_ratio)
        else:
            # fallback
            for c in pop:
                c.scaled_fitness = 1/(1 + c.fitness) if c.fitness>=0 else 1

    # fill buffer
    i = elite_size
    while i < ga_popsize - 1:
        p1 = select_one_parent(pop)
        p2 = select_one_parent(pop)
        c1, c2 = crossover(p1, p2)

        # store children
        if isinstance(c1, list):
            buf[i].gene = c1[:]
        else:
            buf[i].gene = c1
        buf[i].fitness = 0
        buf[i].age = 0

        if isinstance(c2, list):
            buf[i+1].gene = c2[:]
        else:
            buf[i+1].gene = c2
        buf[i+1].fitness = 0
        buf[i+1].age = 0

        # mutate
        if random.random() < ga_mutationrate:
            mutate(buf[i])
        if random.random() < ga_mutationrate:
            mutate(buf[i+1])

        i += 2

    # handle odd population size
    if ga_popsize % 2 == 1 and i < ga_popsize:
        buf[i].gene = buf[i-1].gene
        buf[i].fitness = buf[i-1].fitness
        buf[i].age = buf[i-1].age
        if random.random() < ga_mutationrate:
            mutate(buf[i])

    # aging
    apply_aging_replacement(buf)

###############################################################################
# main loop helpers
###############################################################################

def swap(pop, buf):
    return buf, pop

def print_best(pop):
    best = pop[0]
    print(f"best: {best.gene}  fitness={best.fitness}  age={best.age}")

def sort_by_fitness(pop):
    pop.sort(key=lambda c: c.fitness)

###############################################################################
# example run
###############################################################################

def run_ga():
    start_time = time.time()
    pop, buf = init_population()
    calc_fitness(pop)
    pop.sort(key=lambda x: x.fitness)
    generation = 0

    while generation < ga_maxiter:
        if (time.time() - start_time) > ga_max_runtime:
            print("time limit reached.")
            break

        mate(pop, buf)
        calc_fitness(buf)
        buf.sort(key=lambda x: x.fitness)
        pop, buf = swap(pop, buf)
        generation += 1
        print(f"gen={generation}, best fitness={pop[0].fitness}")

    print_best(pop)

###############################################################################
# if needed, you can call run_ga() or integrate in your code
###############################################################################
if __name__ == "__main__":
    # example usage
    # set problem type to binpacking if you want
    # ga_problem_type = "binpacking"
    run_ga()
