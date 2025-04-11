import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

############################################################################
# DEFAULT VALUES
############################################################################
DEFAULT_PROBLEM_TYPE = "string"         # "string", "binpacking", or "arc"
DEFAULT_POP_SIZE = 50
DEFAULT_MAX_ITER = 250
DEFAULT_ELITE_RATE = 0.10
DEFAULT_MUTATION_RATE = 0.55
DEFAULT_SELECTION_METHOD = "tournament_probabilistic"
DEFAULT_USE_LINEAR_SCALING = True
DEFAULT_MAX_FITNESS_RATIO = 2.0
DEFAULT_TOURNAMENT_K = 3
DEFAULT_TOURNAMENT_K_PROB = 3
DEFAULT_TOURNAMENT_P = 0.75
DEFAULT_USE_AGING = True
DEFAULT_AGE_LIMIT = 100

# string
DEFAULT_TARGET_STRING = "testing string123 diff_chars"
DEFAULT_STRING_FITNESS_MODE = "ascii"   # "ascii","lcs","combined"
DEFAULT_LCS_BONUS = 5
DEFAULT_DISTANCE_METRIC = "levenshtein"

# large penalty magnitude if we still want some penalty, but not overshadow everything
# can tweak if you want more or less punishment for overflow
DEFAULT_OVERFLOW_PENALTY_FACTOR = 5.0

# binpack data
DEFAULT_BINPACK_DATA_FILE = "binpack1.txt"

# binpacking heuristic (for how we interpret the candidate's permutation to compute # of bins)
DEFAULT_BINPACK_HEURISTIC = "first_fit"   # or "best_fit", "first_fit_decreasing", "best_fit_decreasing"

# crossovers / mutation
DEFAULT_CROSSOVER_METHOD = "two_point"
DEFAULT_MUTATION_OPERATOR = "exchange"

############################################################################
# CONSTANTS
############################################################################
HUGE_PENALTY = 1e9  # fallback if something is truly invalid

############################################################################
# GAState
############################################################################
class GAState:
    """
    GA configuration class
    """
    def __init__(self):
        self.problem_type = DEFAULT_PROBLEM_TYPE
        self.pop_size = DEFAULT_POP_SIZE
        self.max_iter = DEFAULT_MAX_ITER
        self.elite_rate = DEFAULT_ELITE_RATE
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.selection_method = DEFAULT_SELECTION_METHOD
        self.use_linear_scaling = DEFAULT_USE_LINEAR_SCALING
        self.max_fitness_ratio = DEFAULT_MAX_FITNESS_RATIO
        self.tournament_k = DEFAULT_TOURNAMENT_K
        self.tournament_k_prob = DEFAULT_TOURNAMENT_K_PROB
        self.tournament_p = DEFAULT_TOURNAMENT_P
        self.use_aging = DEFAULT_USE_AGING
        self.age_limit = DEFAULT_AGE_LIMIT

        # string
        self.target_string = DEFAULT_TARGET_STRING
        self.string_fitness_mode = DEFAULT_STRING_FITNESS_MODE
        self.lcs_bonus = DEFAULT_LCS_BONUS
        self.distance_metric = DEFAULT_DISTANCE_METRIC

        # binpacking
        self.binpack_data_file = DEFAULT_BINPACK_DATA_FILE
        self.binpack_problem_cache = None
        self.binpack_current_problem = None
        self.binpack_heuristic = DEFAULT_BINPACK_HEURISTIC
        self.overflow_penalty_factor = DEFAULT_OVERFLOW_PENALTY_FACTOR

        # crossovers
        self.crossover_method = DEFAULT_CROSSOVER_METHOD
        self.mutation_operator = DEFAULT_MUTATION_OPERATOR


class Candidate:
    """represents one solution in the population."""
    def __init__(self, gene, fitness=0):
        self.gene = gene
        self.fitness = fitness
        self.age = 0

############################################################################
# BINPACK FILE READING
############################################################################
def read_binpack_problems(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    idx = 0
    P = int(lines[idx].strip())
    idx+=1
    problems=[]
    for _ in range(P):
        prob_id = lines[idx].strip()
        idx+=1
        parts = lines[idx].split()
        idx+=1
        capacity = int(parts[0])
        n_items = int(parts[1])
        best_bins = int(parts[2])
        items=[]
        for __ in range(n_items):
            val = int(lines[idx])
            idx+=1
            items.append(val)
        problems.append((prob_id, capacity, n_items, best_bins, items))
    return problems

def maybe_load_binpack_data(gastate):
    if gastate.binpack_problem_cache is None:
        gastate.binpack_problem_cache = read_binpack_problems(gastate.binpack_data_file)

def pick_random_binpack_problem(gastate):
    maybe_load_binpack_data(gastate)
    gastate.binpack_current_problem = random.choice(gastate.binpack_problem_cache)

############################################################################
# INIT POPULATION
############################################################################
def init_population_string(gastate):
    """randomly create strings for each candidate"""
    tlen = len(gastate.target_string)
    pop = []
    for _ in range(gastate.pop_size):
        gene = "".join(chr(random.randint(32,121)) for __ in range(tlen))
        pop.append(Candidate(gene))
    buf = [Candidate(None) for __ in range(gastate.pop_size)]
    return pop, buf

def init_population_binpacking(gastate):
    """
    create random permutations for each candidate, ensuring we get some diversity
    in the first generation => avoids zero stdev in generation 1.
    """
    if gastate.binpack_current_problem is None:
        pick_random_binpack_problem(gastate)
    _, _, n_items, _, _ = gastate.binpack_current_problem

    pop=[]
    for _ in range(gastate.pop_size):
        perm = list(range(n_items))
        random.shuffle(perm)
        pop.append(Candidate(perm))
    buf = [Candidate([0]*n_items) for __ in range(gastate.pop_size)]
    return pop, buf

def init_population_arc(gastate):
    raise NotImplementedError("ARC is not implemented")

def init_population(gastate):
    if gastate.problem_type=="string":
        return init_population_string(gastate)
    elif gastate.problem_type=="binpacking":
        return init_population_binpacking(gastate)
    else:
        # fallback => string
        return init_population_string(gastate)

############################################################################
# FITNESS (STRING)
############################################################################
def lcs_length(a,b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def calc_fitness_string(pop, gastate):
    t = gastate.target_string
    tlen = len(t)
    for c in pop:
        if gastate.string_fitness_mode=="ascii":
            c.fitness = sum(abs(ord(c.gene[i]) - ord(t[i])) for i in range(tlen))
        elif gastate.string_fitness_mode=="lcs":
            val = lcs_length(c.gene, t)
            c.fitness = tlen - val
        elif gastate.string_fitness_mode=="combined":
            ascii_sum = sum(abs(ord(c.gene[i]) - ord(t[i])) for i in range(tlen))
            val = lcs_length(c.gene, t)
            c.fitness = ascii_sum + gastate.lcs_bonus*(tlen - val)
        else:
            # fallback => ascii
            c.fitness = sum(abs(ord(c.gene[i]) - ord(t[i])) for i in range(tlen))

############################################################################
# FITNESS (BINPACK)
############################################################################
def place_items_heuristic(perm, all_items, capacity, method):
    """
    Place items in the order of 'perm' (or sorted if method says 'decreasing'),
    using either 'first_fit' or 'best_fit'. Return (bins_usage).
    If we fail, raise an exception or something. But we won't, because these
    heuristics won't fail, they just create new bins as needed.
    """
    # check if we do a "decreasing" sort of perm by item size
    if "decreasing" in method:
        perm = sorted(perm, key=lambda idx: all_items[idx], reverse=True)

    bins_usage = []
    if "best_fit" in method:
        # best fit or best_fit_decreasing
        for idx in perm:
            size_ = all_items[idx]
            chosen_bin = None
            chosen_left = None
            for b_idx in range(len(bins_usage)):
                leftover = capacity - (bins_usage[b_idx])
                if size_ <= leftover:
                    # feasible
                    # see if it leaves less leftover
                    new_left = leftover - size_
                    if chosen_bin is None or new_left < chosen_left:
                        chosen_bin = b_idx
                        chosen_left = new_left
            if chosen_bin is not None:
                bins_usage[chosen_bin] += size_
            else:
                bins_usage.append(size_)
    else:
        # fallback => "first_fit" or "first_fit_decreasing"
        for idx in perm:
            size_ = all_items[idx]
            placed = False
            for b_idx in range(len(bins_usage)):
                if bins_usage[b_idx] + size_ <= capacity:
                    bins_usage[b_idx]+= size_
                    placed = True
                    break
            if not placed:
                bins_usage.append(size_)
    return bins_usage


def calc_fitness_binpack(pop, gastate):
    """
    For each candidate => interpret gene as a permutation of item indices.
    We apply the chosen heuristic (first_fit, best_fit, first_fit_decreasing, best_fit_decreasing).
    Then fitness = #bins + 0.001*(sum leftover).
    If there's overflow, we do a partial penalty = overflow * overflow_penalty_factor
    That punishes invalid solutions, but won't overshadow everything completely.
    """
    if gastate.binpack_current_problem is None:
        pick_random_binpack_problem(gastate)
    _, capacity, n_items, _, all_items = gastate.binpack_current_problem
    method = gastate.binpack_heuristic.lower()

    for c in pop:
        perm = c.gene[:]  # copy
        # place them
        bins_usage = place_items_heuristic(perm, all_items, capacity, method)

        # check if any bin is over capacity => define partial penalty
        overflow = 0
        for usage in bins_usage:
            if usage > capacity:
                overflow += (usage - capacity)
        if overflow > 0:
            # partial penalty
            c.fitness = len(bins_usage) + 0.01* gastate.overflow_penalty_factor* overflow
        else:
            # valid => # bins + leftover
            leftover = 0
            for usage in bins_usage:
                leftover += (capacity - usage)
            c.fitness = len(bins_usage) + 0.1* leftover


############################################################################
# MASTER FITNESS
############################################################################
def calc_fitness(pop, gastate):
    if gastate.problem_type=="string":
        calc_fitness_string(pop, gastate)
    elif gastate.problem_type=="binpacking":
        calc_fitness_binpack(pop, gastate)
    else:
        calc_fitness_string(pop, gastate)

############################################################################
# LINEAR SCALING / SELECTION
############################################################################
def linear_scaling(pop, gastate):
    fvals = [c.fitness for c in pop]
    fmin = min(fvals)
    fmax = max(fvals)
    if abs(fmax - fmin) < 1e-9:
        for c in pop:
            c.scaled = 1.0
        return
    # invert => bigger => better
    scores = [fmax - f for f in fvals]
    smin, smax = min(scores), max(scores)
    base = [s - smin for s in scores]
    bmax = smax - smin
    if abs(bmax)<1e-9:
        for c in pop:
            c.scaled= 1.0
        return
    avg_ = sum(base)/len(base)
    ratio= bmax/avg_ if avg_>1e-9 else 1.0
    a=1.0
    if ratio> gastate.max_fitness_ratio:
        a= gastate.max_fitness_ratio/ ratio
    for i,c in enumerate(pop):
        c.scaled= a* base[i]
    tot= sum(c.scaled for c in pop)
    if tot<1e-9:
        for c in pop:
            c.scaled=1.0

def rws_select_one(pop):
    s= sum(c.scaled for c in pop)
    if s<1e-9:
        return random.choice(pop)
    pick= random.random()* s
    run=0
    for c in pop:
        run+= c.scaled
        if run>= pick:
            return c
    return pop[-1]

def sus_select_parents(pop, n):
    s= sum(c.scaled for c in pop)
    if s<1e-9:
        return [random.choice(pop) for __ in range(n)]
    dist= s/n
    start= random.random()* dist
    chosen=[]
    run=0
    idx=0
    for i in range(n):
        ptr= start+ i* dist
        while run< ptr and idx< len(pop)-1:
            run+= pop[idx].scaled
            idx+=1
        chosen.append(pop[idx-1])
    return chosen

def sus_select_one(pop):
    return sus_select_parents(pop,1)[0]

def tournament_det(pop, k=3):
    cands= random.sample(pop, k)
    cands.sort(key=lambda x:x.fitness)
    return cands[0]

def tournament_prob(pop, k=3, p=0.75):
    cands= random.sample(pop, k)
    cands.sort(key=lambda x:x.fitness)
    r= random.random()
    cum=0
    for i,c in enumerate(cands):
        prob_i= p*((1-p)**i)
        cum+= prob_i
        if r<= cum:
            return c
    return cands[-1]

def old_rw_select(pop):
    inv= [1/(1+ c.fitness) for c in pop]
    tot= sum(inv)
    pick= random.random()* tot
    run=0
    for i,v in enumerate(inv):
        run+=v
        if pick<= run:
            return pop[i]
    return pop[-1]

def select_one(pop, gastate):
    m= gastate.selection_method.lower()
    if m=="rws":
        return rws_select_one(pop)
    elif m=="sus":
        return sus_select_one(pop)
    elif m=="tournament_deterministic":
        return tournament_det(pop, gastate.tournament_k)
    elif m=="tournament_probabilistic":
        return tournament_prob(pop, gastate.tournament_k_prob, gastate.tournament_p)
    else:
        return old_rw_select(pop)

############################################################################
# CROSSOVER
############################################################################
def single_point_cx_str(p1,p2):
    ln= len(p1.gene)
    cut= random.randint(0, ln-1)
    c1= p1.gene[:cut] + p2.gene[cut:]
    c2= p2.gene[:cut] + p1.gene[cut:]
    return c1,c2

def two_point_cx_str(p1,p2):
    ln= len(p1.gene)
    i,j= sorted(random.sample(range(ln),2))
    c1= p1.gene[:i]+ p2.gene[i:j]+ p1.gene[j:]
    c2= p2.gene[:i]+ p1.gene[i:j]+ p2.gene[j:]
    return c1,c2

def uniform_cx_str(p1,p2):
    ln= len(p1.gene)
    arr1=[]
    arr2=[]
    for i in range(ln):
        if random.random()<0.5:
            arr1.append(p1.gene[i])
            arr2.append(p2.gene[i])
        else:
            arr1.append(p2.gene[i])
            arr2.append(p1.gene[i])
    return "".join(arr1), "".join(arr2)

def pmx_crossover(g1,g2):
    ln= len(g1)
    c1=[None]*ln
    c2=[None]*ln
    s,e= sorted(random.sample(range(ln),2))
    for i in range(s,e):
        c1[i]= g1[i]
        c2[i]= g2[i]
    def fill_pmx(child, donor, ss, ee):
        for i in range(ss, ee):
            val= donor[i]
            if val not in child:
                pos= i
                while child[pos] is not None:
                    pos= donor.index(g1[pos])
                child[pos]= val
    fill_pmx(c1, g2, s,e)
    fill_pmx(c2, g1, s,e)
    for i in range(ln):
        if c1[i] is None:
            c1[i]= g2[i]
        if c2[i] is None:
            c2[i]= g1[i]
    return c1,c2

def ox_crossover(g1,g2):
    ln= len(g1)
    c1=[None]*ln
    c2=[None]*ln
    s,e= sorted(random.sample(range(ln),2))
    c1[s:e]= g1[s:e]
    c2[s:e]= g2[s:e]
    def fill_ox(child, parent, ss, ee):
        pos= e
        if pos>=ln: pos=0
        for x in parent:
            if x not in child:
                child[pos]= x
                pos+=1
                if pos>=ln:
                    pos=0
    fill_ox(c1, g2, s,e)
    fill_ox(c2, g1, s,e)
    return c1,c2

def cx_crossover(g1,g2):
    ln= len(g1)
    c1=[None]*ln
    c2=[None]*ln
    used=set()
    idx=0
    while True:
        if idx in used:
            free= [x for x in range(ln) if x not in used]
            if not free:
                break
            idx= free[0]
        start_val= g1[idx]
        while True:
            c1[idx]= g1[idx]
            c2[idx]= g2[idx]
            used.add(idx)
            idx= g1.index(g2[idx])
            if g1[idx]== start_val:
                c1[idx]= g1[idx]
                c2[idx]= g2[idx]
                used.add(idx)
                break
    for i in range(ln):
        if c1[i] is None:
            c1[i]= g2[i]
        if c2[i] is None:
            c2[i]= g1[i]
    return c1,c2

def crossover_operator(p1,p2,gastate):
    if gastate.problem_type=="string":
        method= gastate.crossover_method.lower()
        if method=="single":
            return single_point_cx_str(p1,p2)
        elif method=="two_point":
            return two_point_cx_str(p1,p2)
        elif method=="uniform":
            return uniform_cx_str(p1,p2)
        else:
            return single_point_cx_str(p1,p2)
    elif gastate.problem_type=="binpacking":
        g1,g2= p1.gene, p2.gene
        m= gastate.crossover_method.lower()
        if m=="pmx":
            return pmx_crossover(g1,g2)
        elif m=="ox":
            return ox_crossover(g1,g2)
        elif m=="cx":
            return cx_crossover(g1,g2)
        elif m=="two_point" or m=="single" or m=="uniform":
            return pmx_crossover(g1,g2)
        else:
            return pmx_crossover(g1,g2)
    else:
        # fallback
        if isinstance(p1.gene,str):
            return single_point_cx_str(p1,p2)
        else:
            return pmx_crossover(p1.gene, p2.gene)

############################################################################
# MUTATION
############################################################################
def mutate_string(c, gastate):
    ln= len(c.gene)
    pos= random.randint(0, ln-1)
    delta= random.randint(32,121)
    old_val= ord(c.gene[pos])
    new_val= 32+ ((old_val-32+delta)%(121-32+1))
    arr= list(c.gene)
    arr[pos]= chr(new_val)
    c.gene= "".join(arr)

def exchange_mutation(c):
    g= c.gene
    if len(g)<2: return
    i,j= random.sample(range(len(g)),2)
    g[i], g[j] = g[j], g[i]

def displacement_mutation(c):
    g= c.gene
    i,j= random.sample(range(len(g)),2)
    val= g.pop(i)
    j= min(j,len(g))
    g.insert(j,val)

def insertion_mutation(c):
    g= c.gene
    i,j= sorted(random.sample(range(len(g)),2))
    val= g.pop(j)
    g.insert(i,val)

def simple_inversion_mutation(c):
    g= c.gene
    i,j= sorted(random.sample(range(len(g)),2))
    g[i:j]= reversed(g[i:j])

def scramble_mutation(c):
    g= c.gene
    i,j= sorted(random.sample(range(len(g)),2))
    chunk= g[i:j]
    random.shuffle(chunk)
    g[i:j]= chunk

def mutate_candidate(c, gastate):
    if gastate.problem_type=="string":
        mutate_string(c, gastate)
    elif gastate.problem_type=="binpacking":
        m= gastate.mutation_operator.lower()
        if m=="displacement":
            displacement_mutation(c)
        elif m=="insertion":
            insertion_mutation(c)
        elif m=="simple_inversion":
            simple_inversion_mutation(c)
        elif m=="scramble":
            scramble_mutation(c)
        elif m=="exchange":
            exchange_mutation(c)
        else:
            # fallback
            exchange_mutation(c)
    else:
        if isinstance(c.gene,str):
            mutate_string(c, gastate)
        else:
            exchange_mutation(c)


############################################################################
# ELITISM + AGING
############################################################################
def elitism(pop, buf, esize):
    for i in range(esize):
        if isinstance(pop[i].gene, list):
            buf[i].gene= pop[i].gene[:]
        else:
            buf[i].gene= pop[i].gene
        buf[i].fitness= pop[i].fitness
        buf[i].age= pop[i].age+1

def apply_aging(buf, gastate):
    if not gastate.use_aging:
        return
    for i,c in enumerate(buf):
        if c.age> gastate.age_limit:
            p1= select_one(buf, gastate)
            p2= select_one(buf, gastate)
            c1, _= crossover_operator(p1,p2,gastate)
            if isinstance(c1,list):
                buf[i]= Candidate(c1[:],0)
            else:
                buf[i]= Candidate(c1,0)

############################################################################
# MATE
############################################################################
def mate(pop, buf, gastate):
    esize= int(gastate.elite_rate * gastate.pop_size)
    elitism(pop, buf, esize)

    if gastate.selection_method in ["rws","sus"]:
        if gastate.use_linear_scaling:
            linear_scaling(pop, gastate)
        else:
            for c in pop:
                c.scaled= 1/(1+ c.fitness)

    i= esize
    while i< gastate.pop_size -1:
        p1= select_one(pop, gastate)
        p2= select_one(pop, gastate)
        c1, c2= crossover_operator(p1,p2,gastate)

        if isinstance(c1,list):
            buf[i].gene= c1[:]
        else:
            buf[i].gene= c1
        buf[i].fitness= 0
        buf[i].age= 0

        if isinstance(c2,list):
            buf[i+1].gene= c2[:]
        else:
            buf[i+1].gene= c2
        buf[i+1].fitness= 0
        buf[i+1].age= 0

        # mutate
        if random.random()< gastate.mutation_rate:
            mutate_candidate(buf[i], gastate)
        if random.random()< gastate.mutation_rate:
            mutate_candidate(buf[i+1], gastate)

        i+=2

    if gastate.pop_size%2==1 and i< gastate.pop_size:
        buf[i].gene= buf[i-1].gene if isinstance(buf[i-1].gene,list) else buf[i-1].gene
        buf[i].fitness= buf[i-1].fitness
        buf[i].age= buf[i-1].age
        if random.random()< gastate.mutation_rate:
            mutate_candidate(buf[i], gastate)

    apply_aging(buf, gastate)

def swap_pop(buf, pop):
    return buf, pop

############################################################################
# STATISTICS
############################################################################
def compute_fitness_statistics(pop, gastate):
    pop.sort(key=lambda x:x.fitness)
    best_= pop[0].fitness
    worst_= pop[-1].fitness
    fvals= [c.fitness for c in pop]
    mean_= sum(fvals)/ len(fvals)
    if len(fvals)>1:
        var_= sum((f- mean_)**2 for f in fvals)/(len(fvals)-1)
    else:
        var_=0
    std_= math.sqrt(var_)
    rng_= worst_- best_

    inv= [1/(1+ f) for f in fvals]
    sum_inv= sum(inv)
    if sum_inv<1e-9:
        selection_var= 0.0
        top_avg_prob_ratio= 1.0
    else:
        sProbs= [v/sum_inv for v in inv]
        sProbs.sort(reverse=True)
        selection_var= np.var(sProbs)
        if selection_var<1e-12:
            selection_var= 0.0
        top_size= max(1, int(gastate.elite_rate* len(pop)))
        top_probs= sProbs[:top_size]
        p_top= sum(top_probs)/ top_size
        p_avg= 1/ len(pop)
        top_avg_prob_ratio= p_top/p_avg if p_avg>1e-9 else 1.0

    stats= {
        "best": best_,
        "worst": worst_,
        "mean": mean_,
        "std": std_,
        "range": rng_,
        "selection_variance": selection_var,
        "top_avg_prob_ratio": top_avg_prob_ratio
    }
    return stats

############################################################################
# STRING DIVERSITY
############################################################################
def levenshtein_distance(a,b):
    la,lb= len(a), len(b)
    d= [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):
        d[i][0]= i
    for j in range(lb+1):
        d[0][j]= j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost=0 if a[i-1]==b[j-1] else 1
            d[i][j]= min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+ cost)
    return d[la][lb]

def compute_string_diversity(pop):
    """
    returns (avg_dist, avg_diff, avg_shannon)
    """
    n= len(pop)
    if n<2:
        return (0.0, 0.0, 0.0)
    # 1) avg dist
    sample_size= min(50, n)
    sample= random.sample(pop, sample_size)
    dist_sum=0
    count=0
    for i in range(sample_size):
        for j in range(i+1,sample_size):
            dist_sum+= levenshtein_distance(sample[i].gene, sample[j].gene)
            count+=1
    avg_dist= dist_sum/count if count>0 else 0

    # 2) avg diff
    ln= len(pop[0].gene)
    total_diff=0
    pairs=0
    for i in range(n):
        for j in range(i+1,n):
            dif = sum(pop[i].gene[k]!= pop[j].gene[k] for k in range(ln))
            total_diff+= dif
            pairs+=1
    avg_diff= total_diff/pairs if pairs>0 else 0

    # 3) shannon
    total_ent=0
    for pos in range(ln):
        freq={}
        for c in pop:
            ch= c.gene[pos]
            freq[ch]= freq.get(ch,0)+1
        ent=0
        for k in freq:
            p= freq[k]/ n
            ent-= p* math.log2(p)
        total_ent+= ent
    avg_shannon= total_ent/ ln

    return (avg_dist, avg_diff, avg_shannon)

############################################################################
# PLOTTING
############################################################################
def plot_fitness_evolution(best_hist, mean_hist, worst_hist):
    gens = list(range(len(best_hist)))
    plt.figure(figsize=(10,5))
    plt.plot(gens, best_hist, label="best", linewidth=2)
    plt.plot(gens, mean_hist, label="mean", linewidth=2)
    plt.plot(gens, worst_hist, label="worst", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitness_boxplots(fitness_history):
    plt.figure(figsize=(10,5))
    plt.boxplot(fitness_history, patch_artist=True)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("Fitness Distribution per Generation")
    plt.grid(True)
    plt.show()

def plot_diversity(ent_hist, dist_hist, diff_hist):
    gens= list(range(len(ent_hist)))
    plt.figure(figsize=(10,5))
    plt.plot(gens, ent_hist, label="shannon entropy", linewidth=2)
    plt.plot(gens, dist_hist, label="avg distance", linewidth=2)
    plt.plot(gens, diff_hist, label="avg diff alleles", linewidth=2)
    plt.xlabel("generation")
    plt.ylabel("diversity metrics")
    plt.title("Population Diversity Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

############################################################################
# PRINTING
############################################################################
def print_gen_info(gen, stats, dist_, diff_, ent_, cpu_time, wall_time, raw_ticks, best_candidate):
    print(f"generation {gen}: mean fitness = {stats['mean']:.2f}, std = {stats['std']:.2f}, worst = {stats['worst']}, range = {stats['range']}")
    print(f"   selection_var = {stats['selection_variance']:.4f}, top_avg_prob_ratio = {stats['top_avg_prob_ratio']:.2f}, avg distance = {dist_:.2f}, avg diff_alleles = {diff_:.2f}, avg shannon entropy = {ent_:.2f}")
    print(f"   CPU time = {cpu_time:.4f}s, elapsed = {wall_time:.2f}s, raw ticks = {raw_ticks}, tick_time = ???")
    print(f"best: {best_candidate.gene} ({best_candidate.fitness}) age={best_candidate.age}")

############################################################################
# GA RUN
############################################################################
def run_ga(
        problem_type=DEFAULT_PROBLEM_TYPE,
        pop_size=DEFAULT_POP_SIZE,
        max_iter=DEFAULT_MAX_ITER,
        elite_rate=DEFAULT_ELITE_RATE,
        mutation_rate=DEFAULT_MUTATION_RATE,
        selection_method=DEFAULT_SELECTION_METHOD,
        use_linear_scaling=DEFAULT_USE_LINEAR_SCALING,
        max_fitness_ratio=DEFAULT_MAX_FITNESS_RATIO,
        tournament_k=DEFAULT_TOURNAMENT_K,
        tournament_k_prob=DEFAULT_TOURNAMENT_K_PROB,
        tournament_p=DEFAULT_TOURNAMENT_P,
        use_aging=DEFAULT_USE_AGING,
        age_limit=DEFAULT_AGE_LIMIT,

        # string
        target_string=DEFAULT_TARGET_STRING,
        string_fitness_mode=DEFAULT_STRING_FITNESS_MODE,
        lcs_bonus=DEFAULT_LCS_BONUS,
        distance_metric=DEFAULT_DISTANCE_METRIC,

        # binpacking
        binpack_data_file=DEFAULT_BINPACK_DATA_FILE,
        binpack_heuristic=DEFAULT_BINPACK_HEURISTIC,
        overflow_penalty_factor=DEFAULT_OVERFLOW_PENALTY_FACTOR,

        # crossovers
        crossover_method=DEFAULT_CROSSOVER_METHOD,
        mutation_operator=DEFAULT_MUTATION_OPERATOR
):
    """
    GA engine with updated bin-packing initialization for real diversity
    and a partial penalty for overflow. This ensures invalid solutions
    are punished but not so extremely that the entire search is overshadowed.

    For binpacking, the user can pick:
      - first_fit
      - best_fit
      - first_fit_decreasing
      - best_fit_decreasing
    If unknown => default to 'first_fit'.
    """
    # fill gastate
    gastate= GAState()
    gastate.problem_type= problem_type
    gastate.pop_size= pop_size
    gastate.max_iter= max_iter
    gastate.elite_rate= elite_rate
    gastate.mutation_rate= mutation_rate
    gastate.selection_method= selection_method
    gastate.use_linear_scaling= use_linear_scaling
    gastate.max_fitness_ratio= max_fitness_ratio
    gastate.tournament_k= tournament_k
    gastate.tournament_k_prob= tournament_k_prob
    gastate.tournament_p= tournament_p
    gastate.use_aging= use_aging
    gastate.age_limit= age_limit
    gastate.target_string= target_string
    gastate.string_fitness_mode= string_fitness_mode
    gastate.lcs_bonus= lcs_bonus
    gastate.distance_metric= distance_metric
    gastate.binpack_data_file= binpack_data_file
    gastate.binpack_heuristic= binpack_heuristic.lower()
    gastate.overflow_penalty_factor= overflow_penalty_factor
    gastate.crossover_method= crossover_method
    gastate.mutation_operator= mutation_operator

    # init
    pop, buf = init_population(gastate)
    calc_fitness(pop, gastate)
    pop.sort(key=lambda x:x.fitness)
    # track stats
    best_hist=[]
    mean_hist=[]
    worst_hist=[]
    box_history=[]
    ent_hist=[]
    dist_hist=[]
    diff_hist=[]

    overall_start = time.time()
    gen_cpu = time.process_time()

    best_bin_cand= None
    best_bin_fit= float("inf")

    for g in range(gastate.max_iter):
        stats= compute_fitness_statistics(pop, gastate)
        best_hist.append(stats["best"])
        mean_hist.append(stats["mean"])
        worst_hist.append(stats["worst"])
        box_history.append([c.fitness for c in pop])

        # measure diversity if string
        if gastate.problem_type=="string":
            d, df, ent= compute_string_diversity(pop)
            dist_hist.append(d)
            diff_hist.append(df)
            ent_hist.append(ent)
            if stats["best"]<= 0:
                # converge
                now_cpu= time.process_time()
                ctime= now_cpu- gen_cpu
                wtime= time.time()- overall_start
                rawt= time.perf_counter_ns()
                print_gen_info(g, stats, d, df, ent, ctime, wtime, rawt, pop[0])
                print("string: found fitness=0 => converging early.")
                break
        else:
            # binpacking => track best
            dist_hist.append(0.0)
            diff_hist.append(0.0)
            ent_hist.append(0.0)
            if gastate.problem_type=="binpacking":
                if pop[0].fitness< best_bin_fit:
                    best_bin_fit= pop[0].fitness
                    if isinstance(pop[0].gene, list):
                        best_bin_cand= Candidate(pop[0].gene[:], pop[0].fitness)
                    else:
                        best_bin_cand= Candidate(pop[0].gene, pop[0].fitness)
                    best_bin_cand.age= pop[0].age

        now_cpu= time.process_time()
        ctime= now_cpu- gen_cpu
        wtime= time.time()- overall_start
        rawt= time.perf_counter_ns()

        print_gen_info(g, stats, dist_hist[-1], diff_hist[-1], ent_hist[-1], ctime, wtime, rawt, pop[0])

        # reset gen_cpu
        gen_cpu= time.process_time()

        # mate
        mate(pop, buf, gastate)
        calc_fitness(buf, gastate)
        buf.sort(key=lambda x:x.fitness)
        pop, buf= swap_pop(buf, pop)

    # final
    if gastate.problem_type=="binpacking" and best_bin_cand is not None:
        print(f"\nBinpacking best solution => fitness={best_bin_cand.fitness:.2f}")
        print("Permutation =>", best_bin_cand.gene)

    # plot
    plot_fitness_evolution(best_hist, mean_hist, worst_hist)
    plot_fitness_boxplots(box_history)
    if gastate.problem_type=="string":
        plot_diversity(ent_hist, dist_hist, diff_hist)

############################################################################
# EXAMPLE
############################################################################
if __name__=="__main__":
    # run string
    run_ga(
        problem_type="string",
        target_string="testing string123 diff_chars",
        pop_size=500,
        max_iter=300,
        binpack_heuristic="best_fit_decreasing"  # won't matter for string

    )

    # run binpacking
    # run_ga(
    #     problem_type="binpacking",
    #     pop_size=40,
    #     max_iter=150,
    #     binpack_heuristic="best_fit_decreasing",
    #     overflow_penalty_factor=50.0
    # )
