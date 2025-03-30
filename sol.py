import random
import time
import math
import matplotlib.pyplot as plt  #import matplotlib for plotting

# constant parameters
GA_POPSIZE = 8192  #population size
GA_MAXITER = 16384  #maximum iterations
GA_ELITRATE = 0.10  #elitism rate (10%)
GA_MUTATIONRATE = 0.25  #mutation probability
GA_TARGET = "testing something longer than hello_world"  #target string

# candidate class representing an individual
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene  #candidate string
        self.fitness = fitness  #fitness score

# initialize population and buffer
def init_population():
    target_length = len(GA_TARGET)
    population = []
    #create initial population with random genes
    for _ in range(GA_POPSIZE):
        #random char between ascii 32 and 121 for each gene position
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    #create buffer list with empty candidates
    buffer = [Candidate('') for _ in range(GA_POPSIZE)]
    return population, buffer

# calculate fitness for each candidate
def calc_fitness(population):
    target = GA_TARGET
    target_length = len(target)
    for candidate in population:
        fitness = 0
        #sum of absolute differences for each character
        for i in range(target_length):
            fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
        candidate.fitness = fitness

# sort population by fitness (lower is better)
def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)

# copy best candidates to next generation (elitism)
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        #copy candidate gene and fitness to buffer
        buffer[i] = Candidate(population[i].gene, population[i].fitness)

# mutate a candidate by altering one random gene position
def mutate(candidate):
    target_length = len(GA_TARGET)
    pos = random.randint(0, target_length - 1)  #random index in gene
    #delta is a random value between 32 and 121
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    #update gene char by adding delta modulo 122
    new_val = (old_val + delta) % 122
    #convert string to list for mutation since strings are immutable
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)

# mate candidates to produce new generation
def mate(population, buffer):
    elite_size = int(GA_POPSIZE * GA_ELITRATE)
    target_length = len(GA_TARGET)
    #copy best candidates to buffer
    elitism(population, buffer, elite_size)
    #create rest of new generation by crossover and mutation
    for i in range(elite_size, GA_POPSIZE):
        #choose two parents from top half of population
        parent1 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        parent2 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        #random crossover point
        crossover_point = random.randint(0, target_length - 1)
        #single-point crossover: combine parent genes at crossover point
        new_gene = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
        buffer[i] = Candidate(new_gene)
        #apply mutation based on mutation rate
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])

# print best candidate from population
def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")

# swap population and buffer for next generation
def swap(population, buffer):
    return buffer, population

#new: compute fitness stats (mean, std, worst, range)
def compute_fitness_statistics(population):
    #get all fitness values from population
    fitness_values = [cand.fitness for cand in population]
    #calculate mean fitness
    mean_fitness = sum(fitness_values) / len(fitness_values)
    #calculate variance and standard deviation
    variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / len(fitness_values)
    std_fitness = math.sqrt(variance)
    #get best and worst fitness (population is sorted by fitness)
    best_fitness = population[0].fitness
    worst_fitness = population[-1].fitness
    #calculate fitness range (worst - best)
    fitness_range = worst_fitness - best_fitness
    #return stats in a dict
    return {
        "mean": mean_fitness,
        "std": std_fitness,
        "worst_fitness": worst_fitness,
        "fitness_range": fitness_range,
        "worst_candidate": population[-1]
    }

#new: compute timing metrics (cpu time for generation and elapsed time overall)
def compute_timing_metrics(generation_start_cpu, overall_start_wall):
    #get current cpu and wall time
    current_cpu = time.process_time()
    current_wall = time.time()
    #calculate cpu time for this generation and overall elapsed time
    generation_cpu_time = current_cpu - generation_start_cpu
    elapsed_time = current_wall - overall_start_wall
    #return timing info in a dict
    return {
        "generation_cpu_time": generation_cpu_time,
        "elapsed_time": elapsed_time
    }

#new: plot fitness evolution over generations (line plot)
def plot_fitness_evolution(best_history, mean_history, worst_history):
    generations = list(range(len(best_history)))
    plt.figure()  #create new figure
    #plot best fitness
    plt.plot(generations, best_history, label="best")
    #plot mean fitness
    plt.plot(generations, mean_history, label="mean")
    #plot worst fitness
    plt.plot(generations, worst_history, label="worst")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("fitness evolution over generations")
    plt.legend()
    plt.grid(True)
    plt.show()  #display plot

#new: plot boxplots of fitness per generation
def plot_fitness_boxplots(fitness_distributions):
    plt.figure()  #create new figure
    #set flier (outlier) properties to diamond shape
    flierprops = dict(marker='D', markersize=4, linestyle='none', markeredgecolor='black')
    #create boxplot; showmeans is optional but not required
    plt.boxplot(fitness_distributions, flierprops=flierprops, patch_artist=True)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("fitness distribution per generation")
    plt.grid(True)
    plt.show()  #display plot

# main genetic algorithm loop
def main():
    random.seed(time.time())  #seed random number generator
    population, buffer = init_population()

    overall_start_wall = time.time()  #start overall wall time

    #lists to store fitness history and distributions
    best_history = []
    mean_history = []
    worst_history = []
    fitness_distributions = []

    for iteration in range(GA_MAXITER):
        generation_start_cpu = time.process_time()  #start cpu time for this generation

        calc_fitness(population)  #calculate fitness for each candidate
        sort_by_fitness(population)  #sort candidates by fitness
        print_best(population)  #print best candidate

        #compute and print fitness stats
        stats = compute_fitness_statistics(population)
        print(f"generation {iteration}: mean fitness = {stats['mean']:.2f}, std = {stats['std']:.2f}, "
              f"worst fitness = {stats['worst_fitness']}, range = {stats['fitness_range']}, "
              f"worst candidate = {stats['worst_candidate'].gene}")

        #compute and print timing metrics
        timing = compute_timing_metrics(generation_start_cpu, overall_start_wall)
        print(f"generation {iteration}: cpu time = {timing['generation_cpu_time']:.4f} s, "
              f"elapsed time = {timing['elapsed_time']:.4f} s")

        #store fitness history for plotting later
        best_history.append(population[0].fitness)
        mean_history.append(stats['mean'])
        worst_history.append(stats['worst_fitness'])
        #store full fitness distribution for boxplot
        fitness_distributions.append([cand.fitness for cand in population])

        #analysis: fitness stats help check convergence; timing shows performance tradeoffs

        if population[0].fitness == 0:
            print("target reached!")
            break

        mate(population, buffer)  #generate new candidates by mating
        population, buffer = swap(population, buffer)  #swap population and buffer

    #after ga run, plot fitness evolution and boxplots
    plot_fitness_evolution(best_history, mean_history, worst_history)
    plot_fitness_boxplots(fitness_distributions)

if __name__ == "__main__":
    main()
