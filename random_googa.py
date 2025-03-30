import random
import time
import math

# constant parameters
GA_POPSIZE = 2048           # population size
GA_MAXITER = 16384          # maximum number of iterations
GA_ELITRATE = 0.10          # elitism rate (proportion of best individuals to retain)
GA_MUTATIONRATE = 0.25      # mutation probability
GA_TARGET = "Hello world!"  # target string

# candidate class representing an individual in the population
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene      # candidate's gene (string)
        self.fitness = fitness  # candidate's fitness score

# initialize population and buffer for new generation
def init_population():
    """
    creates initial population and a buffer list for the next generation
    """
    target_length = len(GA_TARGET)
    population = []
    # create population with random genes
    for _ in range(GA_POPSIZE):
        # generate a random string of length equal to target;
        # each character is a random ascii char between 32 and 121
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    # create a buffer list for the next generation
    buffer = [Candidate('') for _ in range(GA_POPSIZE)]
    return population, buffer

# calculate fitness for each candidate in the population
def calc_fitness(population):
    """
    computes fitness for each candidate as the sum of absolute differences
    between candidate's gene characters and target string characters
    """
    target = GA_TARGET
    target_length = len(target)
    for candidate in population:
        fitness = 0
        for i in range(target_length):
            # absolute difference in ascii values between candidate char and target char
            fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
        candidate.fitness = fitness

# sort the population based on fitness (lower fitness is better)
def sort_by_fitness(population):
    """
    sorts population in-place so that best candidate is at index 0
    """
    population.sort(key=lambda cand: cand.fitness)

# copy the top elite candidates from current population to buffer (elitism)
def elitism(population, buffer, elite_size):
    """
    copies the top elite_size candidates to the buffer list
    """
    for i in range(elite_size):
        buffer[i] = Candidate(population[i].gene, population[i].fitness)

# mutate a candidate by altering one random character in its gene
def mutate(candidate):
    """
    randomly mutates one character in the candidate's gene
    """
    target_length = len(GA_TARGET)
    pos = random.randint(0, target_length - 1)  # random position in gene
    # generate random delta value between 32 and 121
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    # update character by adding delta and taking modulo 122
    new_val = (old_val + delta) % 122
    # convert gene to list for mutation since strings are immutable
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)

# mate candidates to create the next generation via single-point crossover and mutation
def mate(population, buffer):
    """
    performs mating on the population to produce a new generation in buffer.
    it applies elitism, then single-point crossover and possible mutation.
    """
    elite_size = int(GA_POPSIZE * GA_ELITRATE)
    target_length = len(GA_TARGET)

    # copy elite candidates directly to buffer
    elitism(population, buffer, elite_size)

    # create the rest of the new generation using crossover and mutation
    for i in range(elite_size, GA_POPSIZE):
        # choose two parents from the top half of the population
        parent1 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        parent2 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        # select a random crossover point
        crossover_point = random.randint(0, target_length - 1)
        # single-point crossover: combine parent's gene segments
        new_gene = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
        buffer[i] = Candidate(new_gene)
        # apply mutation with a probability defined by GA_MUTATIONRATE
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])

# print the best candidate in the current population
def print_best(population):
    """
    prints the candidate with the best (lowest) fitness score
    """
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")

# swap the current population with the buffer for the next generation
def swap(population, buffer):
    """
    returns buffer and population swapped
    """
    return buffer, population

# main genetic algorithm loop
def main():
    # seed random number generator with current time
    random.seed(time.time())
    population, buffer = init_population()

    for iteration in range(GA_MAXITER):
        calc_fitness(population)     # calculate fitness for each candidate
        sort_by_fitness(population)  # sort candidates by fitness
        print_best(population)       # print the best candidate

        # if a candidate perfectly matches the target, exit loop
        if population[0].fitness == 0:
            print(f"solution found in {iteration} iterations")
            break

        mate(population, buffer)     # create new generation via mating
        population, buffer = swap(population, buffer)  # swap generations

if __name__ == "__main__":
    main()
