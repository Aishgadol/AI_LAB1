import random
import time
import math

# constant parameters
GA_POPSIZE = 16384  # population size
GA_MAXITER = 16384  # maximum iterations
GA_ELITRATE = 0.10  # elitism rate (10%)
GA_MUTATIONRATE = 0.25  # mutation probability
GA_TARGET = "but why did you cheat on me? please fuck me harder"  # target string


# candidate class representing an individual
class Candidate:
    def __init__(self, gene, fitness=0):
        self.gene = gene  # candidate string
        self.fitness = fitness  # fitness score


# initialize population and buffer
def init_population():
    target_length = len(GA_TARGET)
    population = []
    # create initial population with random genes
    for _ in range(GA_POPSIZE):
        # random char between ascii 32 and 121 for each gene position
        gene = ''.join(chr(random.randint(32, 121)) for _ in range(target_length))
        population.append(Candidate(gene))
    # create buffer list with empty candidates
    buffer = [Candidate('') for _ in range(GA_POPSIZE)]
    return population, buffer


# calculate fitness for each candidate
def calc_fitness(population):
    target = GA_TARGET
    target_length = len(target)
    for candidate in population:
        fitness = 0
        # sum of absolute differences for each character
        for i in range(target_length):
            fitness += abs(ord(candidate.gene[i]) - ord(target[i]))
        candidate.fitness = fitness


# sort population by fitness (lower is better)
def sort_by_fitness(population):
    population.sort(key=lambda cand: cand.fitness)


# copy best candidates to next generation (elitism)
def elitism(population, buffer, elite_size):
    for i in range(elite_size):
        # copy candidate gene and fitness to buffer
        buffer[i] = Candidate(population[i].gene, population[i].fitness)


# mutate a candidate by altering one random gene position
def mutate(candidate):
    target_length = len(GA_TARGET)
    pos = random.randint(0, target_length - 1)  # random index in gene
    # delta is a random value between 32 and 121
    delta = random.randint(32, 121)
    old_val = ord(candidate.gene[pos])
    # update gene char by adding delta modulo 122
    new_val = (old_val + delta) % 122
    # convert string to list for mutation since strings are immutable
    gene_list = list(candidate.gene)
    gene_list[pos] = chr(new_val)
    candidate.gene = ''.join(gene_list)


# mate candidates to produce new generation
def mate(population, buffer):
    elite_size = int(GA_POPSIZE * GA_ELITRATE)
    target_length = len(GA_TARGET)

    # copy best candidates to buffer
    elitism(population, buffer, elite_size)

    # create rest of new generation by crossover and mutation
    for i in range(elite_size, GA_POPSIZE):
        # choose two parents from top half of population
        parent1 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        parent2 = population[random.randint(0, GA_POPSIZE // 2 - 1)]
        # random crossover point
        crossover_point = random.randint(0, target_length - 1)
        # single-point crossover: combine parent genes at crossover point
        new_gene = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
        buffer[i] = Candidate(new_gene)
        # apply mutation based on mutation rate
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])


# print best candidate from population
def print_best(population):
    best = population[0]
    print(f"best: {best.gene} ({best.fitness})")


# swap population and buffer for next generation
def swap(population, buffer):
    return buffer, population


# main genetic algorithm loop
def main():
    random.seed(time.time())  # seed random number generator
    population, buffer = init_population()

    for _ in range(GA_MAXITER):
        calc_fitness(population)  # calculate fitness for each candidate
        sort_by_fitness(population)  # sort candidates by fitness
        print_best(population)  # print current best candidate

        # if best candidate matches target exactly, exit loop
        if population[0].fitness == 0:
            break

        mate(population, buffer)  # generate new candidates via mating
        population, buffer = swap(population, buffer)  # swap generation buffers


if __name__ == "__main__":
    main()
