import json
import numpy as np 
import random
import time
import copy
import matplotlib.pyplot as plt
import sys  # Import the sys module for command-line arguments


GA_POPSIZE = 2000
GA_MAXITER = 1000
GA_ELITRATE = 0.10     # Elitism rate
GA_MUTATIONRATE = 0.25 # Mutation rate
GA_AGE = 50 

def load_arc_task(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class ARC_GA_Struct:
    def __init__(self, row, col):
        self.grid = np.zeros((row, col), dtype=int)
        self.values = [] 
        self.fitness = float('inf')
        self.age = 0

def init_population(input_grid,output_grid):
    #find expected grid sizes 
    rows = len(output_grid)
    cols = len(output_grid[0])
    values = [item for arr in input_grid for item in arr] #find the input values
    population = [] 
    for _ in range(GA_POPSIZE):
        member = ARC_GA_Struct(rows,cols)
        member.values = values
        member.grid = np.random.choice(values, size=(rows,cols))#randomize the solution
        population.append(member)
    return population 


def calc_fitness(population, target_grid):
    # Count non-matching cells
    for member in population:
        member.fitness = np.sum(member.grid != target_grid)

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def elitism(population, buffer,h,w ,esize):
    buffer[:esize] = [ARC_GA_Struct(h,w) for _ in range(esize)]
    for i in range(esize):
        buffer[i].grid = population[i].grid
        buffer[i].values = population[i].values
        buffer[i].age = population[i].age
        buffer[i].fitness = population[i].fitness

def arc_crossover(parent1, parent2):
    # Create child grid
    child = ARC_GA_Struct(parent1.grid.shape[0], parent1.grid.shape[1])
    
    # Choose random splitting point
    split_row = random.randint(0, parent1.grid.shape[0]-1)
    
    # Take top part from parent1, bottom part from parent2
    child.grid[:split_row, :] = parent1.grid[:split_row, :].copy()
    child.grid[split_row:, :] = parent2.grid[split_row:, :].copy()

    child.values = parent1.values
    
    return child

def arc_mutate(individual):
    # Choose random cell and change its value
    h, w = individual.grid.shape
    row = random.randint(0, h-1)
    col = random.randint(0, w-1)
    individual.grid[row, col] = np.random.choice(individual.values)  

def arc_mate(population, buffer, h,w):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer,h,w, esize)
    for i in range(esize, GA_POPSIZE):
        i1 = random.randint(0, GA_POPSIZE // 2 - 1)
        i2 = random.randint(0, GA_POPSIZE // 2 - 1)

        offspring = arc_crossover(population[i1],population[i2])
        buffer[i] = ARC_GA_Struct(h,w)
        buffer[i].grid = offspring.grid
        buffer[i].values = offspring.values

        if random.random() < GA_MUTATIONRATE:
            arc_mutate(buffer[i])

def print_best(population):
    print("Best:  ")
    print(f"{population[0].grid} ({population[0].fitness})")



def GA_algorithm(input_grid, output_grid,max_time_seconds,start_time_run_all):
    random.seed(time.time())
    start_time = time.time()#clock
    h = len(output_grid)
    w = len(output_grid[0])
    population = init_population(input_grid, output_grid)
    buffer = [ARC_GA_Struct(len(output_grid), len(output_grid[0])) for _ in range(GA_POPSIZE)]

    for i in range(GA_MAXITER):
        start_time_gen = time.time()

        calc_fitness(population,output_grid)
        sort_by_fitness(population)
        print_best(population)

        if population[0].fitness == 0:
            print(f"Perfect solution found at generation {i}")
            break
        
        arc_mate(population,buffer,h,w)
        population, buffer = buffer, population

        time_gen = time.time() - start_time_gen
        if(time.time() - start_time_run_all >= max_time_seconds):
            break

        print(f" clock ticks: {time_gen:.2f}")
    
    time_run = time.time() - start_time
    print(f"elapsed time : {time_run:.2f}")


def main():
    random.seed(time.time())
    if len(sys.argv) != 3:
        print("Usage: python arc_ga.py <jason_file> <max_time_seconds>")
        sys.exit(1)
    file_path = sys.argv[1]
    max_time_seconds = sys.argv[2]
    start_time = time.time()#clock

    data = load_arc_task(file_path)

    for i, temp in enumerate(data["train"]):
        input_grid = temp["input"]
        output_grid = temp["output"]
        print(f"problem training {i}")
        GA_algorithm(input_grid, output_grid,max_time_seconds,start_time)
        run_time = time.time() -start_time
        if  run_time>= max_time_seconds :
            print(f"The run time has exceeded max time of {max_time_seconds} seconds.")
            break
    
    if run_time <   max_time_seconds:
        for i, temp in enumerate(data["test"]):
            input_grid = temp["input"]
            output_grid = temp["output"]
            print(f"problem testing  {i}")
            GA_algorithm(input_grid, output_grid,max_time_seconds,start_time)
            run_time = time.time() -start_time
            if  run_time>= max_time_seconds :
                print(f"The run time has exceeded max time of {max_time_seconds} seconds.")
                break
        

if __name__ == "__main__":
    main()
