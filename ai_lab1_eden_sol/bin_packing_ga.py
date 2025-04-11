import random
import time
import matplotlib.pyplot as plt
import numpy as np 
import copy
import sys  # Import the sys module for command-line arguments

"""change"""
GA_POPSIZE = 2048      # GA population size
GA_MAXITER = 16384     # Maximum iterations
GA_ELITRATE = 0.10     # Elitism rate
GA_MUTATIONRATE = 0.25 # Mutation rate
MAX_AGE = 50


###############read files#####################
def read_falkenauer_file(filepath):
    instances = []

    with open(filepath, 'r') as file:
        num_instances = int(file.readline().strip())
        
        for _ in range(num_instances):
            name = file.readline().strip()
            capacity, num_items, opt_bins = map(int, file.readline().strip().split())
            item_sizes = []

            for _ in range(num_items):
                line = file.readline()
                if line.strip() == "":
                    continue
                item_sizes.append(int(line.strip()))

            instances.append({
                'name': name,
                'capacity': capacity,
                'num_items': num_items,
                'opt_bins': opt_bins,
                'items': item_sizes
            })

    return instances



####################################################################

###############bin packing problem##################################
class BinPacking_GA:
    def __init__(self, bins, bin_capacity):
        self.bins = bins
        self.capacity = bin_capacity
        self.fitness = len(bins) 
        self.age = 0
        self.rank = 0 


def init_bins(bins,bin_capacity,items,fit_type):
    i=1 #indicator that the last bin was not added to the list
    current_bin = []
    current_capacity = 0
    
    if fit_type == "d": #FFD heuristic 
        sorted(items,reverse=True)

    for item in items:
        #check to see if we can add the item to the bin
        if current_capacity + item <= bin_capacity:
            current_bin.append(item)
            current_capacity += item
            i = 0 
        #in case the item can't be added to the bin, add the current bin to the list of bins
        #and create a new bin
        else:
            if current_bin:
                bins.append(list(current_bin))
                i = 1 
            current_bin = [item]
            current_capacity = item 
        
    if current_bin and i == 0:
        bins.append(list(current_bin))

    return bins

def init_population( bin_capacity, items, fit_type):
    population = []
    #create random order of items and insert them into bins 
    for _ in range(GA_POPSIZE):
        bins = [] 
        random.shuffle(items)
        population.append(BinPacking_GA(init_bins(bins,bin_capacity,items,fit_type),bin_capacity,))

    return population



def calc_fitness(population):
    for individual in population:
        #the fitness is the number of bins - lower value = better fitness 
        individual.fitness = len(individual.bins)

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)


def fitness_ranking_linear(population):
    ranked_population = sorted(population, key=lambda x: x.fitness)
    for rank, member in enumerate(ranked_population):
        member.rank = rank  # 0 is best if minimizing

def select_parents(population, num_parents = 2):
    """Tournament selection."""
    fitness_ranking_linear(population)
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, 5)  # Tournament size of 5
        winner = min(tournament, key=lambda x: x.rank)
        parents.append(winner)
    return parents


def crossover(parent1, parent2, bin_capacity, items, fit_type):
    # Binary Partition Crossover (Falkenauer-style BPCX for bin packing)
    clone1 = [bin[:] for bin in parent1.bins]
    clone2 = [bin[:] for bin in parent2.bins]

    # Select crossover points (2 points for each parent)
    cx1_point1 = random.randint(0, len(clone1) - 1)
    cx1_point2 = random.randint(cx1_point1, len(clone1) - 1)
    cx2_point1 = random.randint(0, len(clone2) - 1)
    cx2_point2 = random.randint(cx2_point1, len(clone2) - 1)

    # Copy selected bins from both parents
    if(cx1_point1==cx1_point2):
        cx1_point2= cx1_point1+1
    if(cx2_point1==cx2_point2):
        cx2_point2= cx2_point1+1

    bins_1 = clone1[cx1_point1:cx1_point2]
    bins_2 = clone2[cx2_point1:cx2_point2]

    #in case it is the last bin)
    if(cx1_point1 == len(clone1) - 1):
        bins_1 = clone2[cx1_point1:]
    if(cx2_point1 == len(clone2) - 1):
        bins_2 = clone2[cx2_point1:]
 

    # Transfer bins between parents
    temp_1 = clone1[:cx1_point1] + bins_2 + clone1[cx1_point1:]
    temp_2 = clone2[:cx2_point1] + bins_1 + clone2[cx2_point1:]
    
    #index - we will use later to make sure that the transfered bins arent tampered with
    s1 = cx1_point1 + len(bins_2)
    s2 = cx2_point1 + len(bins_1)

    # Count appearance of each item
    items_1 = [item for bin in temp_1 for item in bin]
    items_2 = [item for bin in temp_2 for item in bin]

    item_freq = {}
    item_freq_1 = {}
    item_freq_2 = {}

    for item in items:
        item_freq[item] = item_freq.get(item, 0) + 1
    
    for item in items_1:
        item_freq_1[item] = item_freq_1.get(item, 0) + 1
    
    for item in items_2:
        item_freq_2[item] = item_freq_2.get(item, 0) + 1

    # Find duplicate items
    duplicates_1 = {}
    for item, count in item_freq.items():
        count1 = item_freq_1.get(item, 0)
        if count < count1:
            duplicates_1[item] = count1 - count
    
    duplicates_2 = {}
    for item, count in item_freq.items():
        count2 = item_freq_2.get(item, 0)
        if count < count2:
            duplicates_2[item] = count2 - count

   
    # Remove duplicates from bins
    def remove_duplicates(temp, duplicates,point1,point2):
        remaining_bins = []
        leftover_bins = [] 
        for bin in temp:
            i = 0 
            removed = False
            if i < point1 or i >= point2: #check that this are not the selected bins we got from the crossover
                for item in bin:
                    if duplicates.get(item, 0) > 0:
                        bin.remove(item)
                        duplicates[item] -= 1
                        removed = True          
                if len(bin)!=0:  # Add the bin if it still contains items
                    if removed:
                        leftover_bins.append(bin)
                    else:
                        remaining_bins.append(bin)
            
            elif bin!=[]:
                remaining_bins.append(bin)

            
            i += 1 

        return remaining_bins,leftover_bins

    clone1 = copy.deepcopy(temp_1)
    clone2 = copy.deepcopy(temp_2)
    #leftover bins are the bins that had items removed and now we will find a new bin for the remaining items
    #temp bins are the bins that remained the same (no item was removed)
    temp_1, leftover_bins1 = remove_duplicates(clone1, duplicates_1,cx1_point1,s1)
    temp_2, leftover_bins2 = remove_duplicates(clone2, duplicates_2,cx2_point1,s2)
    remaining1 =  [item for bin in leftover_bins1 for item in bin]
    remaining2 =  [item for bin in leftover_bins2 for item in bin]

    # Reallocate remaining items using FFD
    def allocate_items(temp, remaining_items,fit_type):
        if fit_type == "d": 
            sorted(remaining_items,reverse=True)
        for item in remaining_items:
            placed = False
            for bin in temp:
                if sum(bin) + item <= bin_capacity:
                    bin.append(item)
                    placed = True
                    break
            if not placed:
                temp.append([item])


    allocate_items(temp_1,remaining1,fit_type)
    allocate_items(temp_2,remaining2,fit_type)


    # Create offspring
    offspring1 = BinPacking_GA(temp_1, bin_capacity)
    offspring2 = BinPacking_GA(temp_2, bin_capacity)

    return offspring1, offspring2

def mutate(individual, bin_capacity, items):
    """Move or swap mutation."""

    mutation_type = random.choice(["move", "swap"])

    if mutation_type == "move":
        if len(individual.bins) >= 1:
            bin_from_index = random.randrange(len(individual.bins)) #choose a random bin 
            if individual.bins[bin_from_index]:
                placed = False
                item_to_move = random.choice(individual.bins[bin_from_index]) #choose a random item to move
                individual.bins[bin_from_index].remove(item_to_move)
                if not individual.bins[bin_from_index]: #if the bin is empty after removing the item then delete the bin 
                    del individual.bins[bin_from_index]

                # Try to fit the item into an existing bin
                random.shuffle(individual.bins)
                for i in range(len(individual.bins)):
                    if sum(individual.bins[i]) + item_to_move <= bin_capacity:
                        individual.bins[i].append(item_to_move)
                        placed = True
                        break
                if not placed:
                    # If it doesn't fit, create a new bin
                    individual.bins.append([item_to_move])

    elif mutation_type == "swap":
        if len(individual.bins) >= 2: #if a memeber in the population has at least 2 bins 
            #select 2 bins randomly 
            bin_index1 = random.randrange(len(individual.bins)) 
            bin_index2 = random.randrange(len(individual.bins))
            if bin_index1 != bin_index2 and individual.bins[bin_index1] and individual.bins[bin_index2]:
                #select 2 items randomly
                item1_index = random.randrange(len(individual.bins[bin_index1]))
                item2_index = random.randrange(len(individual.bins[bin_index2]))
                item1 = individual.bins[bin_index1][item1_index]
                item2 = individual.bins[bin_index2][item2_index]

                # Check if the swap is feasible
                if (sum(individual.bins[bin_index1]) - item1 + item2 <= bin_capacity and
                        sum(individual.bins[bin_index2]) - item2 + item1 <= bin_capacity):
                    individual.bins[bin_index1][item1_index] = item2
                    individual.bins[bin_index2][item2_index] = item1
                    
    return individual


def aging(population,items,fit_type):
    for i in range(GA_POPSIZE):
        population[i].age += 1 #Age Increment
        if population[i].age > MAX_AGE: #Max Age Threshold
            bins = []
            bin_capacity = population[i].capacity
            random.shuffle(items)
            population[i] = BinPacking_GA(init_bins(bins,bin_capacity,items,fit_type),bin_capacity)

def adaptive_fitness(population,generation):
    for member in population:
        base_fitness = member.fitness
        penelty = int((generation / GA_MAXITER)*10)
        member.fitness = base_fitness + penelty #the higher the fitness value the worst the fitness is  

def elitism(population, buffer, esize, bin_capacity):
    bins = [] 
    buffer[:esize] = [BinPacking_GA(bins,bin_capacity) for _ in range(esize)]
    for i in range(esize):
        buffer[i].bins = population[i].bins
        buffer[i].fitness = population[i].fitness

def mate(population, buffer,items, bin_capacity,fit_type):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize, bin_capacity)
    parents = select_parents(population,GA_POPSIZE - esize)
    for i in range(0, len(parents), 2 ):
        parent1 = parents[i]
        parent2 = parents[(i + 1) % len(parents)]
        offspring1,offspring2 = crossover(parent1, parent2, bin_capacity,items,fit_type)
        if random.random() < GA_MUTATIONRATE:
             buffer[esize + i]=(mutate(offspring1, bin_capacity, items))
        else: 
             buffer[esize + i]=(offspring1)
        
        if esize + i + 1 < GA_POPSIZE:
            if random.random() < GA_MUTATIONRATE:
                buffer[esize + i + 1] = mutate(offspring2, bin_capacity, items)
            else:
                buffer[esize + i + 1] = offspring2
        
      

def print_best(population):
    print(f"Best: {population[0].bins} ({population[0].fitness})")

def main():
    random.seed(time.time())
    if len(sys.argv) != 4:
        print("Usage: python bin_packing_ga.py <falkenauer_file> <fit_type> <max_time_seconds>")
        print("Example fit_type: 'd' for FFD, 'f' for first fit, etc.")
        sys.exit(1)

    filepath = sys.argv[1]
    fit_type = sys.argv[2]
    max_time_seconds = sys.argv[3]
    #filepath = r"C:\Users\Eden\Documents\Uni\AI Lab\Lab_1\binpack1.txt"
    instances_data = read_falkenauer_file(filepath)
    problems = 5 
    #fit_type = "d"
    for instance in instances_data:
        problems -= 1
        if problems < 0:
            break
        bin_capacity = instance['capacity']
        items = instance['items']
        opt_bins = instance.get('opt_bins', 'N/A')

        print(f"\n--- Processing instance: {instance['name']} ---")
        start_time = time.time()#clock
        population = init_population(bin_capacity,items,fit_type)
        buffer = [BinPacking_GA([],bin_capacity) for _ in range(GA_POPSIZE)]

        for i in range(GA_MAXITER):
            print(f"Generation {i}:")
            calc_fitness(population)
            adaptive_fitness(population,i)
            sort_by_fitness(population)
            print_best(population)

            if population[0].fitness <= opt_bins:
                break
            
            if(time.time() - start_time >= max_time_seconds):
                break

            aging(population,items,fit_type) 
            mate(population,buffer,items,bin_capacity,fit_type)
            population, buffer = buffer, population
      
        
        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time >= max_time_seconds :
            print(f"The run time has exceeded max time of {max_time_seconds} seconds.")
            break

        if fit_type=="d":
            heuristic = "FFD"
        else:
            heuristic = "FF"
        print(f"Execution time for {instance['name']}: {execution_time:.4f} seconds , Best solution found for {instance['name']}: {population[0].fitness}  ,Optimal solution for {instance['name']}: {opt_bins} , heuristic: {heuristic}")


if __name__ == "__main__":
    main()
