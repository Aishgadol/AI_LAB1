import random
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np 
import sys  # Import the sys module for command-line arguments
"""create a python version of genetics5.cpp """

GA_POPSIZE = 2048      # GA population size
GA_MAXITER = 16384     # Maximum iterations
GA_ELITRATE = 0.10     # Elitism rate
GA_MUTATIONRATE = 0.25 # Mutation rate
GA_TARGET = "Hello world!"
MAX_AGE = 10

def random_string(length):
    return ''.join(chr(random.randint(32, 122)) for _ in range(length))

class GA_Struct:
    def __init__(self, length):
        self.str = random_string(length)
        self.fitness = 0
        self.rank = len(GA_TARGET) * 2
        self.prob = 1 
        self.age = 0 

def init_population():
    return [GA_Struct(len(GA_TARGET)) for _ in range(GA_POPSIZE)]


########################## New Heuristic #######################################
def LCS(s1,s2):#q7
    dp = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
    #calculate the longest common subsequence via a matrix 
    for i in range(len(s1)+1):
        for j in range(len(s2)+1):
            if i==0 or j==0:
                dp[i][j]=0 #the first row and first col are initilize to 0
            elif s1[i-1]==s2[j-1]: #if the chars are the same then add +1 to the correct postion in the matrix
                dp[i][j]=dp[i-1][j-1]+1
            else:  #else add the maximum value from the row before or col before the corrent one
                dp[i][j]= max(dp[i-1][j],dp[i][j-1])
    
    return dp[len(s1)][len(s2)]

def calc_fitness_new(population):
    for member in population:
        member.fitness = len(GA_TARGET) * 2 
        
        #LCS Heuristic:
        lcs_score = LCS(member.str, GA_TARGET) #the bigger value the better
        #Bonus if the chars in the correct position:
        correct_pos = sum(1 for i in range(len(GA_TARGET)) if member.str[i]==GA_TARGET[i]) #if eq to len(GA_TARGET) then perfect
        #calculate fitness:
        member.fitness -= lcs_score + correct_pos 

################################################################################

def calc_fitness(population):
    for member in population:
        member.fitness = sum(abs(ord(member.str[i]) - ord(GA_TARGET[i])) for i in range(len(GA_TARGET)))

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def elitism(population, buffer, esize):
    buffer[:esize] = [GA_Struct(len(GA_TARGET)) for _ in range(esize)]
    for i in range(esize):
        buffer[i].str = population[i].str
        buffer[i].fitness = population[i].fitness

########################### parent selection #############################################
def linear_scaling(population):
    fitnesses = [member.fitness for member in population]
    f_min = min(fitnesses)
    f_max = max(fitnesses)
    f_avg = sum(fitnesses) / len(fitnesses)
    f_range = f_max - f_min


    # Decide scaling strategy dynamically
    if f_range < f_max/2:
        a = 1.5  # boost small differences
        b = 0
    elif f_range > f_max/2 - 4:
        a = 0.5# dampen strong dominance
        b = 0
    else:
        # Default adaptive linear scaling (like before)
        scaling_factor = 2
        a = (scaling_factor - 1) * f_avg / (f_max - f_min)
        b = f_avg * (f_max - scaling_factor * f_min) / (f_max - f_min)

    return a,b
def RWS(population): #q10

    #a and b are constants chosen to adjust the range of fitness values.
    #look back on that !!!!
    a = 1.1
    b=0

    a,b = linear_scaling(population)
    selected_idx = -1
    parents = []
    cumulative_probs = []

    #linare scaling + sum total fitness : 
    fitness_vals = [a * member.fitness + b for member in population]
    inverse_fitness = [(1 / (f + 1e-8)) for f in fitness_vals]  #since low fitness is better
    total_fitness = sum(inverse_fitness)

    # Check if total fitness is zero
    if total_fitness == 0:
        return random.sample(population, 2)  # Return 2 random members if all fitness is zero
    
    #cumulative probability:
    p=0
    for i, member in enumerate(population):
        p += inverse_fitness[i] / total_fitness #calculate probabilty 
        cumulative_probs.append(p)

    #select 2 parents: 
    for _ in range(2):
        random_number = random.random()
        for i, member in enumerate(population):            
            if cumulative_probs[i] > random_number and selected_idx!=i:
                parents.append(member)
                selected_idx = i 
                break

    return tuple(parents)
        


def SUS(population):

    a,b = linear_scaling(population)
    selected_idx = -1
    parents = []
    cumulative_fitness = 0
    index = 0 
    num_offspring = 2 

    #linare scaling: 
    fitness_vals = [a * member.fitness + b for member in population]
    total_fitness = sum(fitness_vals)
    
    # Check if total fitness is zero
    if total_fitness == 0:
        return random.sample(population, 2)
    
    #Normalize fitness:
    normalized_fitness = [fitness/total_fitness for fitness in fitness_vals]
    
    #calculate pointer_distance
    pointer_distance = 1/num_offspring #number of parents = 2
    start_point = random.uniform(0, pointer_distance)
    pointers = [start_point + i * pointer_distance for i in range(num_offspring)]
    
    for pointer in pointers:
        while cumulative_fitness < pointer and index<len(population):
            cumulative_fitness += normalized_fitness[index]
            index += 1

        if selected_idx != index - 1 :
            parents.append(population[index-1])
            selected_idx = index -1 
        
    
    return tuple(parents)


def fitness_ranking_linear(population):
    ranked_population = sorted(population, key=lambda x: x.fitness)
    for rank, member in enumerate(ranked_population):
        member.rank = rank  # 0 is best if minimizing

def fitness_ranking_exponential(population,decay_rate=1.0):
    #(lower fitness = better)
    ranked_population = sorted(population, key=lambda x: x.fitness)
    n = len(ranked_population)
    
    #Assign ranks: best gets rank 0 , Compute exponential weights
    weights = [np.exp(-decay_rate * rank) for rank in range(n)]

    #Normalize weights to sum to 1 (turn into probabilities)
    total = sum(weights)
    probabilities = [w / total for w in weights]

    #Pair individual with its selection probability
    for rank, (member,prob) in enumerate(zip(ranked_population,probabilities)):
        member.rank = rank  # 0 is best if minimizing
        member.prob = prob #!!!!!!!!!!!!!!!!!!check to change , maybe use in RWS and SUS

def tournament_deterministic(population,K):
    parents = []
    for _ in range(2):  # select 2 parents
        competitors = random.sample(population, K)
        winner = min(competitors, key=lambda x: x.rank)
        parents.append(winner)
    return tuple(parents)

def tournament_probabilistic(population, K=3, P=0.75):
    parents = []
    for _ in range(2):  # select 2 parents
        competitors = sorted(random.sample(population, K), key=lambda x: x.rank)
        r = random.random()
        if r < P:
            winner = competitors[0]
        else:
            winner = random.choice(competitors[1:])  # pick randomly from the rest
        parents.append(winner)
    return tuple(parents)

def aging(population):
    for i in range(GA_POPSIZE):
        population[i].age += 1 #Age Increment
        if population[i].age > MAX_AGE: #Max Age Threshold
            population[i] = GA_Struct(len(GA_TARGET))

##########################################################################################

def mate_parent_selection(op,selection_type,population, buffer):#q10
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    tsize = len(GA_TARGET)
    
    for i in range(esize, GA_POPSIZE):
        #selection parents method:
        match selection_type:
            case "tournament probabilistic":
                fitness_ranking_linear(population)
                parent1, parent2 = tournament_probabilistic(population,3,0.75)
            case "tournament deterministic":
                fitness_ranking_linear(population)
                parent1, parent2 = tournament_deterministic(population,3)
            case "RWS":
                parent1, parent2 = RWS(population)
            case "SUS":
                parent1, parent2 = SUS(population)
            case _: 
                raise ValueError("Invalid selection method")
        
        #choose operator:
        match op:
            case "single":
                new_str = single_cross(parent1, parent2)
            case "two":
                new_str = two_cross(parent1, parent2)
            case "uniform": 
                new_str = uniform_cross(parent1, parent2)  
            case _: 
                raise ValueError("Invalid crossover operator")
        
        buffer[i] = GA_Struct(len(GA_TARGET))
        buffer[i].str = new_str
        
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])

############################ Crossover Operators #########################################

def single_cross(parent_1, parent_2): #q4
    tsize = len(GA_TARGET)
    spos = random.randint(0, tsize - 1)
    new_str = parent_1.str[:spos] + parent_2.str[spos:]
    return new_str

def two_cross(parent_1, parent_2): #q4
    tsize = len(GA_TARGET)
    #find two postions in the strings 
    spos1 = random.randint(0, tsize - 2)#-2 to ensure that there's at least one character available 
    spos2 = random.randint(spos1 + 1, tsize - 1)
    new_str = parent_1.str[:spos1] + parent_2.str[spos1:spos2] + parent_1.str[spos2:]
    return new_str

def uniform_cross(parent_1, parent_2): #q4
    tsize = len(GA_TARGET)
    new_str =""
    #each bit is selected randomly
    for i in range(tsize):
        if random.random()<0.5:
            new_str+=parent_2.str[i]
        else:
            new_str+=parent_1.str[i]
    return new_str

def mate_new(op,mutation_type,population, buffer):#q4
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    tsize = len(GA_TARGET)
    
    for i in range(esize, GA_POPSIZE):
        #random parents pick:
        i1 = random.randint(0, GA_POPSIZE // 2 - 1)
        i2 = random.randint(0, GA_POPSIZE // 2 - 1)
        #choose operator:
        match op:
            case "single":
                new_str = single_cross(population[i1], population[i2])
            case "two":
                new_str = two_cross(population[i1], population[i2])
            case "uniform": 
                new_str = uniform_cross(population[i1], population[i2])
            case "none":
                #this means that we run without a crossover opertaion
                new_str = random_string(len(GA_TARGET)) #so we generate a random string instead  
            case _: 
                raise ValueError("Invalid crossover operator")
        
        buffer[i] = GA_Struct(len(GA_TARGET))
        buffer[i].str = new_str
        
        if random.random() < GA_MUTATIONRATE and mutation_type:
            mutate(buffer[i])



#################################################################################################

def mutate(member):
    tsize = len(GA_TARGET)
    ipos = random.randint(0, tsize - 1)
    delta = random.randint(32, 122)
    member.str = member.str[:ipos] + chr((ord(member.str[ipos]) + delta) % 122) + member.str[ipos + 1:]

def print_best(population):
    print(f"Best: {population[0].str} ({population[0].fitness})")


####################### log states ################################
def log_states(population, gen, start,start_time_gen, logs_fit):#q1+2
    #fitness states:
    fit_vals = np.array([member.fitness for member in population])
    avg_fit = sum(fit_vals) / len(fit_vals)
    worst_fit = max(fit_vals)
    range_fit = worst_fit - min(fit_vals) 
    std_fit = np.std(fit_vals)
    #time:
    elapsed_time = time.time() - start
    clock_ticks = time.time() - start_time_gen

    print(f"Generation {gen}: Avg Fitness = {avg_fit:.2f}, Worst Fitness = {worst_fit}, Fitness Range = {range_fit}, Standard devation = {std_fit:.2f} ")
    print(f"Elapsed Time = {elapsed_time:.2f}s, Clock Ticks Generation = {clock_ticks:.2f}")

    #store logs for graphs:
    logs_fit["best"].append(min(fit_vals))
    logs_fit["avg"].append(avg_fit)
    logs_fit["worst"].append(worst_fit)
    logs_fit["all"].append(fit_vals)

def plot_fitness_behavior(logs_fit):
    generations = range(len(logs_fit["best"]))

    # Plot 1: Line Graph for Best, Avg, Worst fitness
    plt.figure(figsize=(12, 5))
    plt.plot(generations, logs_fit["best"], label="Best Fitness", color='green')
    plt.plot(generations, logs_fit["avg"], label="Average Fitness", color='blue')
    plt.plot(generations, logs_fit["worst"], label="Worst Fitness", color='red')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Evolution Over Generations")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot 2: Boxplot for Fitness Distribution Per Generation
    plt.figure(figsize=(12, 5))
    plt.boxplot(logs_fit["all"], showfliers=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Distribution Per Generation (Boxplot)")
    plt.grid()
    plt.show()

def log_selection_pressure(population):#q8
    
    #fitness variance
    fitness_vals = np.array([member.fitness for member in population])
    fitness_var = np.var(fitness_vals)#

    #top average selection probability ratio
    inverse_fitness = 1/(fitness_vals + 1e-8) #since low fitness is better
    probabilites =  inverse_fitness / np.sum(inverse_fitness)
    sorted_indices = np.argsort(-probabilites)#soet in descending order
    top_10_percent = int(len(population) * 0.1)
    top_fitness_probs= probabilites[sorted_indices[:top_10_percent]]
    all_avg = np.mean(probabilites)
    top_avg = np.mean(top_fitness_probs)
    top_avg_ratio = top_avg/all_avg
    
    print(f"Fitness Variance: {fitness_var:.2f}")
    print(f"Top-Average Selection Probability Ratio: {top_avg_ratio:.2f}")

def log_genetic_diversification(population):#q9
    #distance - averahe hamming distance
    population_strs = [member.str for member in population]
    p_len = len(population)
    #calculate the distance between every string in population
    dis_array = [
        sum(1 for a, b in zip(population_strs[i], population_strs[j]) if a!=b) 
        for i in range(p_len) for j in range(i+1,p_len)
                 ]
    avg_ham_dis = np.mean(dis_array)

    #count the different alleles and their frequancies
    temp =  [set(seq[i] for seq in population_strs) for i in range(len(population_strs[0]))] #avoid duplicates, characters in position i 
    alleles_count = sum(len(alleles) for alleles in temp) 
    
    alleles_freq = {}
    total_count = 0
    for i in range(len(population_strs[0])):
        temp = [seq[i] for seq in population_strs]
        for allele in set(temp):
            count = temp.count(allele)
            alleles_freq[(i,allele)] = count
            total_count += count
    
    #shanon entropy:
    shanon_entropy = 0.0
    for val in alleles_freq.values():
        p = val / total_count
        if p>0:
            shanon_entropy -= p * np.log2(p)
    
    

    print(f"Average Hamming Distance: {avg_ham_dis:.2f}")
    print(f"Unique alleles: {alleles_count:.2f}")
    print(f"Shanon entropy: {shanon_entropy:.2f}")


###################################################################

def main(): 
    random.seed(time.time())
    if len(sys.argv) != 5:
        print("Usage: python GA_solution.py <heuristic_type> <mutation_type> <crossover_type> <selection_type> <max_time_seconds>")
        sys.exit(1)

    heuristic_type = sys.argv[1]
    mutation = sys.argv[2]
    crossover_type = sys.argv[3]
    selection_type = sys.argv[4]
    max_time_seconds = sys.argv[5]

    # Validation checks
    if heuristic_type not in ["LCS", "original"]:
        raise ValueError(f"Warning: Invalid heuristic type '{heuristic_type}'. Options: LCS, original")

    if mutation not in ["y,n"]:
        raise ValueError(f"Warning: Invalid mutation type '{mutation_type}'. Options: y,n")

    if crossover_type not in ["single", "two", "uniform", "none"]:
        raise ValueError(f"Warning: Invalid crossover operator '{crossover_type}'. Options: single, two, uniform , none")

    if selection_type not in ["tournament deterministic", "tournament probabilistic", "RWS", "SUS", "none"]:
        raise ValueError(f"Warning: Invalid selection method '{selection_type}'. Options: tournament deterministic ,tournament probabilistic, RWS, SUS, none")

    population = init_population()
    buffer = [GA_Struct(len(GA_TARGET)) for _ in range(GA_POPSIZE)]
    start_time = time.time()#clock
    logs_fit = {"best": [], "avg": [], "worst": [], "all": []}#logs for grahps

    if mutation == "y":
        mutation_type = True
    else:
        mutation_type = False 


    for i in range(GA_MAXITER):
        start_time_gen = time.time()
        
        if heuristic_type == "LCS":
            calc_fitness_new(population) #q7
        else:
            calc_fitness(population)
        
        sort_by_fitness(population)
        print_best(population)
        log_states(population,i,start_time,start_time_gen,logs_fit)#q1+2
        log_selection_pressure(population)#q8
        log_genetic_diversification(population)#9   takes a long ass time

        if population[0].fitness == 0:
            # Plot the final population
            print(f"Heuristic type: {heuristic_type} , Mutation: {mutation_type} , Crossover operator: {crossover_type}, Selection Method: {selection_type}")
            break
        
        if (time.time() - start_time  >= max_time_seconds):
            print(f"The run time has exceeded max time of {max_time_seconds} seconds.")
            break

        if selection_type == "none":
            mate_new(crossover_type,mutation_type,population, buffer)#q4
        else:
            mate_parent_selection(crossover_type,selection_type,population, buffer)#q10
        
        aging(population)
        population, buffer = buffer, population

    plot_fitness_behavior(logs_fit)#q3

if __name__ == "__main__":
    main()
