import numpy as np
import logging
import os
import subprocess
import yaml
import threading
import time
import matplotlib.pyplot as plt
import random
import re


# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("ga_optimization.log"), logging.StreamHandler()])

# GA Parameters
POPULATION_SIZE = 50
MAX_GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.02
TOURNAMENT_SIZE = 3
ELITISM = True

# Parameters Range
P0_NOMINALWITHGRANT_RANGE = [-45, -60, -75, -90, -105, -120, -135, -150, -165, -180]
SSPBCH_BLOCKPOWER_RANGE = [-3, -6, -9, -12, -15, -18, -21, -24, -27, -30]
PUSCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
PUCCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
MAX_RXGAIN_RANGE = list(range(10, 126))  # 0 to 125
PRACH_DTX_THRESHOLD_RANGE = list(range(100, 181))  # 100 to 180
NROF_UPLINK_SYMBOLS_RANGE = list(range(1, 7))  # 0 to 6
RSRP_THRESHOLD_SSB_RANGE = list(range(1, 30))  # 1 to 29
SESSION_AMBR_UL0_RANGE = list(range(100, 500))  # 100 to 900
SESSION_AMBR_DL0_RANGE = list(range(100, 500))  # 100 to 900

# Define the bounds for each parameter
position_bounds = [
    (min(P0_NOMINALWITHGRANT_RANGE), max(P0_NOMINALWITHGRANT_RANGE)),
    (min(SSPBCH_BLOCKPOWER_RANGE), max(SSPBCH_BLOCKPOWER_RANGE)),
    (min(PUSCH_TARGETSNRX10_RANGE), max(PUSCH_TARGETSNRX10_RANGE)),
    (min(PUCCH_TARGETSNRX10_RANGE), max(PUCCH_TARGETSNRX10_RANGE)),
    (min(MAX_RXGAIN_RANGE), max(MAX_RXGAIN_RANGE)),
    (min(PRACH_DTX_THRESHOLD_RANGE), max(PRACH_DTX_THRESHOLD_RANGE)),
    (min(NROF_UPLINK_SYMBOLS_RANGE), max(NROF_UPLINK_SYMBOLS_RANGE)),
    (min(RSRP_THRESHOLD_SSB_RANGE), max(RSRP_THRESHOLD_SSB_RANGE)),
    (min(SESSION_AMBR_UL0_RANGE), max(SESSION_AMBR_UL0_RANGE)),
    (min(SESSION_AMBR_DL0_RANGE), max(SESSION_AMBR_DL0_RANGE))
]

# Deployments using OSM
def deploy_osm_network_slice():
    logging.info("Deploying OAI 5G Core using OSM")
    os.system("osm ns-create --ns_name 5G_SLICE_OAI --nsd_name 5G_CORE_SLICED_NSD --vim_account 5GSTACK")
    time.sleep(180)  # Wait 3 minutes for the 5G Core to be deployed

    logging.info("Deploying OAI 5G gNB using OSM")
    os.system("osm ns-create --ns_name 5G_SLICE_GNB --nsd_name 5G_GNB_SLICED_NSD --vim_account 5GSTACK")
    time.sleep(180)  # Wait 3 minutes for the gNB to be deployed

    logging.info("Deploying OAI 5G UE using OSM")
    os.system("osm ns-create --ns_name 5G_SLICE_UE --nsd_name 5G_UE_SLICED_NSD --vim_account 5GSTACK")
    time.sleep(180)  # Wait 3 minutes for the UE to be deployed

def modify_config_map(chromosome):
    # Load the existing core.yaml for SMF configuration
    with open('core.yaml', 'r') as file:
        core_config = yaml.safe_load(file)
    
    # Modify session AMBR values in the SMF configuration
    for info in core_config['data']['config.yaml']['smf']['local_subscription_infos']:
        if info['single_nssai']['sst'] == 1:  # Assuming SST 1 is the target slice
            info['qos_profile']['session_ambr_ul'] = f"{chromosome[8]}Mbps"
            info['qos_profile']['session_ambr_dl'] = f"{chromosome[9]}Mbps"
    
    # Save the modified core.yaml
    with open('core_modified.yaml', 'w') as file:
        yaml.dump(core_config, file)

    # Apply the modified core.yaml using kubectl
    subprocess.run(['kubectl', 'apply', '-f', 'core_modified.yaml'], check=True)

    # Load the existing gnb.yaml for gNB configuration
    with open('gnb.yaml', 'r') as file:
        gnb_config = yaml.safe_load(file)

    # Modify the gNB configuration values
    gnb_conf_str = gnb_config['data']['gnb.conf']
    gnb_conf_str = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {chromosome[0]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {chromosome[2]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {chromosome[3]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {chromosome[5]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {chromosome[6]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {chromosome[7]}", gnb_conf_str)

    # Update the gNB configuration in the ConfigMap
    gnb_config['data']['gnb.conf'] = gnb_conf_str

    # Save the modified gnb.yaml
    with open('gnb_modified.yaml', 'w') as file:
        yaml.dump(gnb_config, file)

    # Apply the modified gnb.yaml using kubectl
    subprocess.run(['kubectl', 'apply', '-f', 'gnb_modified.yaml'], check=True)

def run_iperf(pod_name_start, jitter_results):
    try:
        full_pod_name = get_full_pod_name(pod_name_start)
        if not full_pod_name:
            logging.error(f"Pod starting with '{pod_name_start}' not found.")
            jitter_results.append(float('inf'))  # Use a high jitter value as an indication of failure
            return

        # Run iperf command in client mode, adjust the server IP and other parameters as needed
        iperf_command = f"kubectl exec {full_pod_name} -- iperf -c <iperf-server-ip> -u -t 10"
        result = subprocess.run(iperf_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode()

        # Extract jitter value from iperf output
        jitter_match = re.search(r"Jitter: ([\d\.]+) ms", output)
        if jitter_match:
            jitter = float(jitter_match.group(1))
            jitter_results.append(jitter)
            
        else:
            logging.warning(f"Jitter not found in iperf output for {full_pod_name}")

    except Exception as e:
        logging.error(f"Error running iperf on pod starting with '{pod_name_start}': {e}")
        jitter_results.append(float('inf'))  # Use a high jitter value in case of error


def restart_pods():
    pod_prefixes = ['oai-gnb', 'oai-amf', 'oai-smf', 'oai-nr-ue']
    
    for prefix in pod_prefixes:
        # Get the full name of the pod using its prefix and the `kubectl get pods` command
        get_pod_cmd = f"kubectl get pods --no-headers -o custom-columns=\":metadata.name\" | grep ^{prefix}"
        try:
            pod_names = subprocess.check_output(get_pod_cmd, shell=True).decode().strip().split('\n')
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting pods with prefix {prefix}: {e}")
            continue
        
        # Delete each pod found with the prefix to initiate a restart
        for pod_name in pod_names:
            if pod_name:  # Ensure the pod name is not empty
                delete_pod_cmd = f"kubectl delete pod {pod_name}"
                try:
                    subprocess.run(delete_pod_cmd, shell=True, check=True)
                    
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to delete pod {pod_name}: {e}")
    
    
    time.sleep(60)  # Wait for 1 minute for pods to restart. Adjust this value as necessary for your setup.

# Initialization of population
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.array([np.random.uniform(low, high) for low, high in position_bounds])
        population.append(individual)
    return population

def evaluate_individual(individual):
    # Apply the network configuration based on the individual's chromosome
    modify_config_map(individual)  # You might need to adjust this function to work with individual chromosomes
    restart_pods()
    time.sleep(180)  # Wait for the network to stabilize after the configuration changes

    jitter_results = []  # Store jitter results from each UE
    threads = []  # To run iperf tests in parallel

    num_ues = 5  # Define the number of UEs you want to test; adjust as needed

    # Iterate over the number of UEs and create threads to run iperf tests
    for i in range(1, num_ues + 1):
        pod_name_start = f"oai-nr-ue-{i}-"  # Construct the pod name starting pattern
        thread = threading.Thread(target=run_iperf, args=(pod_name_start, jitter_results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate the overall jitter, e.g., as the average jitter across all UEs
    if jitter_results:
        average_jitter = sum(jitter_results) / len(jitter_results)
        
        return average_jitter
    else:
        logging.error("No jitter results collected for individual with chromosome {individual}.")
        return float('inf')  # Return a high value to indicate an issue

def evaluate_population(population):
    # Evaluate the fitness of the entire population at once and return the fitness scores
    fitness_scores = [evaluate_individual(individual) for individual in population]
    return fitness_scores

def select_parents(population, fitness_scores, tournament_size):
    parents = []
    population_size = len(population)

    for _ in range(population_size):
        # Selecting random indices for tournament participants
        indices = np.random.choice(range(population_size), size=tournament_size, replace=False)
        tournament_contestants = [population[index] for index in indices]
        tournament_scores = [fitness_scores[index] for index in indices]

        # Selecting the individual with the best (lowest for minimization) fitness score
        winner_index = np.argmin(tournament_scores)  # Use argmin for minimization problems
        winner = tournament_contestants[winner_index]

        # Adding the winner to the list of parents
        parents.append(winner)

    return parents

# Crossover
def crossover(parent1, parent2):
    offspring1, offspring2 = parent1.copy(), parent2.copy()  # Start by copying the parents
    if np.random.rand() < CROSSOVER_RATE:
        crossover_point = np.random.randint(1, len(parent1)-1)  # Choose a crossover point that is not at the ends of the chromosome
        # Swap the genes beyond the crossover point
        offspring1[crossover_point:], offspring2[crossover_point:] = parent2[crossover_point:], parent1[crossover_point:]
    return offspring1, offspring2

def mutate(individual):
    # Iterate through each gene in the individual
    for i in range(len(individual)):
        # Check if mutation should occur for this gene
        if np.random.rand() < MUTATION_RATE:
            # Determine the mutation amount, ensuring it's within the gene's bounds
            gene_bounds = position_bounds[i]  # Get the bounds for the current gene
            mutation_amount = np.random.uniform(-1, 1) * (gene_bounds[1] - gene_bounds[0]) * 0.1  # Mutation step size is 10% of the gene range
            
            # Apply the mutation
            individual[i] += mutation_amount
            
            # Ensure the mutated gene is within bounds
            individual[i] = np.clip(individual[i], gene_bounds[0], gene_bounds[1])


def create_next_generation(population, fitness_scores):
    next_generation = []
    
    if ELITISM:
        # Assuming minimization
        best_index = np.argmin(fitness_scores)
        best_individual = population[best_index]
        next_generation.append(best_individual)

    while len(next_generation) < POPULATION_SIZE:
        # Select parents with tournament selection
        parent1, parent2 = select_parents(population, fitness_scores, TOURNAMENT_SIZE)

        # Perform crossover
        offspring1, offspring2 = crossover(parent1, parent2)
        
        # Apply mutation
        mutate(offspring1)
        mutate(offspring2)
        
        # Add offspring to the new generation, checking for population size limit
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(offspring1)
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(offspring2)

    return next_generation

# Main GA loop
population = initialize_population()
best_fitness = float('inf')  # Assuming a minimization problem
best_individual = None

for generation in range(MAX_GENERATIONS):
    fitness_scores = evaluate_population(population)  # Evaluate the entire population at the start

    # Update the best individual based on the new fitness scores
    for individual, fitness in zip(population, fitness_scores):
        if fitness < best_fitness:  # Assuming minimization
            best_fitness = fitness
            best_individual = individual

    
    # Log the progress at the end of each generation
    logging.info(f"Generation {generation + 1}/{MAX_GENERATIONS}: Best Fitness = {best_fitness}")
    
    # Selection based on the fitness scores evaluated at the start
    parents = select_parents(population, fitness_scores, TOURNAMENT_SIZE)

    # Crossover and mutation to create a new generation
    new_population = create_next_generation(parents, CROSSOVER_RATE, MUTATION_RATE)

    # Replace the old population with the new one
    population = new_population

# Final logging after the GA loop
logging.info(f"GA Optimization completed. Best Fitness: {best_fitness}, Best Individual: {best_individual}")
