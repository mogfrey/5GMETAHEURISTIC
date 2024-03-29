import numpy as np
import matplotlib.pyplot as plt
import logging
import random
import yaml
import subprocess
import os
import time
import re


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pso_optimization.log"), logging.StreamHandler()])

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

# GA parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

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

def fitness_function(chromosome):
    

    # Modify system configuration based on chromosome's values
    modify_config_map(chromosome)
    restart_pods()
    time.sleep(180)  # Wait for 3 minutes after restarting pods to ensure the network is stable

    # Identify the UE pod
    ue_pod_cmd = "kubectl get pods --no-headers | grep '^oai-nr-ue' | awk '{print $1}'"
    ue_pod_name = subprocess.check_output(ue_pod_cmd, shell=True).decode().strip()

    if not ue_pod_name:
        logging.error("Could not find UE pod. Aborting evaluation.")
        return float('inf')  # Return a high latency to indicate failure

    # Ping the UPF pod from the UE pod
    destination_ip = "12.1.1.1"  # Destination IP address of the UPF pod
    ping_cmd = f"kubectl exec {ue_pod_name} -- ping -c 5 {destination_ip}"  # '-c 5' sends 5 packets
    ping_result = subprocess.run(ping_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ping_output = ping_result.stdout.decode()

    # Parse the output to find the average latency
    latency_match = re.search(r"avg = ([0-9\.]+)/", ping_output)
    if latency_match:
        average_latency = float(latency_match.group(1))
       
    else:
        logging.error("Could not parse average latency from ping output. Setting average latency to a high value.")
        average_latency = float('inf')  # Return a high latency to indicate failure

    return average_latency

# Initialization
population = [np.array([np.random.choice(range) for range in [
    P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, 
    PUCCH_TARGETSNRX10_RANGE, MAX_RXGAIN_RANGE, PRACH_DTX_THRESHOLD_RANGE, 
    NROF_UPLINK_SYMBOLS_RANGE, RSRP_THRESHOLD_SSB_RANGE, SESSION_AMBR_UL0_RANGE, 
    SESSION_AMBR_DL0_RANGE]]) for _ in range(POPULATION_SIZE)]

# Selection
def tournament_selection(population, fitnesses, tournament_size=3):

    # Randomly select tournament_size individuals from the population
    selected_indices = np.random.choice(range(len(population)), tournament_size, replace=False)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]

    # Find the index of the individual with the best fitness in the tournament
    winner_index = selected_indices[np.argmin(selected_fitnesses)]  # Use np.argmin because we assume lower fitness is better

    return winner_index

def select_parents(population, fitnesses, num_parents=2):
    parents_indices = []
    for _ in range(num_parents):
        parent_index = tournament_selection(population, fitnesses)
        parents_indices.append(parent_index)
    
    return parents_indices

# Crossover
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2

# Mutation
def mutate(chromosome):
    for i in range(len(chromosome)):
        if np.random.rand() < MUTATION_RATE:
            chromosome[i] = np.random.choice(eval(f"{chromosome[i]}_RANGE"))
    return chromosome
# Before the GA main loop
best_global_score = float('inf')  # High initial value since we are minimizing
best_global_position = None  # Placeholder for the best global chromosome

# GA main loop
for generation in range(NUM_GENERATIONS):
    fitnesses = [fitness_function(chromosome) for chromosome in population]

    # Find the best fitness in the current generation and its corresponding chromosome
    gen_best_index = np.argmin(fitnesses)
    gen_best_fitness = fitnesses[gen_best_index]
    gen_best_chromosome = population[gen_best_index]

    # Check if the best fitness of the current generation is better than the global best
    if gen_best_fitness < best_global_score:
        best_global_score = gen_best_fitness
        best_global_position = gen_best_chromosome

    # Log the best fitness of the current generation
    logging.info(f"Generation {generation + 1}/{NUM_GENERATIONS}: Best Fitness = {gen_best_fitness}")

    # Genetic operations: selection, crossover, and mutation to form a new population
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        parent_indices = select_parents(population, fitnesses)
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])
    population = new_population[:POPULATION_SIZE]

# Log the global best fitness and chromosome after completing all generations
logging.info(f"Optimization completed. Best Global Fitness: {best_global_score}, Best Global Chromosome: {best_global_position}")

