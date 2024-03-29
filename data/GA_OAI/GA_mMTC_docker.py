import subprocess
import time
import numpy as np 
import re
import docker
import logging
import matplotlib.pyplot as plt
import os
import threading

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("individual_optimization_mMTC.log"), logging.StreamHandler()])

# GA Parameters
POPULATION_SIZE = 5
MAX_GENERATIONS = 15
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# Parameters Range
P0_NOMINALWITHGRANT_RANGE = [-45, -60, -75, -90, -105, -120, -135, -150, -165, -180]
SSPBCH_BLOCKPOWER_RANGE = [ -3, -6, -9, -12, -15, -18, -21, -24, -27, -30]
PUSCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
PUCCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
MAX_RXGAIN_RANGE = list(range(10, 126))  # 0 to 125
PRACH_DTX_THRESHOLD_RANGE = list(range(100, 181))  # 100 to 180
NROF_UPLINK_SYMBOLS_RANGE = list(range(1, 7))  # 0 to 6
RSRP_THRESHOLD_SSB_RANGE = list(range(1, 30))  # 1 to 29
SESSION_AMBR_UL0_RANGE = list(range(100, 500))  # 100 to 900
SESSION_AMBR_DL0_RANGE = list(range(100, 500))  # 100 to 900

PARAMETER_RANGES = [P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, 
                    PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE,
                    MAX_RXGAIN_RANGE, PRACH_DTX_THRESHOLD_RANGE, 
                    NROF_UPLINK_SYMBOLS_RANGE, RSRP_THRESHOLD_SSB_RANGE,
                    SESSION_AMBR_UL0_RANGE, SESSION_AMBR_DL0_RANGE]
					
client = docker.from_env()
BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")

def modify_config_file(individual, conf_file):
    # Modify the gNB config file
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Update the parameters in the gNB configuration file
        content = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {individual[0]}", content)
        content = re.sub(r"ssPBCH_BlockPower\s*=\s*[-0-9]+", f"ssPBCH_BlockPower = {individual[1]}", content)
        content = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {individual[2]}", content)
        content = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {individual[3]}", content)
        content = re.sub(r"max_rxgain\s*=\s*[0-9]+", f"max_rxgain = {individual[4]}", content)
        content = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {individual[5]}", content)
        content = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {individual[6]}", content)
        content = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {individual[7]}", content)

        with open(conf_file, 'w') as file:
            file.write(content)

    # Modify the docker-compose file
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Updating SESSION_AMBR_UL0 and SESSION_AMBR_DL0 in the docker-compose file
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={individual[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={individual[9]}Mbps", content)

        with open(conf_file, 'w') as file:
            file.write(content)
    #subprocess.run(["docker-compose", "down"])
    #subprocess.run(["docker-compose", "up","-d"])

def start_docker_containers(num_ues):
    # Wait for MySQL container
    logging.info("Waiting for MySQL container to start...")
    time.sleep(10)  # Delay for MySQL container

    # Start all UEs and setup iPerf servers
    for i in range(1, num_ues + 1):
        ue_service_name = f"oai-nr-ue{i}"
        logging.info(f"Deploying {ue_service_name}...")
        subprocess.run(["docker-compose", "up", "-d", ue_service_name])
        time.sleep(10)  # Short delay between starting UEs

        # Setup iPerf server on this UE
        ue_container = client.containers.get(f"rfsim5g-oai-nr-ue{i}")
        ue_container.exec_run("iperf -s -u", detach=True)
        logging.info(f"iPerf server setup completed on {ue_service_name}.")



def extract_packet_info(iperf_output):
    # Extract the number of lost and sent packets from the iPerf output
    match = re.search(r"(\d+)/\s+(\d+)\s+\((\d+)%\)", iperf_output)
    if match:
        lost_packets = int(match.group(1))
        sent_packets = int(match.group(2))
        return lost_packets, sent_packets
    return 0, 0

def extract_jitter(iperf_output):
    # Extract jitter from the iPerf output
    match = re.search(r"\s+(\d+\.\d+)\s+ms", iperf_output)
    if match:
        jitter = float(match.group(1))
        return jitter
    return 0.0


def run_iperf_test(container_name, ue_ip, results_list):
    try:
        iperf_command = f"iperf -c {ue_ip} -u -i 1 -t 20 -b 500K"
        result = client.containers.get(container_name).exec_run(iperf_command).output.decode()
        jitter = extract_jitter(result)
        results_list.append(jitter)
    except Exception as e:
        logging.error(f"Error during iPerf test for {container_name}: {e}")
        results_list.append(np.inf)
		



# Define your fitness function here
def fitness_function(individual):
    # Evaluate the individual and return fitness
    successful_ues = []
    modify_config_file(individual, GNB_CONF_FILE)
    modify_config_file(individual, DOCKER_COMPOSE_FILE)
    # Start all containers
    subprocess.run(["docker-compose", "down"])
    subprocess.run(["docker-compose", "up", "-d", "mysql", "oai-nrf", "oai-amf", "oai-smf", "oai-spgwu", "oai-ext-dn"])
    time.sleep(10)  # Delay for MySQL
    subprocess.run(["docker-compose", "up", "-d", "oai-gnb"])
    time.sleep(5)  # Delay for gNB
    # Run iPerf tests concurrently
    threads = []
    jitter_results = []

    start_docker_containers(num_ues)
    total_lost_packets, total_sent_packets = 0, 0
    penalty_score = 1000  # Define a high penalty score

    for ue in range(1, num_ues + 1):
        container_name = f"rfsim5g-oai-nr-ue{ue}"
        cmd_result = client.containers.get(container_name).exec_run(["ip", "a", "show", "oaitun_ue1"])
        ip_search = re.search(r"inet ([\d\.]+)", cmd_result.output.decode())

        if ip_search:
            ue_ip = ip_search.group(1)
            successful_ues.append(ue)
            thread = threading.Thread(target=run_iperf_test, args=(container_name, ue_ip, jitter_results))
            threads.append(thread)
            thread.start()
        else:
            logging.error(f"IP address not allocated for UE {ue}.")
            


    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate Average Jitter
    average_jitter = sum(jitter_results) / len(successful_ues) if successful_ues else np.inf
    logging.info(f"individual with parameters {individual} has an average jitter of {average_jitter} ms")
    return average_jitter
	
def selection(population, fitness_scores):
    selected_individuals = []
    total_fitness = sum(fitness_scores)
    relative_fitness = [f / total_fitness for f in fitness_scores]
    probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    for _ in range(len(population)):
        r = np.random.random()
        for (individual, probability) in zip(population, probabilities):
            if r <= probability:
                selected_individuals.append(individual)
                break
    return np.array(selected_individuals)

def crossover(selected_individuals, crossover_rate):
    offspring = []
    for i in range(0, len(selected_individuals), 2):
        parent1 = selected_individuals[i]
        if i+1 < len(selected_individuals):
            parent2 = selected_individuals[i+1]
        else:
            parent2 = selected_individuals[0]
        if np.random.random() < crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    return np.array(offspring)
	
def mutate(offspring, mutation_rate):
    mutated_offspring = []
    for individual in offspring:
        if np.random.random() < mutation_rate:
            mutation_point = np.random.randint(len(individual))
            individual[mutation_point] = np.random.choice(PARAMETER_RANGES[mutation_point])
        mutated_offspring.append(individual)
    return np.array(mutated_offspring)


# Initialize population
population = np.array([[np.random.choice(range_) for range_ in PARAMETER_RANGES] for _ in range(POPULATION_SIZE)])

def replacement_strategy(population, offspring):
    # Replace the old population completely with the offspring
    return offspring
	
# Initialize best tracking variables
best_individual = None
best_fitness = np.inf  # Start with a very high value since we are minimizing
	
# Main GA Loop
MAX_UE_COUNT = 7  # Set the maximum number of UEs
num_ues = MAX_UE_COUNT 
for generation in range(MAX_GENERATIONS):
    # Evaluate fitness
    fitness_scores = [fitness_function(ind) for ind in population]
	
	# Find best fitness score in this generation
    gen_best_fitness = min(fitness_scores)
    gen_best_individual = population[np.argmin(fitness_scores)]

    # Update overall best individual
    if gen_best_fitness < best_fitness:
        best_fitness = gen_best_fitness
        best_individual = gen_best_individual

    # Selection
    selected_individuals = selection(population, fitness_scores)

    # Crossover
    offspring = crossover(selected_individuals, CROSSOVER_RATE)

    # Mutation
    offspring = mutate(offspring, MUTATION_RATE)

    # Replacement strategy to form new population
    population = replacement_strategy(population, offspring)

    # Logging progress
    logging.info(f"Generation {generation + 1} completed. Best fitness: {gen_best_fitness}")

# At the end of all generations, log the best individual and its fitness
logging.info("GA Optimization Complete")
logging.info(f"Best Individual: {best_individual}")
logging.info(f"Best Fitness: {best_fitness}")  # Note: Lower fitness means lower jitter, which is better
