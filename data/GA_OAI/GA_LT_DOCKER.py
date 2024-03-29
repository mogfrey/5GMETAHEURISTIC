import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import docker
import time
import re

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("GENLT_optimization.log"), logging.StreamHandler()])
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
MAX_GENERATIONS = 15
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

# Initialize population
population = np.array([np.array([
    np.random.choice(P0_NOMINALWITHGRANT_RANGE),
    np.random.choice(SSPBCH_BLOCKPOWER_RANGE),
    np.random.choice(PUSCH_TARGETSNRX10_RANGE),
    np.random.choice(PUCCH_TARGETSNRX10_RANGE),
    np.random.choice(MAX_RXGAIN_RANGE),
    np.random.choice(PRACH_DTX_THRESHOLD_RANGE),
    np.random.choice(NROF_UPLINK_SYMBOLS_RANGE),
    np.random.choice(RSRP_THRESHOLD_SSB_RANGE),
    np.random.choice(SESSION_AMBR_UL0_RANGE),
    np.random.choice(SESSION_AMBR_DL0_RANGE)
]) for _ in range(POPULATION_SIZE)])


# Best scores history for plotting
best_scores_history = []


client = docker.from_env()

BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")

def modify_config_file(individual, conf_file):
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()
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
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={individual[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={individual[9]}Mbps", content)
        with open(conf_file, 'w') as file:
            file.write(content)
			

# Fitness evaluation function
def evaluate_individual(individual):
    logging.info(f"Evaluating individual with parameters: {individual}")
    modify_config_file(individual, GNB_CONF_FILE)
    modify_config_file(individual, DOCKER_COMPOSE_FILE)
    
    for container_name in ["rfsim5g-oai-gnb", "rfsim5g-oai-nr-ue"]:
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            print(f"Container {container_name} not found!")
        except docker.errors.APIError as e:
            print(f"Error removing container {container_name}: {e}")

    os.system("docker-compose down")
    os.system("docker-compose up -d oai-gnb")
    time.sleep(15)
    os.system("docker-compose up -d oai-nr-ue")
    time.sleep(15)
    
    container = client.containers.get("rfsim5g-oai-nr-ue")
    cmd_result = container.exec_run(["ip", "a", "show", "oaitun_ue1"])
    ip_search = re.search(r"inet ([\d\.]+)", cmd_result.output.decode())
    score = 0
    avg_rtt= 0
    if ip_search:
        bind_ip = ip_search.group(1)
        logging.info(f"Using IP: {bind_ip}")
        logging.info("Setup and execute the ping command")
        container = client.containers.get("rfsim5g-oai-nr-ue")
        ping_output = container.exec_run("ping -c 60 192.168.71.134")
        ping_result = ping_output.output.decode()
        logging.info(f"Ping output: {ping_result}")
        try:
            if "avg" in ping_result:
                avg_rtt_match = re.search(r"rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/[\d\.]+/[\d\.]+ ms", ping_result)
                if avg_rtt_match:
                    avg_rtt = float(avg_rtt_match.group(1))
                    score = float(avg_rtt_match.group(1))
        except Exception as e:
            logging.error(f"Error processing result: {e}")
            score = 0
    else:
        logging.info("IP address not allocated. Recording bandwidth as 0.")
        score = np.inf
    
    logging.info(f"individual with parameters {individual} has an average RTT of {avg_rtt} ms")
    return score

# Selection function: Tournament selection
def tournament_selection(population, scores, k=TOURNAMENT_SIZE):
    # Select k indices randomly
    indices = np.random.choice(len(population), k, replace=False)
    
    # Find the individual with the best score among selected indices
    best_idx = indices[0]
    best_score = scores[best_idx]
    for idx in indices[1:]:
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_idx = idx

    return population[best_idx]

# Crossover function
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2
	
# Define parameter ranges
param_ranges = [
    P0_NOMINALWITHGRANT_RANGE, 
    SSPBCH_BLOCKPOWER_RANGE, 
    PUSCH_TARGETSNRX10_RANGE, 
    PUCCH_TARGETSNRX10_RANGE, 
    MAX_RXGAIN_RANGE, 
    PRACH_DTX_THRESHOLD_RANGE, 
    NROF_UPLINK_SYMBOLS_RANGE, 
    RSRP_THRESHOLD_SSB_RANGE, 
    SESSION_AMBR_UL0_RANGE, 
    SESSION_AMBR_DL0_RANGE
]

# Mutation function
def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.choice(param_ranges[i])
    return individual

# Initialize best score and best individual
best_individual = None
best_individual_score = np.inf


def plot_latency_improvement(best_scores_history):
    plt.figure(figsize=(10,6))
    plt.plot(best_scores_history, 'o-', linewidth=2)
    plt.title("Best Average RTT (Latency) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average RTT (ms)")
    plt.grid(True)
    plt.show()

# Main GA loop
best_scores_history = []
for generation in range(MAX_GENERATIONS):
    scores = np.array([evaluate_individual(ind) for ind in population])
	
	# Update best individual and best score
    for ind, score in zip(population, scores):
        if score < best_individual_score:
            best_individual_score = score
            best_individual = ind.copy()

    # Selection and reproduction
    new_population = []
    for _ in range(POPULATION_SIZE // 2):
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([child1, child2])

    # Mutation
    new_population = [mutate(ind) for ind in new_population]

    # Replace old population
    population = np.array(new_population)

    # Logging and history tracking
    best_scores_history.append(best_individual_score)
    logging.info(f"Generation {generation + 1}: Best Score = {best_individual_score}")


    # Plotting
    plot_latency_improvement(best_scores_history)

# Final results
logging.info(f"Optimal parameters after {MAX_GENERATIONS} generations: {best_individual}")
plot_latency_improvement(best_scores_history)
