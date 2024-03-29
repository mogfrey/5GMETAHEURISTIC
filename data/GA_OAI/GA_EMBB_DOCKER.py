import numpy as np
import subprocess
import re
import docker
import time
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("genetic_optimization.log"), logging.StreamHandler()])

# Parameters Ranges
P0_NOMINALWITHGRANT_RANGE = [-45, -60, -75, -90, -105, -120, -135, -150, -165, -180]
SSPBCH_BLOCKPOWER_RANGE = [0, -3, -6, -9, -12, -15, -18, -21, -24, -27, -30]
PUSCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
PUCCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))

# Population Size
POPULATION_SIZE = 10

# Number of Generations
MAX_GENERATIONS = 10

# Crossover probability
P_CROSSOVER = 0.7

# Mutation probability
P_MUTATION = 0.3

# Initialization for GA
population = np.array([[np.random.choice(param_range) for param_range in [
    P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE]] 
    for _ in range(POPULATION_SIZE)])
    
def select_parents(fitness_scores):
    """Selects parents based on tournament selection."""
    selected_indices = np.argsort(fitness_scores)[-2:]
    return population[selected_indices]

def crossover(parents):
    """Single-point crossover."""
    if np.random.random() < P_CROSSOVER:
        crossover_point = np.random.randint(1, len(parents[0]))
        offspring1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        offspring2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
        return offspring1, offspring2
    return parents[0], parents[1]

def mutate(offspring):
    """Apply mutation to the offspring."""
    for i, param_range in enumerate([P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE]):
        if np.random.random() < P_MUTATION:
            offspring[i] = np.random.choice(param_range)
    return offspring

client = docker.from_env()
BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
def modify_config_file(chromosome):
    with open(CONF_FILE, 'r') as file:
        content = file.read()

    content = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {chromosome[0]}", content)
    content = re.sub(r"ssPBCH_BlockPower\s*=\s*[-0-9]+", f"ssPBCH_BlockPower = {chromosome[1]}", content)
    content = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {chromosome[2]}", content)
    content = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {chromosome[3]}", content)

    with open(CONF_FILE, 'w') as file:
        file.write(content)


def evaluate_chromosome(chromosome):
    logging.info(f"Evaluating chromosome with parameters: {chromosome}")
    modify_config_file(chromosome)

    # Remove containers
    for container_name in ["rfsim5g-oai-gnb", "rfsim5g-oai-nr-ue"]:
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            print(f"Container {container_name} not found!")
        except docker.errors.APIError as e:
            print(f"Error removing container {container_name}: {e}")

    os.system("docker-compose up -d oai-gnb")
    time.sleep(15)
    os.system("docker-compose up -d oai-nr-ue")
    time.sleep(15)

    # Get the IP of oaitun_ue1
    container = client.containers.get("rfsim5g-oai-nr-ue")
    cmd_result = container.exec_run(["ip", "a", "show", "oaitun_ue1"])
    bind_ip = re.search(r"inet ([\d\.]+)", cmd_result.output.decode()).group(1)
    logging.info(f"Using IP: {bind_ip}")

    logging.info(f"TCP Test using subprocess")
    cmd_tcp_server = f"docker exec -d rfsim5g-oai-nr-ue iperf -B {bind_ip} -i 1 -s"
    subprocess.run(cmd_tcp_server, shell=True)

    time.sleep(5)
    logging.info(f"Setting up iperf client")
    cmd_tcp_client = f"docker exec -it rfsim5g-oai-ext-dn iperf -c {bind_ip} -i 1 -t 60"
    result_tcp_client = subprocess.run(cmd_tcp_client, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    iperf_output_tcp = result_tcp_client.stdout.decode()

    try:
        avg_bandwidth_tcp = re.search(r"[0-9]+(\.[0-9]+)? Mbits/sec", iperf_output_tcp).group(0)
        return float(avg_bandwidth_tcp.split()[0])
    except Exception as e:
        logging.error(f"Error processing result: {iperf_output_tcp}. Error: {e}")
        return 0

best_scores = []
global_best_chromosome = None
global_best_score = float("-inf")

# GA Optimization Loop
for generation in range(MAX_GENERATIONS):
    fitness_scores = np.array([evaluate_chromosome(chromosome) for chromosome in population])
    
    # Get global best
    generation_best_score = np.max(fitness_scores)
    if generation_best_score > global_best_score:
        global_best_score = generation_best_score
        global_best_chromosome = population[np.argmax(fitness_scores)]
    best_scores.append(generation_best_score)

    # Generate new population
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        parents = select_parents(fitness_scores)
        offspring1, offspring2 = crossover(parents)
        new_population.append(mutate(offspring1))
        for i, param_range in enumerate([P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE]):
            if new_population[-1][i] < min(param_range):
                new_population[-1][i] = min(param_range)
            elif new_population[-1][i] > max(param_range):
                new_population[-1][i] = max(param_range)

        new_population.append(mutate(offspring2))

    # Replace old population
    population = np.array(new_population)[:POPULATION_SIZE]
    
    logging.info(f"Generation {generation + 1}: Best Average TCP Bandwidth = {generation_best_score} Mbits/sec with parameters = {global_best_chromosome}")

logging.info(f"Optimal parameters after {MAX_GENERATIONS} generations: {global_best_chromosome}")
plot_bandwidth_improvement(best_scores)  # Updated to plot best score of each generation
