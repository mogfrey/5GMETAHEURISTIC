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
                    handlers=[logging.FileHandler("PSOLT_optimization.log"), logging.StreamHandler()])

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

# PSO parameters
NUM_PARTICLES = 5
MAX_ITERATIONS = 15
INERTIA_WEIGHT = 0.5
COGNITIVE_COEFF = 1.4
SOCIAL_COEFF = 1.4

# Particle initialization
particles = np.array([{
    'position': np.array([
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
    ]),
    'velocity': np.random.rand(10) - 0.5,
    'best_position': None,
    'best_score': np.inf
} for _ in range(NUM_PARTICLES)])

# Initialize global best
global_best_score = np.inf
global_best_position = None



# Best scores history for plotting
best_scores_history = []


client = docker.from_env()

BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")

def modify_config_file(particle, conf_file):
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()
        content = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {particle[0]}", content)
        content = re.sub(r"ssPBCH_BlockPower\s*=\s*[-0-9]+", f"ssPBCH_BlockPower = {particle[1]}", content)
        content = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {particle[2]}", content)
        content = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {particle[3]}", content)
        content = re.sub(r"max_rxgain\s*=\s*[0-9]+", f"max_rxgain = {particle[4]}", content)
        content = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {particle[5]}", content)
        content = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {particle[6]}", content)
        content = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {particle[7]}", content)
        with open(conf_file, 'w') as file:
            file.write(content)
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={particle[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={particle[9]}Mbps", content)
        with open(conf_file, 'w') as file:
            file.write(content)
			
def evaluate_particle(particle):
    logging.info(f"Evaluating particle with parameters: {particle}")
    modify_config_file(particle, GNB_CONF_FILE)
    modify_config_file(particle, DOCKER_COMPOSE_FILE)
    
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
    
    logging.info(f"particle with parameters {particle} has an average RTT of {avg_rtt} ms")
    return score

# Initial evaluation and setting initial best positions
for particle in particles:
    particle_score = evaluate_particle(particle['position'])
    particle['best_score'] = particle_score
    particle['best_position'] = particle['position'].copy()
    if particle_score < global_best_score:
        global_best_score = particle_score
        global_best_position = particle['position'].copy()

# Ensure position stays within bounds
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

# Initialize no improvement counter
no_improvement_count = 0
NO_IMPROVEMENT_THRESHOLD = 7 

def plot_latency_improvement(best_scores_history):
    plt.figure(figsize=(10,6))
    plt.plot(best_scores_history, 'o-', linewidth=2)
    plt.title("Best Average RTT (Latency) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average RTT (ms)")
    plt.grid(True)
    plt.show()

# PSO algorithm
for iteration in range(MAX_ITERATIONS):
    global_best_position_updated = False  # Initialize the flag for each iteration
    
    for particle in particles:
        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()
        particle['velocity'] = INERTIA_WEIGHT * particle['velocity'] \
            + COGNITIVE_COEFF * r1 * (particle['best_position'] - particle['position']) \
            + SOCIAL_COEFF * r2 * (global_best_position - particle['position'])
        
        # Update position and ensure it stays within bounds
        # Update position
        new_position = particle['position'] + particle['velocity']
        # Round the new position to the nearest integer and ensure it stays within bounds
        for i, param_range in enumerate(param_ranges):
            rounded_position = round(new_position[i])
            if rounded_position < min(param_range):
                particle['position'][i] = min(param_range)
            elif rounded_position > max(param_range):
                particle['position'][i] = max(param_range)
            else:
                particle['position'][i] = rounded_position

        # Evaluate particle
        particle_score = evaluate_particle(particle['position'])
        if particle_score < particle['best_score']:
            particle['best_score'] = particle_score
            particle['best_position'] = particle['position'].copy()
        
        # Update global best
        if particle_score < global_best_score:
            global_best_score = particle_score
            global_best_position = particle['position'].copy()
            global_best_position_updated = True

    if global_best_position_updated:
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    best_scores_history.append(global_best_score)
    logging.info(f"Iteration {iteration + 1}: Best Average RTT = {global_best_score} ms with parameters = {global_best_position}")

    if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
        logging.info(f"Breaking out of optimization loop after {iteration + 1} iterations due to no improvement in the best score for {NO_IMPROVEMENT_THRESHOLD} iterations.")
        break

# Final results
logging.info(f"Optimal parameters after {MAX_ITERATIONS} iterations: {global_best_position}")
plot_latency_improvement(best_scores_history)

