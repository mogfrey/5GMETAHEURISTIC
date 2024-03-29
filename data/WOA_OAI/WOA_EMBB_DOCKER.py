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
                    handlers=[logging.FileHandler("whale_optimization.log"), logging.StreamHandler()])

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
# Population Size
NUM_WHALES = 5

# Max Iterations
MAX_ITERATIONS = 15

no_improvement_count = 0
NO_IMPROVEMENT_THRESHOLD = 7



# Initialization
whales = np.array([[
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
] for _ in range(NUM_WHALES)])

client = docker.from_env()

BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
# Constants for file paths
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")



def modify_config_file(whale, conf_file):
    # Modify the gNB config file
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Update the parameters in the gNB configuration file
        content = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {whale[0]}", content)
        content = re.sub(r"ssPBCH_BlockPower\s*=\s*[-0-9]+", f"ssPBCH_BlockPower = {whale[1]}", content)
        content = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {whale[2]}", content)
        content = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {whale[3]}", content)
        content = re.sub(r"max_rxgain\s*=\s*[0-9]+", f"max_rxgain = {whale[4]}", content)
        content = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {whale[5]}", content)
        content = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {whale[6]}", content)
        content = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {whale[7]}", content)

        with open(conf_file, 'w') as file:
            file.write(content)

    # Modify the docker-compose file
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Updating SESSION_AMBR_UL0 and SESSION_AMBR_DL0 in the docker-compose file
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={whale[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={whale[9]}Mbps", content)

        with open(conf_file, 'w') as file:
            file.write(content)
 
def evaluate_whale(whale):
    logging.info(f"Evaluating whale with parameters: {whale}")
    # Modify both configuration files
    modify_config_file(whale, GNB_CONF_FILE)
    modify_config_file(whale, DOCKER_COMPOSE_FILE)
    avg_bandwidth_tcp = 0
    # Remove containers
    for container_name in ["rfsim5g-oai-gnb", "rfsim5g-oai-nr-ue"]:
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            print(f"Container {container_name} not found!")
        except docker.errors.APIError as e:
            print(f"Error removing container {container_name}: {e}")

    # Using docker-compose
    #os.chdir(os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator"))
    os.system("docker-compose down")
    os.system("docker-compose up -d mysql")
    time.sleep(60)
    os.system("docker-compose up -d oai-gnb")
    time.sleep(15)
    os.system("docker-compose up -d oai-nr-ue")
    time.sleep(15)
   
    # Get the IP of oaitun_ue1
    container = client.containers.get("rfsim5g-oai-nr-ue")
    cmd_result = container.exec_run(["ip", "a", "show", "oaitun_ue1"])
    ip_search = re.search(r"inet ([\d\.]+)", cmd_result.output.decode())
    score = 0  # Initialize score to 0w    
    if ip_search:
        bind_ip = ip_search.group(1)
        logging.info(f"Using IP: {bind_ip}")
        
        # TCP Test using subprocess
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
            score= float(avg_bandwidth_tcp.split()[0])
        
        except Exception as e:
            logging.error(f"Error processing result: {iperf_output_tcp}. Error: {e}")
            score= 0
    
    else:
        logging.info("IP address not allocated. Recording bandwidth as 0.")
        score = 0


    
    
    logging.info(f"Whale with parameters {whale} has a score of {avg_bandwidth_tcp} Mbits/sec")
    return score
	
	

# Initialize best_whale
scores = [evaluate_whale(whale) for whale in whales]
best_idx = np.argmax(scores)
best_whale = whales[best_idx].copy()
best_scores_history = []

# Store previous positions and scores
prev_whale_positions = np.copy(whales)
prev_scores = np.array(scores)

def plot_bandwidth_improvement(best_scores_history):
    plt.figure(figsize=(10,6))
    plt.plot(best_scores_history, 'o-', linewidth=2)
    plt.title("Best Average TCP Bandwidth over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average TCP Bandwidth (Mbits/sec)")
    plt.grid(True)
    plt.show()


# Optimization
a = 2  # Start value for 'a'
a_decrease = 2.0 / MAX_ITERATIONS

for iteration in range(MAX_ITERATIONS):
    a -= a_decrease
    for idx, whale in enumerate(whales):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        b = 1  # Defines the shape of the spiral (usually set to 1)
        l = np.random.uniform(-1, 1)

        p = np.random.uniform(0, 1)
        if p < 0.5:
            if abs(A) < 1:
                # Updating whale position using Encircling prey mechanism
                D = abs(C * best_whale - whale)
                whales[idx] = best_whale - A * D
            else:
                # Exploration phase
                random_whale = whales[np.random.randint(NUM_WHALES)]
                D = abs(C * random_whale - whale)
                whales[idx] = random_whale - A * D
        else:
            # Spiral model (Exploitation phase)
            distance_to_best = abs(best_whale - whale)
            whales[idx] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
        # Ensure that the parameters are within bounds
        for i, param_range in enumerate([
        P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, 
        PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE,
        MAX_RXGAIN_RANGE, PRACH_DTX_THRESHOLD_RANGE, 
        NROF_UPLINK_SYMBOLS_RANGE, RSRP_THRESHOLD_SSB_RANGE,
        SESSION_AMBR_UL0_RANGE, SESSION_AMBR_DL0_RANGE
    ]):
            if whales[idx][i] < min(param_range):  # if below the lower bound
               whales[idx][i] = min(param_range)
            elif whales[idx][i] > max(param_range):  # if above the upper bound
               whales[idx][i] = max(param_range)
        
    # Only evaluate whales that have moved
    scores = []
    for idx, whale in enumerate(whales):
        if np.array_equal(whale, prev_whale_positions[idx]):
            scores.append(prev_scores[idx])
        else:
            scores.append(evaluate_whale(whale))
            
    # Store current positions and scores for next iteration
    prev_whale_positions = np.copy(whales)
    prev_scores = np.array(scores)
    
    # Update the best whale if a better one is found
    current_best_idx = np.argmax(scores)
    if scores[current_best_idx] > scores[best_idx]:
        best_idx = current_best_idx
        best_whale = whales[best_idx].copy()
    
    # Check for improvement in best score
    if iteration == 0 or scores[best_idx] > best_scores_history[-1]:
        no_improvement_count = 0
    else:
        no_improvement_count += 1        
        
    logging.info(f"Iteration {iteration + 1}: Best Average TCP Bandwidth = {scores[best_idx]} Mbits/sec with parameters = {best_whale}")

    best_scores_history.append(scores[best_idx])
    if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
        logging.info(f"Breaking out of optimization loop after {iteration + 1} iterations due to no improvement in the best score for {NO_IMPROVEMENT_THRESHOLD} iterations.")
        break

# At the end of the iterations, best_whale contains the optimal set of parameters
logging.info(f"Optimal parameters after {MAX_ITERATIONS} iterations: {best_whale}")
plot_bandwidth_improvement(best_scores_history)
