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
SSPBCH_BLOCKPOWER_RANGE = [-3, -6, -9, -12, -15, -18, -21, -24, -27, -30]
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
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")

def modify_config_file(whale, conf_file):
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()
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
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={whale[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={whale[9]}Mbps", content)
        with open(conf_file, 'w') as file:
            file.write(content)

def evaluate_whale(whale):
    logging.info(f"Evaluating whale with parameters: {whale}")
    modify_config_file(whale, GNB_CONF_FILE)
    modify_config_file(whale, DOCKER_COMPOSE_FILE)
    
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
        score = 0
    
    logging.info(f"Whale with parameters {whale} has an average RTT of {avg_rtt} ms")
    return score

scores = [evaluate_whale(whale) for whale in whales]
best_idx = np.argmin(scores)
best_whale = whales[best_idx].copy()
best_scores_history = []

prev_whale_positions = np.copy(whales)
prev_scores = np.array(scores)

def plot_latency_improvement(best_scores_history):
    plt.figure(figsize=(10,6))
    plt.plot(best_scores_history, 'o-', linewidth=2)
    plt.title("Best Average RTT (Latency) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average RTT (ms)")
    plt.grid(True)
    plt.show()

a = 2
a_decrease = 2.0 / MAX_ITERATIONS

for iteration in range(MAX_ITERATIONS):
    a -= a_decrease
    for idx, whale in enumerate(whales):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        b = 1
        l = np.random.uniform(-1, 1)

        p = np.random.uniform(0, 1)
        if p < 0.5:
            if abs(A) < 1:
                D = abs(C * best_whale - whale)
                whales[idx] = best_whale - A * D
            else:
                random_whale = whales[np.random.randint(NUM_WHALES)]
                D = abs(C * random_whale - whale)
                whales[idx] = random_whale - A * D
        else:
            distance_to_best = abs(best_whale - whale)
            whales[idx] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
        for i, param_range in enumerate([
            P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, 
            PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE,
            MAX_RXGAIN_RANGE, PRACH_DTX_THRESHOLD_RANGE, 
            NROF_UPLINK_SYMBOLS_RANGE, RSRP_THRESHOLD_SSB_RANGE,
            SESSION_AMBR_UL0_RANGE, SESSION_AMBR_DL0_RANGE
        ]):
            if whales[idx][i] < min(param_range):
                whales[idx][i] = min(param_range)
            elif whales[idx][i] > max(param_range):
                whales[idx][i] = max(param_range)

    scores = []
    for idx, whale in enumerate(whales):
        if np.array_equal(whale, prev_whale_positions[idx]):
            scores.append(prev_scores[idx])
        else:
            scores.append(evaluate_whale(whale))
    
    prev_whale_positions = np.copy(whales)
    prev_scores = np.array(scores)
    
    current_best_idx = np.argmin(scores)
    if scores[current_best_idx] < scores[best_idx]:
        best_idx = current_best_idx
        best_whale = whales[best_idx].copy()
    
    if iteration == 0 or scores[best_idx] < best_scores_history[-1]:
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        
    logging.info(f"Iteration {iteration + 1}: Best Average RTT = {scores[best_idx]} ms with parameters = {best_whale}")
    best_scores_history.append(scores[best_idx])
    if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
        logging.info(f"Breaking out of optimization loop after {iteration + 1} iterations due to no improvement in the best score for {NO_IMPROVEMENT_THRESHOLD} iterations.")
        break

logging.info(f"Optimal parameters after {MAX_ITERATIONS} iterations: {best_whale}")
plot_latency_improvement(best_scores_history)
