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
                    handlers=[logging.FileHandler("particle_optimization_mMTC.log"), logging.StreamHandler()])


# PSO Parameters
NUM_PARTICLES = 5
MAX_ITERATIONS = 15
NO_IMPROVEMENT_THRESHOLD = 7
W = 0.5  # Inertia weight
C1 = 0.8  # Cognitive (particle) weight
C2 = 0.9  # Social (swarm) weight

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
# Initialize particles
particles = np.array([[np.random.choice(range_) for range_ in PARAMETER_RANGES] for _ in range(NUM_PARTICLES)])
velocities = np.zeros_like(particles)  # Initialize velocities

# Initialize personal bests
personal_bests = particles.copy()
personal_best_scores = np.full(NUM_PARTICLES, np.inf)


client = docker.from_env()
BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
GNB_CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
DOCKER_COMPOSE_FILE = os.path.join(BASE_DIR, "yaml_files/5g_rfsimulator/docker-compose.yaml")

def modify_config_file(particle, conf_file):
    # Modify the gNB config file
    if conf_file.endswith("rfsim.conf"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Update the parameters in the gNB configuration file
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

    # Modify the docker-compose file
    elif conf_file.endswith("docker-compose.yaml"):
        with open(conf_file, 'r') as file:
            content = file.read()

        # Updating SESSION_AMBR_UL0 and SESSION_AMBR_DL0 in the docker-compose file
        content = re.sub(r"SESSION_AMBR_UL0=\d+Mbps", f"SESSION_AMBR_UL0={particle[8]}Mbps", content)
        content = re.sub(r"SESSION_AMBR_DL0=\d+Mbps", f"SESSION_AMBR_DL0={particle[9]}Mbps", content)

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



def evaluate_particle(particle, num_ues):
    successful_ues = []
    modify_config_file(particle, GNB_CONF_FILE)
    modify_config_file(particle, DOCKER_COMPOSE_FILE)
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
    logging.info(f"Particle with parameters {particle} has an average jitter of {average_jitter} ms")
    return average_jitter

# PSO algorithm and other functions
# Initialize global best with the first particle's position
global_best = particles[0].copy()
global_best_score = np.inf 
no_improvement_count = 0
total_jitter = 0
def update_particle(particle, velocity, personal_best, global_best, parameter_ranges):
    r1, r2 = np.random.rand(), np.random.rand()
    new_velocity = W * velocity + C1 * r1 * (personal_best - particle) + C2 * r2 * (global_best - particle)
    new_position = particle + new_velocity

    # Ensure new_position is within bounds
    for i in range(len(new_position)):
        if new_position[i] < min(parameter_ranges[i]):
            new_position[i] = min(parameter_ranges[i])
        elif new_position[i] > max(parameter_ranges[i]):
            new_position[i] = max(parameter_ranges[i])

    return new_position, new_velocity



# Main PSO Loop
# Initialize a dictionary to store results for each UE count
results_by_ue_count = {}
MAX_UE_COUNT = 7  # Set the maximum number of UEs
num_ues = MAX_UE_COUNT 
for iteration in range(MAX_ITERATIONS):
        for i in range(NUM_PARTICLES):
        	# Update particle
        	particles[i], velocities[i] = update_particle(particles[i], velocities[i], personal_bests[i], global_best, PARAMETER_RANGES)

        	# Evaluate particle
        	score = evaluate_particle(particles[i], num_ues)

        	# Update personal best
        	if score < personal_best_scores[i]:
            		personal_bests[i] = particles[i].copy()
            		personal_best_scores[i] = score

        	# Update global best
        	if score < global_best_score:
            		global_best = particles[i].copy()
            		global_best_score = score
    	# Check for no improvement
        if global_best_score < min(personal_best_scores):
              no_improvement_count = 0
        else:
              no_improvement_count += 1

        if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
              logging.info(f"Stopping early at iteration {iteration + 1} due to no improvement.")
              break
    

	# After each PSO iteration loop
        logging.info(f"Completed PSO Iterations for {num_ues} UEs")
        average_jitter = np.mean([evaluate_particle(p, num_ues) for p in personal_bests])  # Average jitter across personal bests
        best_jitter = global_best_score  # Best jitter found in this iteration
        logging.info(f"Average Jitter for {num_ues} UEs: {average_jitter} ms")
        logging.info(f"Best Jitter for {num_ues} UEs: {best_jitter} ms")

	# Store results in the dictionary
        results_by_ue_count[num_ues] = {
                'average_jitter': average_jitter,
                'best_jitter': best_jitter,
                'best_parameters': global_best
        }

# At the end of all iterations, log the summary of results
logging.info("Summary of Jitter Optimization Results:")
for ue_count, result in results_by_ue_count.items():
    logging.info(f"UE Count: {ue_count}, Average Jitter: {result['average_jitter']} ms, Best Jitter: {result['best_jitter']} ms, Best Parameters: {result['best_parameters']}")
