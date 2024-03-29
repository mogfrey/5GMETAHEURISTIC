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
                    handlers=[logging.FileHandler("pso_optimization.log"), logging.StreamHandler()])

# Parameters Ranges
P0_NOMINALWITHGRANT_RANGE = [-45, -60, -75, -90, -105, -120, -135, -150, -165, -180]
SSPBCH_BLOCKPOWER_RANGE = [0, -3, -6, -9, -12, -15, -18, -21, -24, -27, -30]
PUSCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))
PUCCH_TARGETSNRX10_RANGE = list(range(100, 310, 20))

# Population Size
NUM_PARTICLES = 10

# Max Iterations
MAX_ITERATIONS = 10

# PSO specific parameters
C1 = 1.5
C2 = 1.5
W = 0.5

client = docker.from_env()
BASE_DIR = os.path.expanduser("~/openairinterface5g/ci-scripts")
CONF_FILE = os.path.join(BASE_DIR, "conf_files/gnb.sa.band78.106prb.rfsim.conf")
def modify_config_file(particle):
    with open(CONF_FILE, 'r') as file:
        content = file.read()

    content = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {particle[0]}", content)
    content = re.sub(r"ssPBCH_BlockPower\s*=\s*[-0-9]+", f"ssPBCH_BlockPower = {particle[1]}", content)
    content = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {particle[2]}", content)
    content = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {particle[3]}", content)

    with open(CONF_FILE, 'w') as file:
        file.write(content)

def evaluate_particle(particle):
    logging.info(f"Evaluating particle with parameters: {particle}")
    modify_config_file(particle)

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
# Initialization for PSO
particles = np.array([[np.random.choice(param_range) for param_range in [
    P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE]] 
    for _ in range(NUM_PARTICLES)])
velocities = np.zeros_like(particles)

# Initialize personal best positions and global best positions
personal_best_positions = np.copy(particles)
personal_best_scores = np.array([float("-inf")] * NUM_PARTICLES)
global_best_position = particles[np.random.choice(particles.shape[0])]
global_best_score = float("-inf")

def plot_bandwidth_improvement(scores):
    """Plot the improvement in bandwidth over iterations."""
    iterations = list(range(1, len(scores) + 1))
    
    # Create the plot
    plt.plot(iterations, scores, '-o', markerfacecolor='red', markersize=8, linewidth=2)
    plt.title('Bandwidth Improvement Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Bandwidth (Mbits/sec)')
    plt.grid(True)
    plt.xticks(iterations)
    plt.tight_layout()
    
    # Save the plot as a file and show the plot
    plt.savefig('bandwidth_improvement.png')
    plt.show()

# PSO Optimization Loop
for iteration in range(MAX_ITERATIONS):
    for idx, particle in enumerate(particles):
        score = evaluate_particle(particle)
        if score > personal_best_scores[idx]:
            personal_best_scores[idx] = score
            personal_best_positions[idx] = particle

        if score > global_best_score:
            global_best_score = score
            global_best_position = particle

        # Update velocities and move particles
        inertia = W * velocities[idx]
        personal_attraction = C1 * np.random.random() * (personal_best_positions[idx] - particle)
        global_attraction = C2 * np.random.random() * (global_best_position - particle)
        
        velocities[idx] = inertia + personal_attraction + global_attraction
        particles[idx] += velocities[idx].astype(int)

        # Ensure particles are within bounds
        for i, param_range in enumerate([P0_NOMINALWITHGRANT_RANGE, SSPBCH_BLOCKPOWER_RANGE, PUSCH_TARGETSNRX10_RANGE, PUCCH_TARGETSNRX10_RANGE]):
            if particles[idx][i] < min(param_range):
                particles[idx][i] = min(param_range)
            elif particles[idx][i] > max(param_range):
                particles[idx][i] = max(param_range)
    
    logging.info(f"Iteration {iteration + 1}: Best Average TCP Bandwidth = {global_best_score} Mbits/sec with parameters = {global_best_position}")

logging.info(f"Optimal parameters after {MAX_ITERATIONS} iterations: {global_best_position}")
plot_bandwidth_improvement([global_best_score for _ in range(MAX_ITERATIONS)])  # We only have global best score for plotting here.
