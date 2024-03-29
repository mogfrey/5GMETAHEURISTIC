import numpy as np
import matplotlib.pyplot as plt
import logging

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


# Swarm size and other PSO parameters
NUM_PARTICLES = 5
MAX_ITERATIONS = 30
W = 0.5  # Inertia weight
C1 = 0.8  # Cognitive parameter
C2 = 0.9  # Social parameter

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-abs(high-low), abs(high-low)) for low, high in bounds])
        self.best_position = self.position.copy()
        self.best_score = -np.inf

    def update_velocity(self, global_best_position, W, C1, C2):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_velocity = C1 * r1 * (self.best_position - self.position)
        social_velocity = C2 * r2 * (global_best_position - self.position)
        self.velocity = W * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)


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

# Initialize the swarm
particles = [Particle(position_bounds) for _ in range(NUM_PARTICLES)]

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


def modify_config_map(particle):
    # Load the existing core.yaml for SMF configuration
    with open('core.yaml', 'r') as file:
        core_config = yaml.safe_load(file)
    
    # Modify session AMBR values in the SMF configuration
    for info in core_config['data']['config.yaml']['smf']['local_subscription_infos']:
        if info['single_nssai']['sst'] == 1:  # Assuming SST 1 is the target slice
            info['qos_profile']['session_ambr_ul'] = f"{particle[8]}Mbps"
            info['qos_profile']['session_ambr_dl'] = f"{particle[9]}Mbps"
    
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
    # Replace 'whale' with 'particle' in the configuration modifications
    gnb_conf_str = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {particle[0]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {particle[2]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {particle[3]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {particle[5]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {particle[6]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {particle[7]}", gnb_conf_str)

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
        pod_names = subprocess.check_output(get_pod_cmd, shell=True).decode().strip().split('\n')
        
        # Delete each pod found with the prefix
        for pod_name in pod_names:
            if pod_name:  # Ensure the pod name is not empty
                delete_pod_cmd = f"kubectl delete pod {pod_name}"
                
                subprocess.run(delete_pod_cmd, shell=True)
                
    
    
    time.sleep(60)  # Wait for 1 minute for pods to restart. Adjust this value as necessary for your setup.

def evaluate_particle(particle):
    

    # Assuming modify_config_map and restart_pods are implemented elsewhere
    # Modify system configuration based on particle's position
    modify_config_map(particle.position)
    restart_pods()
    time.sleep(180)  # Wait for 3 minutes after restarting pods to ensure the network is stable

    # Identify the UE pod
    ue_pod_cmd = "kubectl get pods --no-headers | grep '^oai-nr-ue' | awk '{print $1}'"
    ue_pod_name = subprocess.check_output(ue_pod_cmd, shell=True).decode().strip()

    # Identify the UPF pod
    upf_pod_cmd = "kubectl get pods --no-headers | grep '^oai-upf' | awk '{print $1}'"
    upf_pod_name = subprocess.check_output(upf_pod_cmd, shell=True).decode().strip()

    if not ue_pod_name or not upf_pod_name:
        logging.error("Could not find UE or UPF pod. Aborting evaluation.")
        return 0

    # Start iperf server on UE pod
    iperf_server_cmd = f"kubectl exec -it {ue_pod_name} -- iperf -s -D"
    subprocess.run(iperf_server_cmd, shell=True)

    time.sleep(10)  # Wait a bit for the server to start

    # Execute iperf client on UPF pod to measure the throughput
    iperf_client_cmd = f"kubectl exec -it {upf_pod_name} -- iperf -c oaitun_ue1 -t 30"
    iperf_result = subprocess.run(iperf_client_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    iperf_output = iperf_result.stdout.decode()

    # Parse the output to find the bandwidth
    bandwidth_match = re.search(r"([0-9\.]+ Mbits/sec)", iperf_output)
    if bandwidth_match:
        bandwidth = bandwidth_match.group(1)
        performance_score = float(bandwidth.split()[0])  # Extract the numeric part
        
    else:
        logging.error("Could not parse bandwidth from iperf output. Setting performance score to 0.")
        performance_score = 0

    return performance_score

# PSO main loop
best_global_score = -np.inf
best_global_position = None

for iteration in range(MAX_ITERATIONS):
    for i, particle in enumerate(particles):
        # Evaluate the fitness of each particle
        score = evaluate_particle(particle.position)

        # Update personal best
        if score > particle.best_score:
            particle.best_score = score
            particle.best_position = particle.position.copy()

        # Update global best
        if score > best_global_score:
            best_global_score = score
            best_global_position = particle.position.copy()

    for particle in particles:
        # Update velocity
        particle.velocity = (W * particle.velocity +
                             C1 * np.random.rand() * (particle.best_position - particle.position) +
                             C2 * np.random.rand() * (best_global_position - particle.position))

        # Update position
        particle.position += particle.velocity

    logging.info(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Best Score = {best_global_score}")
      
# Final output and plotting code 
logging.info(f"Optimization completed. Best Score: {best_global_score}, Best Position: {best_global_position}")
# Main execution flow remains the same
deploy_osm_network_slice()
