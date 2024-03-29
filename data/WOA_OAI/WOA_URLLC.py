import numpy as np
import re
import time
import os
import yaml
import subprocess
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("whale_optimization_k8s.log"), logging.StreamHandler()])

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
MAX_ITERATIONS = 30

# Initialization of whales
# This initializes the whale population with random parameters chosen from the above ranges
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
# Defining minimum bounds for each parameter
param_min_bounds = [
    min(P0_NOMINALWITHGRANT_RANGE),  
    min(SSPBCH_BLOCKPOWER_RANGE),    
    min(PUSCH_TARGETSNRX10_RANGE),   
    min(PUCCH_TARGETSNRX10_RANGE),   
    min(MAX_RXGAIN_RANGE),           
    min(PRACH_DTX_THRESHOLD_RANGE),  
    min(NROF_UPLINK_SYMBOLS_RANGE),  
    min(RSRP_THRESHOLD_SSB_RANGE),   
    min(SESSION_AMBR_UL0_RANGE),     
    min(SESSION_AMBR_DL0_RANGE)      
]

# Defining maximum bounds for each parameter
param_max_bounds = [
    max(P0_NOMINALWITHGRANT_RANGE),  
    max(SSPBCH_BLOCKPOWER_RANGE),    
    max(PUSCH_TARGETSNRX10_RANGE),   
    max(PUCCH_TARGETSNRX10_RANGE),   
    max(MAX_RXGAIN_RANGE),           
    max(PRACH_DTX_THRESHOLD_RANGE),  
    max(NROF_UPLINK_SYMBOLS_RANGE),  
    max(RSRP_THRESHOLD_SSB_RANGE),   
    max(SESSION_AMBR_UL0_RANGE),     
    max(SESSION_AMBR_DL0_RANGE)      
]

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


def modify_config_map(whale):
    # Load the existing core.yaml for SMF configuration
    with open('core.yaml', 'r') as file:
        core_config = yaml.safe_load(file)
    
    # Modify session AMBR values in the SMF configuration
    for info in core_config['data']['config.yaml']['smf']['local_subscription_infos']:
        if info['single_nssai']['sst'] == 1:  # Assuming SST 1 is the target slice
            info['qos_profile']['session_ambr_ul'] = f"{whale[8]}Mbps"
            info['qos_profile']['session_ambr_dl'] = f"{whale[9]}Mbps"
    
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
    gnb_conf_str = re.sub(r"p0_NominalWithGrant\s*=\s*[-0-9]+", f"p0_NominalWithGrant = {whale[0]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pusch_TargetSNRx10\s*=\s*[0-9]+", f"pusch_TargetSNRx10 = {whale[2]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"pucch_TargetSNRx10\s*=\s*[0-9]+", f"pucch_TargetSNRx10 = {whale[3]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"prach_dtx_threshold\s*=\s*[0-9]+", f"prach_dtx_threshold = {whale[5]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"nrofUplinkSymbols\s*=\s*[0-9]+", f"nrofUplinkSymbols = {whale[6]}", gnb_conf_str)
    gnb_conf_str = re.sub(r"rsrp_ThresholdSSB\s*=\s*[0-9]+", f"rsrp_ThresholdSSB = {whale[7]}", gnb_conf_str)

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


def evaluate_whale(whale):
   
    # Modify the config map and restart pods as per the new whale configuration
    modify_config_map(whale)
    restart_pods()
    time.sleep(180)  # Wait for 3 minutes after restarting pods to ensure the network is stable

    # Identify the UE pod
    ue_pod_cmd = "kubectl get pods --no-headers | grep '^oai-nr-ue' | awk '{print $1}'"
    ue_pod_name = subprocess.check_output(ue_pod_cmd, shell=True).decode().strip()

    if not ue_pod_name:
        logging.error("Could not find UE pod. Aborting evaluation.")
        return 0

    # Execute ping command from the UE pod to the UPF's IP address
    ping_cmd = f"kubectl exec {ue_pod_name} -- ping -c 10 12.1.1.1"
    ping_result = subprocess.run(ping_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ping_output = ping_result.stdout.decode()

    # Parse the output to find the average latency
    
    avg_latency_match = re.search(r"rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/", ping_output)
    if avg_latency_match:
        avg_latency = avg_latency_match.group(1)
        score = float(avg_latency)  # Convert the average latency to a float value
        
    else:
        logging.error("Could not parse average latency from ping output. Setting score to a high value to indicate poor     performance.")
        score = float('inf')  # Assign a high score to indicate poor performance as latency could not be measured
  
    return score

best_whale = whales[0].copy()

# Optimization loop
best_scores_history = []  # To track the best score in each iteration

for iteration in range(MAX_ITERATIONS):
    a = 2 - iteration * (2.0 / MAX_ITERATIONS)  # 'a' decreases linearly from 2 to 0

    for idx, whale in enumerate(whales):
        r1, r2 = np.random.rand(), np.random.rand()  # Random coefficients
        A = 2 * a * r1 - a  # Coefficient A
        C = 2 * r2  # Coefficient C
        b = 1  # Parameter for the spiral equation
        l = np.random.uniform(-1, 1)  # Random number in [-1,1]

        p = np.random.rand()  # Random number [0,1]

        if p < 0.5:
            if abs(A) < 1:
                # Shrinking encircling mechanism
                D = abs(C * best_whale - whale)
                whales[idx] = best_whale - A * D
            else:
                # Spiral updating position
                distance_to_best = abs(best_whale - whale)
                whales[idx] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
        else:
            # Spiral updating position
            distance_to_best = abs(best_whale - whale)
            whales[idx] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale

        # Ensure the whale's parameters are within their respective bounds
        whales[idx] = np.clip(whales[idx], param_min_bounds, param_max_bounds)

    # Evaluate the performance of the new whale positions
    scores = np.array([evaluate_whale(whale) for whale in whales])
    current_best_idx = np.argmin(scores)
    current_best_score = scores[current_best_idx]

    # Update the best solution if the current best is better
    if iteration == 0 or current_best_score < best_scores_history[-1]:
        best_whale = whales[current_best_idx].copy()
        best_scores_history.append(current_best_score)
        logging.info(f"Iteration {iteration + 1}/{MAX_ITERATIONS}, Best Score: {current_best_score}")

# Plot the best scores history to visualize the optimization progress
plt.plot(best_scores_history, marker='o')
plt.title('Best Score Evolution')
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.grid(True)
plt.show()

# Final output
logging.info(f"Optimization completed. Best Score: {best_scores_history[-1]}, Best Whale: {best_whale}")