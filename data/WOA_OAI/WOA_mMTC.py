import subprocess
import time
import numpy as np
import re
import logging
import os
import yaml
import threading

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("whale_optimization_mMTC_k8s.log"), logging.StreamHandler()])

# WOA Parameters
NUM_WHAALES = 5  # Number of whales
MAX_ITERATIONS = 30
NO_IMPROVEMENT_THRESHOLD = 7
B = 1  # Defines the shape of the spiral

# Parameters Range
PARAMETER_RANGES = {
    'P0_NOMINALWITHGRANT': [-45, -60, -75, -90, -105, -120, -135, -150, -165, -180],
    'SSPBCH_BLOCKPOWER': [-3, -6, -9, -12, -15, -18, -21, -24, -27, -30],
    'PUSCH_TARGETSNRX10': list(range(100, 310, 20)),
    'PUCCH_TARGETSNRX10': list(range(100, 310, 20)),
    'MAX_RXGAIN': list(range(10, 126)),  # 0 to 125
    'PRACH_DTX_THRESHOLD': list(range(100, 181)),  # 100 to 180
    'NROF_UPLINK_SYMBOLS': list(range(1, 7)),  # 0 to 6
    'RSRP_THRESHOLD_SSB': list(range(1, 30)),  # 1 to 29
    'SESSION_AMBR_UL0': list(range(100, 500)),  # 100 to 900
    'SESSION_AMBR_DL0': list(range(100, 500))  # 100 to 900
}

# Initialize whales
whales = np.array([[np.random.choice(PARAMETER_RANGES[param]) for param in PARAMETER_RANGES] for _ in range(NUM_WHAALES)])

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




def get_full_pod_name(pod_name_start):
    # Use kubectl to get the full name of the pod based on its starting pattern
    command = f"kubectl get pods --no-headers -o custom-columns=\":metadata.name\" | grep \"{pod_name_start}\""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    full_pod_name = result.stdout.decode().strip().split('\n')[0]  # Take the first matching pod
    return full_pod_name

def run_iperf(pod_name_start, jitter_results):
    try:
        full_pod_name = get_full_pod_name(pod_name_start)
        if not full_pod_name:
            logging.error(f"Pod starting with '{pod_name_start}' not found.")
            jitter_results.append(float('inf'))  # Use a high jitter value as an indication of failure
            return

        # Run iperf command in client mode, adjust the server IP and other parameters as needed
        iperf_command = f"kubectl exec {full_pod_name} -- iperf -c <iperf-server-ip> -u -t 10"
        result = subprocess.run(iperf_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.decode()

        # Extract jitter value from iperf output
        jitter_match = re.search(r"Jitter: ([\d\.]+) ms", output)
        if jitter_match:
            jitter = float(jitter_match.group(1))
            jitter_results.append(jitter)
            
        else:
            logging.warning(f"Jitter not found in iperf output for {full_pod_name}")

    except Exception as e:
        logging.error(f"Error running iperf on pod starting with '{pod_name_start}': {e}")
        jitter_results.append(float('inf'))  # Use a high jitter value in case of error

def evaluate_whale(whale, num_ues):
    # Assuming modify_config_map and restart_pods are implemented elsewhere
    # modify_config_map(whale)
    # restart_pods()
    # time.sleep(180)  # Wait for the network to stabilize

    jitter_results = []  # Store jitter results from each UE
    threads = []  # To run iperf tests in parallel

    # Iterate over the number of UEs and create threads to run iperf tests
    for i in range(1, num_ues + 1):
        pod_name_start = f"oai-nr-ue-{i}-"  # Construct the pod name starting pattern
        thread = threading.Thread(target=run_iperf, args=(pod_name_start, jitter_results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate the overall jitter, e.g., as the average jitter across all UEs
    if jitter_results:
        average_jitter = sum(jitter_results) / len(jitter_results)
        logging.info(f"Whale with parameters {whale} achieved an average jitter of {avg_jitter} ms")
        return average_jitter
    else:
        logging.error("No jitter results collected.")
        return float('inf')  # Return a high value to indicate an issue

# WOA algorithm
global_best = whales[0].copy()
global_best_score = np.inf
no_improvement_count = 0

def update_whale(whale, global_best, a):
    r = np.random.rand(len(whale))  # Random vector [0, 1]
    A = 2 * a * np.random.rand() - a  # A is a single value
    C = 2 * r  # Equation (2.4) in WOA paper

    if np.abs(A) < 1:
        # Shrinking encircling mechanism
        D = np.abs(C * global_best - whale)
        new_whale = global_best - A * D
    else:
        # Spiral-shaped path to simulate the helix-shaped movement of whales
        D = np.abs(global_best - whale)
        l = np.random.uniform(-1, 1)
        new_whale = D * np.exp(B * l) * np.cos(2 * np.pi * l) + global_best

    # Ensure new_whale is within bounds
    for i in range(len(new_whale)):
        if new_whale[i] < min(PARAMETER_RANGES[i]):
            new_whale[i] = min(PARAMETER_RANGES[i])
        elif new_whale[i] > max(PARAMETER_RANGES[i]):
            new_whale[i] = max(PARAMETER_RANGES[i])

    return new_whale


for iteration in range(MAX_ITERATIONS):
    a = 2 - iteration * (2 / MAX_ITERATIONS)  # Decreases from 2 to 0

    # Update positions of all whales and evaluate them
    for i in range(NUM_WHAALES):
        whales[i] = update_whale(whales[i], global_best, a)
        score = evaluate_whale(whales[i], num_ues=5)  # Assuming 5 UEs for mMTC scenario

        # Update global best if necessary
        if score < global_best_score:
            global_best = whales[i].copy()
            global_best_score = score

    # Early stopping if no improvement
    #if no_improvement_count >= NO_IMPROVEMENT_THRESHOLD:
        #logging.info(f"Stopping early due to no improvement.")
        #break
    # Log the status after each iteration
    logging.info(f"Iteration {iteration + 1}/{MAX_ITERATIONS}: Best Score = {global_best_score}")
# Log final results
logging.info(f"Optimization completed. Best Score: {global_best_score}, Best Whale: {global_best}")
