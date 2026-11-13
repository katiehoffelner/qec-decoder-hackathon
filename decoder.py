import numpy as np
from hackathon_2026 import problem_info
import stim
from hackathon_2026 import logical_error_rate
import pandas as pd

df = pd.read_csv("data/train/train.csv")
print(df.columns.tolist())

#get z errors 
df_z = df[df["basis"] == "Z"]

#detector columns 
det_cols = [f"D{i}" for i in range(70)]
events = df_z[det_cols].values.astype(np.uint8)

# true observable columns
obs_cols = ["L0", "L1", "L2", "L3"]
true_observables = df_z[obs_cols].values.astype(np.uint8)

circuit = stim.Circuit.from_file("data/test/Z_circuit.stim")#
# x stabilisers needed to correct for z error noise 
info = problem_info(circuit)
# 18, 4, 4
# 5 rounds of error correction 
H = info["H"]  # matrix shape
L = info["L"]  # observables
p = info["p"]  # prob vector shape

def bp_decode(H, L, p, events, n=50):

    num_shots, num_checks = events.shape
    num_mechanisms = H.shape[1]
    predictions = np.zeros((num_shots, L.shape[0]), dtype=np.uint8)
    
    # log likleyhood ratios = weights how much the communication is trusted in each bp round 
    LLR = np.log((1 - p) / p) 

    #monte carlo for no. events/shots
    for shot in range(num_shots):
        syndrome = events[shot] # syndrome value to check for errors later
        
        # to talk back and forth 
        # list for one way and one for back 
        v2c = np.tile(LLR, (num_checks, 1))  # valiable to check 
        c2v = np.zeros((num_checks, num_mechanisms)) #  check to variable 
        
        converged = False
        #for each round of bp
        for _ in range(n):
            #each node communicates w the neignbhouting one 
            for i in range(num_checks):
                neighbors = np.where(H[i] == 1)[0] 
                if len(neighbors) == 0:
                    continue
                msgs = v2c[i, neighbors] #communication
                tanh_msgs = np.tanh(msgs / 2) #LLR messages to easier form 
                for j in neighbors:
                    not_j = tanh_msgs[neighbors != j] #all other messages (bc nodes communicate)
                    prod = np.prod(not_j) # multiply
                    prod = np.clip(prod, -1+1e-10, 1-1e-10) #limits
                    sign = -1 if syndrome[i] == 1 else 1 # error is when syndrome is 1 and this is the sign flip of anticommutation
                    c2v[i, j] = sign * 2 * np.arctanh(prod) # final message sent 
            
            # 
            total = LLR + c2v.sum(axis=0) #reliability of messages and all messages gets the most reliable what has happened value
            for i in range(num_checks):
                neighbors = np.where(H[i] == 1)[0] # communication 
                v2c[i, neighbors] = total[neighbors] - c2v[i, neighbors] # avoid duplicates 
            
            e_hat = (total < 0).astype(np.uint8)  # error descision 
            
            # h dot s mod 2 
            if np.all((H @ e_hat % 2) == syndrome):
                converged = True
                break
        
        if converged:
            predictions[shot] = L @ e_hat % 2 # if error then get most likley error type (here only runs z but should work for both later)
        else:
            predictions[shot] = 0  # assume no error bc did not converge to one 
    
    return predictions


events = events[:100] #number of shots tested - increase for better results but that slows it all down 
true_observables = true_observables[:100] # match events no

predictions = bp_decode(H, L, p, events, n=50) 
print("logical error rate:", logical_error_rate(predictions, true_observables))
