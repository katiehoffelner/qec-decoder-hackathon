import numpy as np
from hackathon_2026 import problem_info
import stim
from hackathon_2026 import logical_error_rate
import pandas as pd

df = pd.read_csv("data/train/train.csv")
print(df.columns.tolist())

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

# Z-type errors 
circuit_z = stim.Circuit.from_file("data/train/Z_circuit.stim")
info_z = problem_info(circuit_z)

df_z = df[df["basis"] == "Z"]
events_z = df_z[[f"D{i}" for i in range(70)]].values.astype(np.uint8)[:100]
true_z = df_z[["L0","L1","L2","L3"]].values.astype(np.uint8)[:100]

# no. shots for test and true comparison
pred_z = bp_decode(info_z["H"], info_z["L"], info_z["p"], events_z, n=50)
print("Z LER:", logical_error_rate(pred_z, true_z))
# value was 0.1675

# X- type errors
circuit_x = stim.Circuit.from_file("data/train/X_circuit.stim")
info_x = problem_info(circuit_x)

# no. shots for test and true comparison
df_x = df[df["basis"] == "X"]
events_x = df_x[[f"D{i}" for i in range(70)]].values.astype(np.uint8)[:100]
true_x = df_x[["L0","L1","L2","L3"]].values.astype(np.uint8)[:100]

pred_x = bp_decode(info_x["H"], info_x["L"], info_x["p"], events_x, n=50)
print("X-type error logical error rate:", logical_error_rate(pred_x, true_x))
#value was 0.2125

predictions = np.vstack([pred_z, pred_x])
true_all = np.vstack([true_z, true_x])
print("logical error rate(Z&X):", logical_error_rate(predictions, true_all))
# value was 0.19