
import os
import pickle
import torch
import copy
import math
import hashlib

import numpy as np
from scipy.stats import norm

PERCENTILE = 0.08

def load_dict_count(save_dir):
    dict_path = os.path.join(save_dir, 'count_dict.pkl')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            dict_record_count = pickle.load(f)
        print("Dictionary loaded successfully.")
    else:
        print("Dictionary file not found.")
        dict_record_count = {}
    return dict_record_count

def load_state_return_model(save_dir, device):
    state_return_path = os.path.join(save_dir, 'state_return_model.pth')
    if os.path.exists(state_return_path):
        # Load the entire model
        state_return_model = torch.load(state_return_path, map_location=device)
        print("State-return model loaded successfully.")
        return state_return_model
    else:
        print("State-return model file not found.")

def load_action_dist_model(save_dir, device):
    action_dist_path = os.path.join(save_dir, 'action_dist_model.pth')
    if os.path.exists(action_dist_path):
        # Load the entire model
        action_dist_model = torch.load(action_dist_path, map_location=device)
        print("Action model loaded successfully.")
        return action_dist_model
    else:
        print("Action model file not found.")

def percentile_to_z_score(percentile):
    return norm.ppf(percentile)

def measure_variance_from_covariance(predictions):
    covariance_matrix = np.cov(predictions.T)
    variances = np.diag(covariance_matrix)
    max_variance = np.amax(variances)
    return max_variance

def get_uncertainty_threshold(env_name, dataset):
    """Test value 0.5, 1, 1.5, 2 * STDEV
        At the begining make some mistake so use threshold for medium-replay for the other but threshold should be diff.
    """
    threshold = None
    if (env_name == 'hopper' or "hopper_modify") and dataset == "medium-replay":
        threshold = 0.65          # mean=0.29 from csv file stdev=0.34. Underloading: 0.65 (under baseline), 0.98 (under baseline), 0.46  
                                # No_underloading: 0.65 (under baseline), 0.46, 0.98
    
    if (env_name == 'hopper' or "hopper_modify") and dataset == "random":
        threshold = 0.128          # mean=0.07 from csv file stdev=0.058
                              

    if (env_name == 'hopper' or "hopper_modify") and dataset == "medium":
        threshold = 0.73          # mean=0.23 from csv file stdev=0.25. Underloading: 0.65 (similar baseline), 0.48      
                                    # No_underloading: 0.48 (under baseline), 0.355, 0.73
        # threshold = 0.22            # new mean=0.06, std=0.16

    if (env_name == 'hopper' or "hopper_modify") and dataset == "medium-expert":
        threshold = 0.41          # mean=0.21 from csv file stdev=0.2 Underloading: 0.65 (under baseline), 0.41 (under baseline), 0.31
                                    # No_underloading: 0.41
        # threshold = 0.114           # new mean=0.022, std=0.091

    if (env_name == 'halfcheetah' or "halfcheetah_modify") and dataset == "medium-replay":
        threshold = 4.2           # mean=2 from csv file stdev=2.2. Underloading: 4.2 (under baseline), 3.1 (under baseline), 6.4 
                                    # No_underloading: 4.2

    if (env_name == 'halfcheetah' or "halfcheetah_modify") and dataset == "medium":
        threshold = 2.5           # mean=1.2 from csv file stdev=1.3. Underloading: 4.2 (over baseline), 2.5
                                    # No underloading: 2.5

        # threshold = 1.47           # new mean=0.47, std=0.99

    if (env_name == 'halfcheetah' or "halfcheetah_modify") and dataset == "random":
        threshold = 1.83           # mean=1.08 from csv file stdev=0.75           
                              

    if (env_name == 'halfcheetah' or "halfcheetah_modify") and dataset == "medium-expert":
        threshold = 0.34           # mean=0.18 from csv file stdev=0.16. Underloading: 4.2 (under baseline), 0.34
                                    # No_underloading: 0.34
        # threshold = 0.11

    if (env_name == 'walker2d' or "walker2d_modify") and dataset == "medium-replay":
        threshold = 1.81           # mean=0.93 from csv file stdev=0.88. Underloading: 1.81 (over baseline), 2.25 (under 1.81, over baseline, 2.69
                                # No_underloading: 1.81
                                
    if (env_name == 'walker2d' or "walker2d_modify") and dataset == "medium":
        threshold = 1.78           # mean=0.46 from csv file stdev=0.66 Underloading: 1.81 (over baseline), 1.1
                                    # No_underloading: 1.1, 1.78
        
        # threshold = 0.71            # mean=0.21, std=0.5

    if (env_name == 'walker2d' or "walker2d_modify") and dataset == "medium-expert":
        threshold = 0.44       

    if (env_name == 'walker2d' or "walker2d_modify") and dataset == "random":
        threshold = 2.04           # mean=0.68 from csv file stdev=1.35   

    if (env_name == 'ant') and dataset == "medium":
        # threshold = 0.749           # mean=0.035 from csv file stdev=0.398 
        threshold = 1.5
        
    if (env_name == 'ant') and dataset == "medium-replay":
        threshold = 0.749           # mean=0.17 from csv file stdev=0.575 
    
    if (env_name == 'maze2d') and dataset == "large":
        threshold = 0.1023           # mean=0.063 from csv file stdev=0.038 
        
    if (env_name == 'maze2d') and dataset == "medium":
        threshold = 0.097            # mean=0.052 from csv file stdev=0.045 

    if (env_name == 'maze2d') and dataset == "umaze":
        threshold = 0.076            # mean=0.031 from csv file stdev=0.044    
    
    if (env_name == 'maze2d_nowall'):
        threshold = 0.183
           
    return threshold

def load_augmented_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            augmented_data = pickle.load(f)
        print("Augmented data loaded successfully.")
    else:
        print("Augmented data file not found.")
        augmented_data = {}
    return augmented_data

def get_from_count_dict(dict_record_count, state, action, rtg, timestep):
    state_np = state.cpu().numpy()
    action_np = action.cpu().numpy()
    rtg_np = rtg.cpu().numpy()
    timestep_np = timestep.cpu().numpy()

    # Convert numpy arrays to bytes
    state_bytes = state_np.tobytes()
    action_bytes = action_np.tobytes()
    rtg_bytes = rtg_np.tobytes()
    timestep_bytes = timestep_np.tobytes()

    all_bytes = state_bytes + action_bytes + rtg_bytes + timestep_bytes

    hash_object = hashlib.md5(all_bytes)
    hash_hex = hash_object.hexdigest()

    if hash_hex not in dict_record_count:
        return 1
    else:
        return dict_record_count.get(hash_hex)  
    
def add_aug_data_only_new_count(augmented_data, dict_record_count, action_dist_model, state_return_model, batch_data, threshold, no_search_action=5, step_action_search=0.01):
    """
    Augment data with dist measurement
    """
    action_dist_model.eval()
    state_return_model.eval()
    # self.tracking_aug_model.eval()
    # Update sampling method
    # Step 1: Sample a batch of data
    states, actions, dones, rtg, timesteps, attention_mask = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5] # states (64*20*11)
    for i in range(0, states.shape[0]):  
        ori_rtg = copy.deepcopy(rtg[i])
        accum_dist = 0
        # Step 2: Index from half of the tensor to the end
        ratio = 0.5
        idx = int(states[i].shape[0]*ratio)

        for j in range(idx, states[i].shape[0]-1):   
            lowest_states = copy.deepcopy(states[i][j+1])
            lowest_actions = copy.deepcopy(actions[i][j+1])
            lowest_rtg = copy.deepcopy(rtg[i][:-1][j+1])
            lowest_attention_mask = copy.deepcopy(attention_mask[i][j+1])
            lowest_selected_action = None
            add_accum_dist = accum_dist
            fix_accum_dist = accum_dist
            # Augment actions
            _, _, _, action_dist, _ = action_dist_model.get_value(
                states[i][:j+1], actions[i][:j+1], None, rtg[i][:-1][:j+1], timesteps[i][:j+1],)
            
            no_meet = get_from_count_dict(dict_record_count, states[i][:j+1], actions[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1])
            for action_no in range(no_search_action):
                with torch.no_grad():
                    actions[i][j+1] = sample_low_dist_action_with_count(action_dist, action_no, no_meet, step_action_search).squeeze(0)[-1].detach()

                # Measuring uncertainty in state
                state_return_model.train()
                stack_state_preds = []
                for _ in range(5):
                    state_preds, _, _, _, _ = state_return_model.get_value(
                        states[i][:j+1], actions[i][:j+2], None, rtg[i][:-1][:j+1], timesteps[i][:j+1],)
                    stack_state_preds.append(state_preds.squeeze(0)[-1].detach().cpu())
                
                distance = measure_variance_from_covariance(torch.stack(stack_state_preds).cpu().numpy())
                add_accum_dist += distance
                
                if add_accum_dist > threshold:
                    actions[i][j+1] = lowest_actions
                    add_accum_dist = fix_accum_dist
                    continue
                
                state_return_model.eval()

                # Augment states
                state_preds, _, return_preds, _, mask = state_return_model.get_value(
                    states[i][:j+1], actions[i][:j+2], None, rtg[i][:-1][:j+1], timesteps[i][:j+1],)
                return_preds = return_preds.squeeze(0)[-1].detach()
                with torch.no_grad():
                    if return_preds.item() < lowest_rtg.item():
                        states[i][j+1] = state_preds.squeeze(0)[-1].detach()
                        rtg[i][:-1][j+1] = return_preds
                        attention_mask[i][j+1]  = mask.squeeze(0)[-1].detach()

                        lowest_states = copy.deepcopy(states[i][j+1])
                        lowest_actions = copy.deepcopy(actions[i][j+1])
                        lowest_rtg = copy.deepcopy(rtg[i][:-1][j+1])
                        lowest_attention_mask = copy.deepcopy(attention_mask[i][j+1])
                        # Update accum_dist
                        accum_dist = add_accum_dist
                        add_accum_dist = fix_accum_dist   
                        lowest_selected_action = action_no
                        continue 
                    else:
                        states[i][j+1] = lowest_states
                        actions[i][j+1] = lowest_actions
                        rtg[i][:-1][j+1] = lowest_rtg
                        attention_mask[i][j+1] = lowest_attention_mask
                        add_accum_dist = fix_accum_dist    
                        continue
            
            if accum_dist == fix_accum_dist:
                # If the accum_dist is not updated -> Cannot find suitable action, then break the loop
                break   

            # if j == states[i].shape[0]-2:
            #     output_divergence_to_csv(accum_dist, self.env_name, self.dataset) 
            
            if j == states[i].shape[0]-2:
                # Cond on the return_preds
                if ori_rtg[:-1][j+1].item() < rtg[i][:-1][j+1].item():
                    continue
                else:
                    keys = ["states", "actions", "rtg", "timesteps", "attention_mask", "dones"]
                    new_data = [states[i], actions[i], rtg[i], timesteps[i], attention_mask[i], dones[i]]

                    # Append new data to each list
                    for key, data in zip(keys, new_data):
                        augmented_data[key].appendleft(data.cpu())
    return augmented_data



def sample_low_dist_action_with_count(action_dist, count, no_meet, step_action_search):
    # z_score = self.z_score if torch.rand(1).item() < 0.5 else -self.z_score
    z_score = percentile_to_z_score(PERCENTILE - count*step_action_search)
    # z_score = self.z_score
    # if no_meet == 1 or no_meet == 0:
    #     selected_action = action_dist.mean + z_score * action_dist.stddev
    # else:
    #     # print(no_meet)
    #     # print(math.log(no_meet))
    #     print(math.sqrt(math.log(no_meet)))
    #     selected_action = action_dist.mean + z_score * action_dist.stddev * math.sqrt(math.log(no_meet))
    # print("No meet: ", no_meet) 
    selected_action = action_dist.mean + z_score * action_dist.stddev * math.sqrt(math.log(no_meet))

    return selected_action