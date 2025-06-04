import time
import collections
import random
import copy
import math
import os
import pickle
import torch
import csv
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
from scipy.stats import norm
from scipy.spatial import distance_matrix

import utils


def output_divergence_to_csv(divergence_value, env_name, dataset):
    file_path = "dtv_output_"+ env_name + "_var_from_covar_" + dataset + "_5_steps.csv"
    # Open or create the CSV file in append mode
    with open(file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        
        # Write the divergence value to the CSV file
        csv_writer.writerow([divergence_value])

def save_dict_count(dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'count_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(dict, file)

def save_state_return_model(state_return_model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_return_path = os.path.join(save_dir, 'state_return_model.pth')
    torch.save(state_return_model, state_return_path)

def save_action_dist_model(action_dist_model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    action_dist_path = os.path.join(save_dir, 'action_dist_model.pth')
    torch.save(action_dist_model, action_dist_path)

def update_and_save_model(current_loss, best_loss, model, save_func, save_dir):
    if best_loss is None or current_loss < best_loss:
        best_loss = current_loss
        save_func(model, save_dir)
    return best_loss
    

class Trainer:

    def __init__(self, model,  optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, 
                percentile=0.08, action_dist_model=None, state_return_model=None, action_dist_model_optimizer=None, action_dist_model_scheduler=None, 
                state_return_model_optimizer=None, state_return_model_scheduler=None, dict_record_count=None,  args=None):
        """
        use_data_augmentation (bool): Whether to use use_data_augmentation.
        original_dt (bool): Whether to use original_dt predicting action or use action distribution prediction.
        aug_iter (int): The iteration to start use_data_augmentation.
        max_len (int): The maximum length of the deque buffer
        seed (int): Random seed for reproducibility.
        percentile (float): The percentile to use for the z-score.
        action_dist_model (nn.Module): The model to use for data augmentation for state+return.
        """
        self.args = args
        self.seed = args.seed
        self.env_name = args.env
        self.dataset = args.dataset

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.sampling_aug_ratio = float(args.sampling_aug_ratio)
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.diagnostics['training/state_MSE_error'] = []
        self.diagnostics['training/return_MSE_error'] = []
        self.diagnostics['training/action_MSE_error'] = []
        self.diagnostics['training/dist_z_score_error'] = []
        self.diagnostics['training/diff return predict and target'] = []
        self.diagnostics['training/diff action predict and target'] = []
        self.diagnostics['training/diff state predict and target'] = []
        self.diagnostics['training/eval diff return predict and target'] = []
        self.diagnostics['training/eval diff action predict and target'] = []
        self.diagnostics['training/eval diff state predict and target'] = []
        self.diagnostics['training/state_aug_data_MSE_error'] = []
        self.diagnostics['training/return_aug_data_MSE_error'] = []
        self.diagnostics['training/action_aug_data_MSE_error'] = []
        self.diagnostics['training/diff action_aug predict and target'] = []
        self.diagnostics['training/diff return_aug predict and target'] = []
        self.diagnostics['training/diff state_aug predict and target'] = []

        self.start_time = time.time()
        self.original_dt = args.original_dt
        self.use_data_augmentation = args.use_data_augmentation
        self.aug_iter = args.aug_iter
        self.max_len = args.max_len_aug
        self.under_loading = args.under_loading
        self.percentage_under_loading = args.percentage_under_loading

        # Record count
        self.dict_record_count = {}
        if dict_record_count is not None:
            self.dict_record_count = dict_record_count
        
        # aug_model for state
        self.state_return_model = state_return_model
        self.state_return_model_optimizer = state_return_model_optimizer
        self.state_return_model_scheduler = state_return_model_scheduler
        self.best_state_return_model_loss = None
        # aug_model for action
        self.action_dist_model = action_dist_model
        self.action_dist_model_optimizer = action_dist_model_optimizer
        self.action_dist_model_scheduler = action_dist_model_scheduler
        self.best_action_dist_model_loss = None

        # Augmented data deque buffer
        if args.load_aug_data:
            directory = "gym/data/aug_data"
            file_name = f'augmented_data_{self.env_name}_{self.dataset}_{self.seed}_{self.threshold}_{args.percentage_less_data}.pkl'
            file_path = os.path.join(directory, file_name)
            self.augmented_data = utils.load_augmented_data(file_path=file_path)
            if self.augmented_data == {}:
                keys = ['states', 'actions', 'rewards', 'dones', 'rtg', 'timesteps', 'attention_mask']   
                self.augmented_data = {key: collections.deque(maxlen=self.max_len) for key in keys}
        else:
            keys = ['states', 'actions', 'rewards', 'dones', 'rtg', 'timesteps', 'attention_mask']   
            self.augmented_data = {key: collections.deque(maxlen=self.max_len) for key in keys}
        
        
        self.no_search_action = args.no_search_action
        self.tracking_selected_action = {i: 0 for i in range(args.no_search_action)}

        # Tracking model to saved for augmented data
        self.tracking_aug_iter = 0
        self.tracking_change_return = []
        self.aug_data_new_tracking = 0                  # How many new data?

        # Z-score for uncertainty of action
        self.percentile = percentile

        # Threshold for uncertainty of state

        self.threshold = utils.get_uncertainty_threshold(self.env_name, self.dataset)
        
 
    def train_iteration(self, iter_num, print_logs=False):
        num_steps = self.args.num_steps_per_iter
        train_augment_model = self.args.train_augment_model
        percentage_less_data = self.args.percentage_less_data
        use_less_data= self.args.use_less_data
        load_aug_data = self.args.load_aug_data
        use_modify_env = self.args.use_modify_env
        if self.env_name == "hopper" or "halfcheetah":      
            modification = self.args.modification
        else:
            modification = None

        train_losses = []
        train_state_return_losses = []
        train_action_dist_losses = []
        logs = dict()
        train_start = time.time()

        if self.use_data_augmentation and (not train_augment_model) and (not load_aug_data): 
            if iter_num - self.tracking_aug_iter == self.aug_iter:
                # self.tracking_aug_iter = iter_num
                self.tracking_change_return = []
                print("Begin Augmenting Data")
                for _ in trange(self.max_len):
                    self.add_aug_data_only_new_count()

            # Save augmented data      
            directory = "./CRDT/data/aug_data"
            file_name = f'augmented_data_{self.env_name}_{self.dataset}_{self.seed}_{self.threshold}_small_maze_less_data_1_50.pkl'
            file_path = os.path.join(directory, file_name)
            # # Ensure the directory exists
            os.makedirs(directory, exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self.augmented_data, f)
            # exit()

        self.model.train()
        self.action_dist_model.train()
        self.state_return_model.train()
        print("Begin Training")
        for _ in tqdm(range(num_steps)):
            if self.original_dt:
                train_loss = self.train_step() 
            else:
                if train_augment_model:
                    action_dist_loss = self.train_action_dist_pred(iter_num, max_iters=50)
                    train_state_return_loss = self.train_state_return_predict(iter_num, max_iters=50)
                    # action_dist_loss = self.train_state_return_action_dist_predict_aug_data_only(iter_num, max_iters=50)
                    # action_dist_loss = self.train_state_return_action_dist_predict_aug_data_only_with_count(iter_num, max_iters=50)
                    # print(action_dist_loss)
                    # print(train_state_return_loss)
                else:
                    if self.under_loading:
                        train_loss = self.train_step_update_underloading() 
                    else:
                        train_loss_aug = self.train_step_update()
                        train_loss_ori = self.train_step()
                        train_loss = train_loss_ori + train_loss_aug if train_loss_aug is not None else train_loss_ori
                    


            if (self.use_data_augmentation and (train_augment_model is False)) or self.original_dt: 
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

            if train_augment_model:
                train_action_dist_losses.append(action_dist_loss)
                train_state_return_losses.append(train_state_return_loss)
                if self.action_dist_model_scheduler is not None:
                    self.action_dist_model_scheduler.step()
                if self.state_return_model_scheduler is not None:
                    self.state_return_model_scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()
        print("Begin Evaluation")
        self.model.eval()
        for eval_fn in self.eval_fns:                                           # For each environment there are two evaluating return for DT? 
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs['evaluation/{}'.format(k)] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['aug_data/len'] = len(self.augmented_data["states"])
        logs['aug_data/tracking_new_data'] = self.aug_data_new_tracking

        if len(self.tracking_change_return) > 0:
            logs['aug_data/tracking_change_return'] = sum(self.tracking_change_return)/len(self.tracking_change_return)
            logs['aug_data/sum_tracking_change_return'] = sum(self.tracking_change_return)
            logs['aug_data/max_tracking_change_return'] = max(self.tracking_change_return)
            logs['aug_data/min_tracking_change_return'] = min(self.tracking_change_return)

        for k in self.diagnostics:
            if self.diagnostics[k] != []: 
                self.diagnostics[k] = sum(self.diagnostics[k]) / len(self.diagnostics[k])
                logs[k] = self.diagnostics[k]

        self.diagnostics['training/state_MSE_error'] = []
        self.diagnostics['training/return_MSE_error'] = []
        self.diagnostics['training/action_MSE_error'] = []
        self.diagnostics['training/dist_z_score_error'] = []
        self.diagnostics['training/diff return predict and target'] = []
        self.diagnostics['training/diff action predict and target'] = []
        self.diagnostics['training/diff state predict and target'] = []
        self.diagnostics['training/eval diff return predict and target'] = []
        self.diagnostics['training/eval diff action predict and target'] = []
        self.diagnostics['training/eval diff state predict and target'] = []
        self.diagnostics['training/state_aug_data_MSE_error'] = []
        self.diagnostics['training/return_aug_data_MSE_error'] = []
        self.diagnostics['training/action_aug_data_MSE_error'] = []
        self.diagnostics['training/diff action_aug predict and target'] = []
        self.diagnostics['training/diff return_aug predict and target'] = []
        self.diagnostics['training/diff state_aug predict and target'] = []

        if print_logs:
            print('=' * 80)
            print('Iteration {}'.format(iter_num))
            for k, v in logs.items():
                print('{}: {}'.format(k, v))
        
        # Save model and dict record count
        if train_augment_model and iter_num-1 % 10 == 0:
            if use_less_data:
                save_dir = "saved_model/{}_{}_{}/".format(self.env_name, self.dataset, percentage_less_data)
            elif use_modify_env:
                if modification is not None:
                    save_dir = "saved_model/{}_{}_{}_modified/".format(self.env_name, self.dataset, modification)
                else:
                    save_dir = "saved_model/{}_{}_modified/".format(self.env_name, self.dataset)
            else:
                save_dir = "saved_model/{}_{}/".format(self.env_name, self.dataset)
            
            if self.dict_record_count:
                save_dict_count(self.dict_record_count, save_dir)
            
            self.best_action_dist_model_loss = update_and_save_model(
                np.mean(train_action_dist_losses),
                self.best_action_dist_model_loss,
                self.action_dist_model,
                save_action_dist_model,
                save_dir
            )
            
            self.best_state_return_model_loss = update_and_save_model(
                np.mean(train_state_return_losses),
                self.best_state_return_model_loss,
                self.state_return_model,
                save_state_return_model,
                save_dir
            )
            exit()

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
    

    def add_aug_data_only_new_count(self):
        """
        Augment data with dist measurement
        """
        self.action_dist_model.eval()
        self.state_return_model.eval()
        # self.tracking_aug_model.eval()
        # Update sampling method
        # Step 1: Sample a batch of data
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size) # states (64*20*11)
        for i in range(0, states.shape[0]):
            if len(self.augmented_data["states"]) == self.max_len:
                break
            ori_rtg = copy.deepcopy(rtg[i])
            accum_dist = 0
            # Step 2: Index from half of the tensor to the end
            ratio = 0.5
            idx = int(states[i].shape[0]*ratio)

            for j in range(idx, states[i].shape[0]-1):   
                try:
                    lowest_states = copy.deepcopy(states[i][j+1])
                    lowest_actions = copy.deepcopy(actions[i][j+1])
                    lowest_rtg = copy.deepcopy(rtg[i][:-1][j+1])
                    lowest_attention_mask = copy.deepcopy(attention_mask[i][j+1])
                    lowest_selected_action = None
                    add_accum_dist = accum_dist
                    fix_accum_dist = accum_dist
                    # Augment actions
                    _, _, _, action_dist, _ = self.action_dist_model.get_value(
                        states[i][:j+1], actions[i][:j+1], rewards[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1],)
                    
                    no_meet = self.get_from_count_dict(states[i][:j+1], actions[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1])
                    for action_no in range(self.no_search_action):
                        with torch.no_grad():
                            actions[i][j+1] = self.sample_low_dist_action_with_count(action_dist, action_no, no_meet, self.args.step_action_search).squeeze(0)[-1].detach()
                            # Clamp to correct range
                            actions[i][j+1] = torch.clamp(actions[i][j+1], min=-1, max=1)

                        # Measuring uncertainty in state
                        self.state_return_model.train()
                        stack_state_preds = []
                        for _ in range(5):
                            state_preds, _, _, _, _ = self.state_return_model.get_value(
                                states[i][:j+1], actions[i][:j+2], rewards[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1],)
                            stack_state_preds.append(state_preds.squeeze(0)[-1].detach().cpu())
                        
                        distance = utils.measure_variance_from_covariance(torch.stack(stack_state_preds).cpu().numpy())
                        add_accum_dist += distance
                        
                        if add_accum_dist > self.threshold:
                            actions[i][j+1] = lowest_actions
                            add_accum_dist = fix_accum_dist
                            continue
                        
                        self.state_return_model.eval()

                        # Augment states
                        state_preds, _, return_preds, _, mask = self.state_return_model.get_value(
                            states[i][:j+1], actions[i][:j+2], rewards[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1],)
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
                    
                    if lowest_selected_action is not None:
                        self.tracking_selected_action[lowest_selected_action] += 1
                    
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
                            keys = ["states", "actions", "rewards", "dones", "rtg", "timesteps", "attention_mask"]
                            new_data = [states[i], actions[i], rewards[i], dones[i], rtg[i], timesteps[i], attention_mask[i]]

                            # Append new data to each list
                            for key, data in zip(keys, new_data):
                                self.augmented_data[key].appendleft(data.cpu())
                            self.aug_data_new_tracking += 1
                            # print("Number of new data", self.aug_data_new_tracking)
                            self.tracking_change_return.append(ori_rtg[:-1][j+1].item()-rtg[i][:-1][j+1].item())
                except:
                    print(states.shape)
                    print(actions.shape)
                    print(rewards.shape)
                    print(dones.shape)
                    print(rtg.shape)
                    print(timesteps.shape)
                    print(attention_mask.shape)
                    print("Error")
                    continue


    
    def sample_low_dist_action_with_count(self, action_dist, count, no_meet, step):
        # z_score = self.z_score if torch.rand(1).item() < 0.5 else -self.z_score
        z_score = -utils.percentile_to_z_score(self.percentile - count* step)
        # z_score = self.z_score
        if no_meet == 1 or no_meet == 0:
            selected_action = action_dist.mean + z_score * action_dist.stddev
        else:
            selected_action = action_dist.mean + z_score * action_dist.stddev * math.sqrt(math.log(no_meet))
        print("No meet: ", no_meet) 
        # selected_action = action_dist.mean + z_score * action_dist.stddev * math.sqrt(math.log(no_meet))

        return selected_action
    

    def get_batch_aug_data(self, batch_size=256):
        """
        Sample similar to sample_batch but from augmented data
        """
        i = random.randint(0, len(self.augmented_data["states"])-1)
        s = self.augmented_data["states"][i]
        a = self.augmented_data["actions"][i]
        r = self.augmented_data["rewards"][i]
        d = self.augmented_data["dones"][i]
        rtg = self.augmented_data["rtg"][i]
        timesteps = self.augmented_data["timesteps"][i]
        mask = self.augmented_data["attention_mask"][i]
        return s, a, r, d, rtg, timesteps, mask

    def get_batch_aug_data_only(self, batch_size=256):
        """
        Sample similar to sample_batch but from augmented data
        """
        # Generate random indices for sampling without replacement to avoid duplicate samples in the batch.
        indices = np.random.choice(range(len(self.augmented_data["states"])), size=batch_size)

        # Use the indices to sample from each array in the augmented data.
        s = [self.augmented_data["states"][i] for i in indices]
        a = [self.augmented_data["actions"][i] for i in indices]
        r = [self.augmented_data["rewards"][i] for i in indices]
        d = [self.augmented_data["dones"][i] for i in indices]
        rtg = [self.augmented_data["rtg"][i] for i in indices]
        timesteps = [self.augmented_data["timesteps"][i] for i in indices]
        mask = [self.augmented_data["attention_mask"][i] for i in indices]

        s = torch.from_numpy(np.stack(s, axis=0)).to(dtype=torch.float32, device="cuda")
        a = torch.from_numpy(np.stack(a, axis=0)).to(dtype=torch.float32, device="cuda")  # Changed device to "cuda"
        r = torch.from_numpy(np.stack(r, axis=0)).to(dtype=torch.float32, device="cuda")  # Changed device to "cuda"
        d = torch.from_numpy(np.stack(d, axis=0)).to(dtype=torch.long, device="cuda")  # Changed device to "cuda"
        rtg = torch.from_numpy(np.stack(rtg, axis=0)).to(dtype=torch.float32, device="cuda")  # Changed device to "cuda"
        timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.long, device="cuda")  # Changed device to "cuda"
        mask = torch.from_numpy(np.stack(mask, axis=0)).to(device="cuda")

        return s, a, r, d, rtg, timesteps, mask
    
    def train_step_dist_pred(self):
        pass

    def train_return_predict(self):
        pass

    def train_state_predict(self):
        pass

    def train_state_return_predict(self):
        pass
        
