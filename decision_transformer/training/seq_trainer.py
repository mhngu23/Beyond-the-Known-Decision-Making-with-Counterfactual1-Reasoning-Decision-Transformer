import math
import random
import hashlib
import torch
from numbers import Real

from decision_transformer.training.trainer import Trainer

class SequenceTrainer(Trainer): 

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        _, action_preds, _, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_MSE_error'].append(torch.mean((action_preds-action_target)**2).detach().cpu().item()) 
            self.diagnostics['training/diff action predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
        return loss.detach().cpu().item()

    def add_to_count_dict(self, state, action, rtg, timestep):

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
   
        if hash_hex not in self.dict_record_count:
            self.dict_record_count[hash_hex] = 2
        else:
            self.dict_record_count[hash_hex] += 1
    
    def get_from_count_dict(self, state, action, rtg, timestep):
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
   
        if hash_hex not in self.dict_record_count:
            return 1
        else:
            return self.dict_record_count.get(hash_hex)  
        
    def train_state_return_predict(self, iter_num, max_iters=50):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        state_preds, _, return_preds, _ = self.state_return_model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        rtg_target = torch.clone(rtg[:,:-1])
        state_target = torch.clone(states)

        # Calculate return loss
        rtg_dim = return_preds.shape[2]
        return_preds = return_preds.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
        rtg_target = rtg_target.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
        rtg_loss = torch.mean((return_preds-rtg_target)**2)

        # Calculate state loss
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_loss = torch.mean((state_preds-state_target)**2)

        loss = rtg_loss + state_loss

        self.state_return_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state_return_model.parameters(), .25)
        self.state_return_model_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/return_MSE_error'].append(torch.mean((return_preds-rtg_target)**2).detach().cpu().item()) 
            self.diagnostics['training/diff return predict and target'].append(torch.mean(abs((return_preds-rtg_target)/((rtg_target)))*100).detach().cpu().item())
            self.diagnostics['training/state_MSE_error'].append(torch.mean((state_preds-state_target)**2).detach().cpu().item())
            self.diagnostics['training/diff state predict and target'].append(torch.mean(abs((state_preds-state_target)/((state_target)))*100).detach().cpu().item())
        return loss.detach().cpu().item()
    
    
    def train_action_dist_pred(self, iter_num, max_iters=50):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        for i in range(0, states.shape[0]):  
            for j in range(0, states[i].shape[0]-1): 
                self.add_to_count_dict(states[i][:j+1], actions[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1])  

        action_target = torch.clone(actions)

        _, action_preds, _, action_dist = self.action_dist_model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        
        # Calculate dist loss
        act_dim = action_preds.shape[2]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        var = action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]**2
        log_scale = (
            math.log(action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]) if isinstance(action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0], Real) else action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0].log()
        )
        mean = action_dist.loc.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]             # Convert mean to correct shape
        neg_log_likelihood = -(-((action_target - action_dist.loc.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]) ** 2) / (2 * var) - log_scale -  math.log(math.sqrt(2 * math.pi)))
        dist_loss = torch.mean(neg_log_likelihood)
        
        loss = dist_loss

        self.action_dist_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_dist_model.parameters(), .25)
        self.action_dist_model_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_MSE_error'].append(torch.mean((mean-action_target)**2).detach().cpu().item())
            self.diagnostics['training/dist_z_score_error'].append(torch.mean((action_target-mean)/action_dist.stddev.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]))
            self.diagnostics['training/diff action predict and target'].append(torch.mean(abs((mean-action_target)/((action_target)))*100).detach().cpu().item())
        return loss.detach().cpu().item()
    
    
    # def train_state_return_action_dist_predict_aug_data_only_with_count(self, iter_num, max_iters=50):
    #     use_aug_data = False
    #     if self.use_data_augmentation:                                                               # If using data augmentation
    #         if self.augmented_data["states"]:                                                    # If augmented data is not empty
    #             ratio = 0
    #             if random.random() < ratio:                                                               # If random < threshold then use the augmented data 
    #                 states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch_aug_data_only(self.batch_size)         
    #                 use_aug_data = True
    #             else:                                                                                   # If random > threshold then use the normal data
    #                 states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)    
    #         else:                                                                                # If augmented data is empty (1st iteration) 
    #             states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)    
    #     else:                                                                             # If not using data augmentation
    #         states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

    #     for i in range(0, states.shape[0]):  
    #         for j in range(0, states[i].shape[0]-1): 
    #             self.add_to_count_dict(states[i][:j+1], actions[i][:j+1], rtg[i][:-1][:j+1], timesteps[i][:j+1])   

    #     state_preds, action_preds, return_preds, action_dist = self.aug_model.forward(
    #         states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
    #     )

    #     rtg_target = torch.clone(rtg[:,:-1])
    #     state_target = torch.clone(states)
    #     action_target = torch.clone(actions)

    #     # Calculate return loss
    #     rtg_dim = return_preds.shape[2]
    #     return_preds = return_preds.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
    #     rtg_target = rtg_target.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
    #     rtg_loss = torch.mean((return_preds-rtg_target)**2)

    #     # Calculate state loss
    #     state_dim = state_preds.shape[2]
    #     state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
    #     state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
    #     state_loss = torch.mean((state_preds-state_target)**2)

    #     # Calculate dist loss
    #     act_dim = action_preds.shape[2]
    #     action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    #     var = action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]**2
    #     log_scale = (
    #         math.log(action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]) if isinstance(action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0], Real) else action_dist.scale.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0].log()
    #     )
    #     mean = action_dist.loc.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]             # Convert mean to correct shape
    #     neg_log_likelihood = -(-((action_target - action_dist.loc.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]) ** 2) / (2 * var) - log_scale -  math.log(math.sqrt(2 * math.pi)))
    #     dist_loss = torch.mean(neg_log_likelihood)
        
    #     loss = rtg_loss + state_loss + dist_loss

    #     self.aug_opmizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.aug_model.parameters(), .25)
    #     self.aug_opmizer.step()

    #     with torch.no_grad():
    #         self.diagnostics['training/return_MSE_error'].append(torch.mean((return_preds-rtg_target)**2).detach().cpu().item()) 
    #         self.diagnostics['training/diff return predict and target'].append(torch.mean(abs((return_preds-rtg_target)/((rtg_target)))*100).detach().cpu().item())
    #         self.diagnostics['training/state_MSE_error'].append(torch.mean((state_preds-state_target)**2).detach().cpu().item())
    #         self.diagnostics['training/diff state predict and target'].append(torch.mean(abs((state_preds-state_target)/((state_target)))*100).detach().cpu().item())
    #         self.diagnostics['training/dist_z_score_error'].append(torch.mean((action_target-mean)/action_dist.stddev.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]))
    #         if use_aug_data:
    #             self.diagnostics['training/state_aug_data_MSE_error'].append(torch.mean((state_preds-state_target)**2).detach().cpu().item())
    #             self.diagnostics['training/return_aug_data_MSE_error'].append(torch.mean((return_preds-rtg_target)**2).detach().cpu().item())
    #             self.diagnostics['training/diff return_aug predict and target'].append(torch.mean(abs((return_preds-rtg_target)/((rtg_target)))*100).detach().cpu().item())
    #             self.diagnostics['training/diff state_aug predict and target'].append(torch.mean(abs((state_preds-state_target)/((state_target)))*100).detach().cpu().item())


    #     return loss.detach().cpu().item()
    
    def train_step_update_underloading(self):
        use_aug_data = False
        if self.use_data_augmentation:                                                               # If using data augmentation
            if self.augmented_data["states"]:                                                    # If augmented data is not empty
                ratio = 1 - self.percentage_under_loading/100
                if random.random() < ratio:                                                               # If random < threshold then use the augmented data 
                    states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch_aug_data_only(self.batch_size)      
                    use_aug_data = True   
                else:                                                                                   # If random > threshold then use the normal data
                    states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)    
            else:                                                                                # If augmented data is empty (1st iteration) 
                states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)    
        else:                                                                             # If not using data augmentation
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        action_target = torch.clone(actions)

        _, action_preds, _, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_MSE_error'].append(torch.mean((action_preds-action_target)**2).detach().cpu().item()) 
            self.diagnostics['training/diff action predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
            if use_aug_data:
                self.diagnostics['training/action_aug_data_MSE_error'].append(torch.mean((action_preds-action_target)**2).detach().cpu().item())
                self.diagnostics['training/diff action_aug predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
        return loss.detach().cpu().item()

    def train_step_update(self):
        use_aug_data = False
        if self.use_data_augmentation:                                                               # If using data augmentation
            if self.augmented_data["states"]:                                                    # If augmented data is not empty
                ratio = 1.0
                if random.random() < ratio:                                                               # If random < threshold then use the augmented data 
                    states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch_aug_data_only(int(self.batch_size*self.sampling_aug_ratio))      
                    use_aug_data = True
                else:                  
                    return None
            else:
                return None
        else:
            return None

        action_target = torch.clone(actions)

        _, action_preds, _, _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_MSE_error'].append(torch.mean((action_preds-action_target)**2).detach().cpu().item()) 
            self.diagnostics['training/diff action predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
            if use_aug_data:
                self.diagnostics['training/action_aug_data_MSE_error'].append(torch.mean((action_preds-action_target)**2).detach().cpu().item())
                self.diagnostics['training/diff action_aug predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
        return loss.detach().cpu().item()
    
    
    def eval_model_pred(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size, eval=True)

        rtg_target = torch.clone(rtg[:,:-1])
        state_target = torch.clone(states)
        action_target = torch.clone(actions)


        state_preds, action_preds, return_preds, action_dist = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        rtg_dim = return_preds.shape[2]
        return_preds = return_preds.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]
        rtg_target = rtg_target.reshape(-1, rtg_dim)[attention_mask.reshape(-1) > 0]

        # Calculate state loss
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        with torch.no_grad():
            self.diagnostics['training/eval diff action predict and target'].append(torch.mean(abs((action_preds-action_target)/((action_target)))*100).detach().cpu().item())
            self.diagnostics['training/eval diff return predict and target'].append(torch.mean(abs((return_preds-rtg_target)/((rtg_target)))*100).detach().cpu().item())
            self.diagnostics['training/eval diff state predict and target'].append(torch.mean(abs((state_preds-state_target)/((state_target)))*100).detach().cpu().item())





