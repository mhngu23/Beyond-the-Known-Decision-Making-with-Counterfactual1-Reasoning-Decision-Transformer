import gym
import numpy as np
import torch
import wandb
import d4rl
# import tensorboard
# from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
import utils

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def get_env(env_name, dataset='large'):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'hopper_thigh':
        env = gym.make('Hopper-v3', xml_file='./CRDT/xml_file/hopper_thigh.xml')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'hopper_head':
        env = gym.make('Hopper-v3', xml_file='./CRDT/xml_file/hopper_head.xml')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'halfcheetah_thigh':
        env = gym.make('HalfCheetah-v3', xml_file='./CRDT/xml_file/half_cheetah_thigh.xml')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.  
    elif env_name == 'halfcheetah_head':
        env = gym.make('HalfCheetah-v3', xml_file='./CRDT/xml_file/half_cheetah_head.xml')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.      
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'walker2d_modify':
        env = gym.make('Walker2d-v3', xml_file='./CRDT/xml_file/modified_walker_2d.xml')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'ant':
        env_targets = [3600]
        env = gym.make('Ant-v3')
        scale = 1000.  
        max_ep_len = 1000
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'maze2d':
        env_targets = [270]
        d4rl_env = f"{env_name}-{dataset}-v1"
        env = gym.make(d4rl_env)
        scale = 1000.  
        max_ep_len = 1000
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale

def get_score(env_name, ret):
    if env_name == 'hopper' or "hopper_modify":
        score = (ret - (-20.272305))/(3234.3 - (-20.272305))
    if env_name == 'halfcheetah' or "halfcheetah_modify":
        score = (ret - (-280.178953))/(12135.0 - (-280.178953))
    if env_name == 'walker2d' or "walker2d_modify":
        score = (ret - 1.629008)/(4592.3 - 1.629008)
    else:
        raise NotImplementedError
    return score * 100

def setup_model(model, device, args):
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    return model, optimizer, scheduler

def experiment(exp_prefix, args, variant):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.device

    # writer = SummaryWriter()
    log_to_wandb = args.log_to_wandb
    correct_eval = args.correct_eval

    env_name, dataset = args.env, args.dataset
    model_type = args.model_type
    group_name = '{}-{}-{}'.format(exp_prefix, env_name, dataset)
    exp_prefix = '{}-{}'.format(group_name, random.randint(int(1e5), int(1e6) - 1))
    # exp_prefix = '{}'.format(group_name)


    if args.use_modify_env:
        if env_name == "halfcheetah" or env_name == "hopper":
            if args.modification == "thigh":
                env, max_ep_len, env_targets, scale = get_env('{}_thigh'.format(env_name))
            elif args.modification == "head":
                env, max_ep_len, env_targets, scale = get_env('{}_head'.format(env_name))
        else:
            env, max_ep_len, env_targets, scale = get_env('{}_modify'.format(env_name))
    else:
        env, max_ep_len, env_targets, scale = get_env(env_name, dataset=dataset)

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # load dataset
    if env_name == 'maze2d':
        dataset_path = 'gym/data/{}-{}-v1.pkl'.format(env_name, dataset)
    else:
        dataset_path = 'gym/data/{}-{}-v2.pkl'.format(env_name, dataset)
        
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    if args.use_less_data == True:
        ratio = args.percentage_less_data / 100
        num_samples = int(len(trajectories) * ratio)
        trajectories = random.sample(trajectories, num_samples)


    # save all path information into separate lists
    mode = args.mode
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print('Starting new experiment: {} {}'.format(env_name, dataset))
    print('{} trajectories, {} timesteps found'.format(len(traj_lens), num_timesteps))
    print('Average return: {:.2f}, std: {:.2f}'.format(np.mean(returns), np.std(returns)))
    print('Max return: {:.2f}, min: {:.2f}'.format(np.max(returns), np.min(returns)))
    print('=' * 50)

    K = args.K
    batch_size = args.batch_size
    num_eval_episodes = args.num_eval_episodes
    pct_traj = args.pct_traj

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K, eval=False):
        batch_inds = np.random.choice(np.arange(num_trajectories),size=batch_size,replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew, env_name):
        def fn(model):
            returns, lengths, scores = [], [], []
            if env_name == 'maze2d' and correct_eval == True:
                d4rl_env = f"{env_name}-eval-{dataset}-v1"
                eval_env = gym.make(d4rl_env)
            else:
                eval_env = env
            for i in range(num_eval_episodes):
                # print(i)
 
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(eval_env, state_dim, act_dim, model, max_ep_len=max_ep_len, scale=scale, target_return=target_rew/scale, 
                                                           mode=mode, state_mean=state_mean, state_std=state_std, device=device,)
                    else:
                        ret, length = evaluate_episode(eval_env, state_dim, act_dim, model, max_ep_len=max_ep_len, target_return=target_rew/scale, mode=mode, 
                                                       state_mean=state_mean, state_std=state_std, device=device,)
                
                score = get_score(env_name, ret)
                scores.append(score)
                returns.append(ret)
                lengths.append(length)
            return {
                'target_{}_return_mean'.format(target_rew): np.mean(returns),
                'target_{}_return_std'.format(target_rew): np.std(returns),
                'target_{}_length_mean'.format(target_rew): np.mean(lengths),
                'target_{}_length_std'.format(target_rew): np.std(lengths),
                'target_{}_score_mean'.format(target_rew): np.mean(scores),
            }
        return fn
    


    if model_type == 'dt':
        model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, max_length=K, max_ep_len=max_ep_len, hidden_size=args.embed_dim, 
                                n_layer=args.n_layer, n_head=args.n_head, n_inner=4*args.embed_dim, activation_function=args.activation_function,
                                n_positions=1024, resid_pdrop=args.dropout, attn_pdrop=args.dropout, seed=args.seed, torch_deterministic=args.torch_deterministic)
        action_dist_model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, max_length=K, max_ep_len=max_ep_len, hidden_size=args.embed_dim, 
                                n_layer=args.n_layer, n_head=args.n_head, n_inner=4*args.embed_dim, activation_function=args.activation_function,
                                n_positions=1024, resid_pdrop=args.dropout, attn_pdrop=args.dropout, seed=args.seed, torch_deterministic=args.torch_deterministic)
        state_return_model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, max_length=K, max_ep_len=max_ep_len, hidden_size=args.embed_dim, 
                                n_layer=args.n_layer, n_head=args.n_head, n_inner=4*args.embed_dim, activation_function=args.activation_function,
                                n_positions=1024, resid_pdrop=args.dropout, attn_pdrop=args.dropout, seed=args.seed, torch_deterministic=args.torch_deterministic)
    elif model_type == 'bc':
        model = MLPBCModel(state_dim=state_dim, act_dim=act_dim, max_length=K, hidden_size=args.embed_dim, n_layer=args.n_layer,)
    else:
        raise NotImplementedError

    # Setup the main model
    model, optimizer, scheduler = setup_model(model, device, args)

    # Setup the augmented model
    state_return_model, state_return_model_optimizer, state_return_model_scheduler = setup_model(state_return_model, device, args)
    action_dist_model, action_dist_model_optimizer, action_dist_model_scheduler = setup_model(action_dist_model, device, args)
    dict_record_count = None

    # if args.use_data_augmentation and (args.train_augment_model is False): 
    #     if args.use_less_data:
    #         save_dir = "saved_model/{}_{}_{}/".format(env_name, args.dataset, args.percentage_less_data)
    #     else:
    #         save_dir = "saved_model/{}_{}/".format(env_name, args.dataset)

    if args.use_data_augmentation and (args.train_augment_model is False): 
        if args.use_less_data:
            save_dir = "saved_model/{}_{}_{}/".format(env_name, args.dataset, args.percentage_less_data)
        elif args.use_modify_env:
            if env_name == "hopper" or env_name == "halfcheetah":
                save_dir = "saved_model/{}_{}_{}_modified/".format(env_name, args.dataset, args.modification)
            else:
                save_dir = "saved_model/{}_{}_modified/".format(env_name, args.dataset)
        else:
            save_dir = "saved_model/{}_{}/".format(env_name, args.dataset)

        print(save_dir)
        
        dict_record_count = utils.load_dict_count(save_dir)

        action_dist_model = utils.load_action_dist_model(save_dir, device)
            
        state_return_model = utils.load_state_return_model(save_dir, device)


    if model_type == 'dt':
        trainer = SequenceTrainer(model=model, optimizer=optimizer, batch_size=batch_size, get_batch=get_batch, scheduler=scheduler, 
                                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                                eval_fns=[eval_episodes(tar, env_name) for tar in env_targets],   
                                action_dist_model=action_dist_model, action_dist_model_optimizer=action_dist_model_optimizer, action_dist_model_scheduler=action_dist_model_scheduler, 
                                state_return_model=state_return_model, state_return_model_optimizer=state_return_model_optimizer, state_return_model_scheduler=state_return_model_scheduler,
                                dict_record_count=dict_record_count, args=args)
    elif model_type == 'bc':
        trainer = ActTrainer(model=model, optimizer=optimizer, batch_size=batch_size, get_batch=get_batch, scheduler=scheduler, 
                            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2), eval_fns=[eval_episodes(tar, env_name) for tar in env_targets], 
                            dict_record_count=dict_record_count,
                            args=args)
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer_v_9',
            config= vars(args)
        )

    for iter in range(args.max_iters):
        outputs = trainer.train_iteration(iter_num=iter+1, print_logs=True,)
        if log_to_wandb:
            wandb.log(outputs)
            print("Tracking selected action \n" + str(trainer.tracking_selected_action))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ant')
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'delayed'])  # normal for standard setting, delayed for sparse reward setting
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--torch_deterministic', type=str2bool, default="True")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_to_wandb', '-w', type=str2bool, default="False")
    # Added arguements
    parser.add_argument('--use_data_augmentation', type=str2bool, default="True")
    parser.add_argument('--train_augment_model', type=str2bool, default="True")
    parser.add_argument('--aug_iter', type=int, default=1)
    parser.add_argument('--original_dt', type=str2bool, default="False")
    parser.add_argument('--max_len_aug', type=int, default=1000)
    parser.add_argument('--no_search_action', type=int, default=5)
    parser.add_argument('--step_action_search', type=int, default=0.01)
    parser.add_argument('--load_aug_data', type=str2bool, default="False")
    parser.add_argument('--sampling_aug_ratio', type=float, default=1)
    parser.add_argument('--under_loading', type=str2bool, default="False")
    parser.add_argument('--percentage_under_loading', type=int, default=70)
    # Arguemments for less data
    parser.add_argument('--use_less_data', type=str2bool, default="False")
    parser.add_argument('--percentage_less_data', type=int, default=80)
    # Arguement for modify environment
    parser.add_argument('--use_modify_env', type=str2bool, default="False")
    parser.add_argument('--modification', type=str, default='head')
    parser.add_argument('--correct_eval', type=str2bool, default="False")

    args = parser.parse_args()

    experiment('gym-experiment', args, vars(args))

