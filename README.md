# CRDT Gym Experiments

CRDT is a framework that enhances Decision Transformers (DT) using counterfactual reasoning to improve offline reinforcement learning. By generating and leveraging counterfactual experiences, CRDT enables better decision-making, particularly in settings with limited or suboptimal data.

## 📦 Installation

## Step 1: Install MuJoCo

Follow the instructions in the [mujoco-py GitHub repository](https://github.com/openai/mujoco-py) to install MuJoCo and `mujoco-py`.

## Step 2: Create Conda Environment

Install dependencies using the provided environment file:

```bash
conda env create -f conda_env.yml
conda activate dt_mujoco
```

```
conda env create -f conda_env.yml
```

## 📁 Downloading datasets

Datasets are located in the `data` directory.
Install the [D4RL repository](https://github.com/rail-berkeley/d4rl) by following the instructions provided there.
Then, execute the following script to download the datasets and save them in the required format:

```
python download_d4rl_datasets.py
```

## 🚀 Example usage

Experiments can be reproduced with the following:

1. Run with 2 parameters: `--use_data_augmentation` and `--train_augment_model` set to `True`, to train the **Counterfactual Models**

```
python gym/run_dt_normal.py \
  --env hopper \
  --dataset medium \
  --use_data_augmentation True \
  --train_augment_model True \
  --num_steps_per_iter 10000 \
  --seed 1 \
  --aug_iter 1 \
  --max_len_aug 4000 \
  --device cuda \
  --no_search_action 
```

2. Run with `--use_data_augmentation` set to `True` and `--train_augment_model` set to `False`, to train the **Model** with counterfactual samples.
```
python gym/run_dt_normal.py \
  --env hopper \
  --dataset medium \
  --use_data_augmentation True \
  --train_augment_model False \
  --num_steps_per_iter 10000 \
  --seed 1 \
  --aug_iter 1 \
  --max_len_aug 4000 \
  --device cuda \
  --no_search_action 3 
```

3.  (Optional) Enable Logging with Weights & Biases
Adding `-w True` will log results to Weights and Biases.

## 📄 Citation
If you use this work, please cite our paper:
```
Nguyen, M. H., Van, L. L. P., Karimpanal, T. G., Gupta, S., & Le, H. (2025). Beyond the Known: Decision Making with Counterfactual Reasoning Decision Transformer. arXiv preprint arXiv:2505.09114.
```









