"""
Evaluate performance of trained nn-ensemble for environment drift detection on a specific environment
  The configuration of the Neural Network is based on the work of Haider et al (2023)
    can be found at https://github.com/FraunhoferIKS/pedm-ood
"""

import numpy as np 

from scipy.stats import multivariate_normal

from sklearn.metrics import roc_auc_score 

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, DQN 

import pickle

import os 
import glob
from datetime import datetime 
import json 

from make_envs import *

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchensemble import BaggingRegressor
from torchensemble.utils import io 

import argparse


def sample_and_compute_mse(ensemble, x, y_true, n_samples=200):
    """
    sample state increment from the NN ensemble and compute the average prediction error
    args:
      ensemble: Ensemble of NNs predicting increment of states
      x: input state and action (numpy array)
      y_true: state increment
      n_samples: number of samples to draw from the distribution

    """
    x = torch.from_numpy(x).float()

    ensemble.eval()
    mse_all_estimators = []

    # Get predictions from each estimator in the ensemble
    for estimator in ensemble.estimators_:
        with torch.no_grad():
            output = estimator(x)
        dim = output.shape[-1] // 2
        mu, var = output[:, :dim], output[:, dim:]
        sampled_predictions = multivariate_normal.rvs(mean=mu.detach().numpy().flatten(),
                                                      cov=var.detach().numpy().flatten(),
                                                       size=n_samples)
        # Compute the MSE for each sample
        mse_samples = np.mean((sampled_predictions - y_true) ** 2, axis=1)  # MSE per sample
        mse_mean = np.mean(mse_samples)  # Average MSE across all samples
        mse_all_estimators.append(mse_mean)

    # Compute the average MSE across all estimators
    average_mse = np.mean(mse_all_estimators)

    return average_mse



parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="acrobot", help="name of environment")
parser.add_argument("--env1-steps", type=int, default=3000, help="Undrifted Steps")
parser.add_argument("--env2-steps", type=int, default=3000, help="Drifted Steps")
parser.add_argument("--n-exp-per-model", type=int, default=10, 
                        help="number of experiments of each trained model.") 
args = parser.parse_args() 
allowed_envs = {"acrobot", "cartpole", "lunarlander", "mountaincar", 
                    "mountaincar_continuous", "pendulum"}
if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")
    
print("Parsed arguments") 
print(args) 

## Configuration of NNs for corresponding environment
if args.env == "acrobot":
    n_in = 7
    n_output = 6 
elif args.env == "cartpole":
    n_in = 5
    n_output = 4 
elif args.env == "lunarlander":
    n_in = 9
    n_output = 8 
elif args.env == "mountaincar":
    n_in = 3
    n_output = 2 
elif args.env == "mountaincar_continuous":
    n_in = 3
    n_output = 2 
elif args.env == "pendulum":
    n_in = 4
    n_output = 3 

# Define Base Neural Network
class BaseNN(nn.Module):
    def __init__(self, input_dim=n_in, hidden_dim=500, output_dim=n_output):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        var = torch.exp(log_var)
        output = torch.cat([mu, var], dim=-1)
        return output

# Multivariate Gaussian Loss   
class MultivariateGaussianNLLLossCustom(nn.Module):
    def __init__(self, reduction='mean'):
        super(MultivariateGaussianNLLLossCustom, self).__init__()
        self.reduction = reduction

    def forward(self, output, y):
        # Ensure variance is positive for each dimension
        dim = output.shape[-1] // 2
        mu, var = output[:, :dim], output[:, dim:]
        var = torch.clamp(var, min=1e-6)  # To avoid division by zero or log of zero

        # Compute the negative log-likelihood for each dimension and sum across dimensions
        nll = 0.5 * ((y - mu) ** 2 / var + torch.log(var))

        # Sum across the output dimensions (i.e., features)
        nll = torch.sum(nll, dim=-1)

        # Apply the reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll

loaded_models = [] 
model_folder = os.path.join('.', "experiments", args.env, "trained_models")
model_pattern = os.path.join(model_folder, 
                                      f"nn_ensemble_*")
matching_models = glob.glob(model_pattern)
print(matching_models)
if len(matching_models)==0:
    raise NotImplementedError(f"There is no trained NN ensemble for the environment {args.env}.")

        
for model_path in matching_models:
    scaler_path = os.path.join(model_path, "scaler.pkl")
    #nn_ensemble_path = os.path.join(model_path, "BaggingRegressor_BaseNN_5_ckpt.pth")
    print("scaler path: ", scaler_path)
    #print("NN Ensemble path: ", nn_ensemble_path) 
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f) 
        ## Ensemble of Neural Networks
    ensemble = BaggingRegressor(
            estimator=BaseNN,
            n_estimators=5,
            cuda=False
        )

       
    io.load(ensemble, model_path)

    loaded_models.append((scaler, ensemble))
    
print(f"Number of trained models: {len(loaded_models)}")
        


## Create undrifted environment env1 and drifted environment env2
## and load corresponding trained agents
if args.env == "acrobot":
    env1, env2 = make_acrobot()
    checkpoint = load_from_hub(
        repo_id = "sb3/ppo-Acrobot-v1",
        filename = "ppo-Acrobot-v1.zip",
    )
    agent = PPO.load(checkpoint)
elif args.env == "cartpole":
    env1, env2 = make_cartpole()
    checkpoint = load_from_hub(
            repo_id = "sb3/dqn-CartPole-v1",
            filename = "dqn-CartPole-v1.zip",
        )
    agent = DQN.load(checkpoint)
elif args.env == "lunarlander":
    env1, env2 = make_lunarlander()
    checkpoint = load_from_hub(
            repo_id = "sb3/ppo-LunarLander-v2",
            filename = "ppo-LunarLander-v2.zip",
    )
    agent = PPO.load(checkpoint)
elif args.env == "mountaincar":
    env1, env2 = make_mountaincar()
    checkpoint = load_from_hub(
            repo_id = "sb3/dqn-MountainCar-v0",
            filename = "dqn-MountainCar-v0.zip",
    )
    agent = DQN.load(checkpoint)
elif args.env == "mountaincar_continuous":
    env1, env2 = make_mountaincar_continuous()
    checkpoint = load_from_hub(
          repo_id = "sb3/sac-MountainCarContinuous-v0",
          filename = "sac-MountainCarContinuous-v0.zip",
    )
    agent = SAC.load(checkpoint) 
else:
    env1, env2 = make_pendulum() 
    checkpoint = load_from_hub(
         repo_id = "sb3/sac-Pendulum-v1",
         filename = "sac-Pendulum-v1.zip",
    )
    agent = SAC.load(checkpoint)


result = dict() # A dictionary to store the results
for i in range(len(loaded_models)):
    result[f"nn_ensemble_{i}"] = dict()

# Run the evaluations 

for i, model in enumerate(loaded_models):
    for j in range(args.n_exp_per_model):
        total_steps = args.env1_steps+args.env2_steps
        scores = []
        env_current = env1 
        obs_t, _ = env_current.reset() 

        for t in range(1, total_steps+1):
            if t%1000 == 0:
                print(f"model {i}, experiment {j}, step {t}")
            action_t, _state = agent.predict(obs_t, deterministic=True)
            obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)   

            x = np.concatenate((obs_t, action_t.reshape(-1)))
            x = model[0].transform(x.reshape(1, -1))
            y = obs_tplus1-obs_t
            mse = sample_and_compute_mse(model[1], x, y, n_samples=200)
            scores.append(mse)

            done = terminated or truncated

            obs_t = obs_tplus1

            if done:
               obs_t, _ = env_current.reset()
            if t==args.env1_steps:
                env_current = env2
                obs_t, _ = env_current.reset()

        scores_drift = np.array(scores)

        # compute AUC
        y_env1 = np.zeros(3000)
        y_env2 = np.ones(3000)
        y = np.concatenate([y_env1, y_env2])
        auc = roc_auc_score(y, scores_drift)

        # compute Moving Average of 100 steps and Corresponding AUC
        scores_drift_ma = np.convolve(scores_drift, np.ones(100)/100, mode='valid')
        y_env1 = np.zeros(2901)
        y_env2 = np.ones(3000)
        y = np.concatenate([y_env1, y_env2])
        auc_ma = roc_auc_score(y, scores_drift_ma)

        result[f"nn_ensemble_{i}"][f"exp_{j}"] = {"scores":scores_drift.tolist(),
                                                 "auc":auc,
                                                 "scores_ma":scores_drift_ma.tolist(),
                                                 "auc_ma":auc_ma}
       

        
result_folder = os.path.join('.',"experiments", args.env, "results")
if not os.path.exists(result_folder):
    os.makedirs(result_folder) 

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_file = f"nn_ensemble-{args.env}-{current_time}.json"


print("result file: ", result_file)

result_path = os.path.join(result_folder, result_file) 

print("result path: ", result_path) 

with open(result_path, 'w') as f:
    json.dump(result, f, separators=(',', ':'))



