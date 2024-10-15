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
        



