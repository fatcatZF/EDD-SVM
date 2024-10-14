"""
Evaluate performance of trained SVM for environment drift detection on a specific environment
  FSVM: SVM trained on fabricated (synthetic) drifted examples and undrifted examples
  OSVM: One-Class SVM trained solely on undrifted examples for Anomaly Detection
  SSVM: SVM trained on real drifted examples and undrifted examples
"""

import numpy as np 

from sklearn.pipeline import Pipeline 
from sklearn.metrics import roc_auc_score 

import gymnasium as gym 
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, DQN 
from stable_baselines3.common.evaluation import evaluate_policy


import pickle 

import os 
import glob 
from datetime import datetime 
import json 

from make_envs import * 

import argparse 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="acrobot", help="name of environment")
    parser.add_argument("--env1-steps", type=int, default=3000, help="Undrifted Steps")
    parser.add_argument("--env2-steps", type=int, default=3000, help="Drifted Steps")
    parser.add_argument("--svm", type=str, default="fsvm", help="type of SVM")
    parser.add_argument("--n-exp-per-model", type=int, default=10, 
                        help="number of experiments of each trained model.") 

    args = parser.parse_args() 

    allowed_envs = {"acrobot", "cartpole", "lunarlander", "mountaincar", 
                    "mountaincar_continuous", "pendulum"}
    if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")

    print("Parsed arguments: ")
    print(args) 
    
    loaded_models = []
    model_folder = os.path.join('.', "experiments", args.env, "trained_models")
    model_file_pattern = os.path.join(model_folder, f"pipeline_{args.svm}_*.pkl")
    matching_models = glob.glob(model_file_pattern)
    if len(matching_models)==0:
        print("No available trained models")
        return 
    for model_path in matching_models:
        with open(model_path, 'rb') as f:
            loaded_models.append(pickle.load(f))
    print(f"Number of trained models: {len(loaded_models)}")
    #print(f"Type of trained models: {type(loaded_models[0])}")

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
        
    mean_reward, std_reward = evaluate_policy(agent, env1, n_eval_episodes=10, deterministic=True)
    print(f"Mean Reward: {mean_reward}, std_reward: {std_reward}")



    result = dict() # A dictionary to store the results
    for i in range(len(loaded_models)):
        result[f"{args.svm}_{i}"] = dict() 

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
                transition = np.concatenate([obs_t, obs_tplus1-obs_t]).reshape(1,-1)
                x = np.concatenate([transition, action_t.reshape(1,-1)], axis=1)
                score = -model.decision_function(x)[0] # Anomaly Score
                scores.append(score) 

                done = terminated or truncated

                obs_t = obs_tplus1
                if done:
                    obs_t, _ = env_current.reset()
                if t==args.env1_steps: ## Environment Drift happens 
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

            result[f"{args.svm}_{i}"][f"exp_{j}"] = {"scores":scores_drift.tolist(),
                                                     "auc":auc,
                                                     "scores_ma":scores_drift_ma.tolist(),
                                                     "auc_ma":auc_ma}
            




    result_folder = os.path.join('.',"experiments", args.env, "results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 

    print("results folder", result_folder)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    result_file = f"{args.svm}-{args.env}-{current_time}.json"

    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file) 

    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))


    




if __name__ == "__main__":
    main()