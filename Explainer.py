import os
import numpy as np
import pandas as pd
import re
from utils import *
from GeneralEnv import *
from CustomCallbacks import *
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DDPG, TD3, SAC, A2C, PPO
from sb3_contrib import ARS, TQC, TRPO

class Explainer:
    def __init__(self, dataset, affected_dataset, model, protected_attribute, features_to_change, number_of_counterfactuals, target=None, minimums=None, maximums=None, action_effectiveness=None, actor_critic = 'SAC', timesteps = 10000, model_type='sklearn'):
        # Create log dir
        self.log_dir = "./tmp/gym/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.counterfactuals = {}
        self.time_to_first_cf_set = []
        self.algorithm = actor_critic
        
        self.env = Monitor(GeneralEnv(dataset, affected_dataset, model, model_type, protected_attribute, features_to_change, number_of_counterfactuals, self.counterfactuals, target, minimums, maximums, action_effectiveness), self.log_dir)
        # Create the callback: check every 1000 steps
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=self.log_dir)
        self.env.dataset = dataset
        self.env.affected_dataset = affected_dataset
        self.env.model = model
        self.env.protected_attribute = protected_attribute
        self.env.action_effectiveness = action_effectiveness
        self.env.cf_features = features_to_change
        self.env.nb_cf = number_of_counterfactuals
        self.env.state_dim = self.env.nb_cf * len(self.env.cf_features)
        self.env.model_type = model_type
        self.env.desired_label = target
        self.classifier = model
        self.state_size = self.env.state_dim
        self.action_size = 3
        self.timesteps = timesteps
        self.model = None
    
    def train(self):
        policy = 'MlpPolicy'
        policy_kwargs = dict(net_arch=dict(pi=[32, 64, 64, 128], qf=[300, 200, 100])) #dict(net_arch=dict(pi=[32, 64, 128], qf=[100, 200, 300]))
        if self.algorithm == 'SAC':                                                                                                                            
            model = SAC(policy, self.env, tensorboard_log="./counterfactuals_/", policy_kwargs=policy_kwargs, batch_size = 256, buffer_size = 1000000, verbose=1) #learning_rate=0.0003, learning_starts=50, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef='auto_0.1', target_entropy='auto'
        elif self.algorithm == 'DDPG':
            model = DDPG(policy, self.env,  tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda')
        elif self.algorithm == 'TD3':
            model = TD3(policy, self.env, tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda')
        elif self.algorithm == 'A2C':
            model = A2C(policy, self.env, tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda') 
        elif self.algorithm == 'PPO':
            model = PPO(policy, self.env, tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda')
        elif self.algorithm == 'ARS':
            model = ARS("MlpPolicy", self.env, tensorboard_log="./counterfactuals_/", verbose=1, device='cuda')
        elif self.algorithm == 'TQC':
            model = TQC('MlpPolicy', self.env, tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda')
        elif self.algorithm == 'TRPO':
            model = TRPO('MlpPolicy', self.env, tensorboard_log="./counterfactuals_/" , verbose=1, device='cuda')
        
        
        model.learn(total_timesteps=self.timesteps, log_interval=1, callback=self.callback)
        env = model.get_env()
        self.model = model
        self.env = env
    
    def predict(self):
        print("---Predicting---")
        obs = self.env.reset()
        done = False
        while not done:
            action, states = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            print(obs)
    
    def plot(self):
        results_plotter.plot_results([self.log_dir], self.timesteps, results_plotter.X_TIMESTEPS, "Explainer")
        return self.plot_results(self.log_dir)
    
    def moving_average(self, values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')
    
    def plot_results(self, log_folder, title='Learning Curve'):
      """
      plot the results

      :param log_folder: (str) the save location of the results to plot
      :param title: (str) the title of the task to plot
      """
      x, y = ts2xy(load_results(log_folder), 'timesteps')
      y = self.moving_average(y, window=50)
      
      # Truncate x
      x = x[len(x) - len(y):]
      return (x, y)
      #fig = plt.figure(title)
      #plt.plot(x, y)
      #plt.xlabel('Number of Timesteps')
      #plt.ylabel('Rewards')
      #plt.title(title + " Smoothed")
      #plt.show()
      
    def report_counterfactuals(self):
        # Prepare the header with an empty column between counterfactuals
        header = []
        # Loop over the number of counterfactual sets in the first dictionary
        
        num_cf_sets = self.env.envs[0].unwrapped.nb_cf
        
        for _ in range(num_cf_sets):
            cf_features = self.env.envs[0].unwrapped.cf_features
            header.extend(cf_features)
            header.append(" ") # Add an empty column between counterfactuals
        header[-1] = "Reward" # Replace the last empty column with "Reward" 
        header.append('Average Success Rate')
        header.append('Proportions Difference')
        header.append('L1 Norm')
        
        # Prepare the rows
        rows = []
        counterfactuals = self.env.envs[0].unwrapped.return_counterfactuals()

        #for cf_dict in counterfactuals:
            #for key, cf in cf_dict.items():
        for key, cf in counterfactuals.items():       
            row = []
            match = re.match(r"^([^-]+)-\d+\(([^)]+)\)\(([^)]+)\)\(([^)]+)\)$", key)
                
            if match:
                reward = match.group(1)  # '8567.532'
                average_success_rate = match.group(2)  # '0.75'
                proportions_difference = match.group(3)  # '0.9'
                l1_norm = match.group(4) # '9.45'
                
            for sublist in cf:
                row.extend(sublist)
                row.append(" ") # Add an empty column between counterfactuals
            row[-1] = reward # Replace the last empty column with the reward
            row.append(average_success_rate)
            row.append(proportions_difference)
            row.append(l1_norm)
            rows.append(row)
        
        # Create the dataframe
        df = pd.DataFrame(rows, columns=header) 
        
        return df
        
      
    def detect_convergence(self, patience, threshold):
      """
      Detects when an increasing curve has converged based on the patience and threshold parameters.

      Parameters:
          patience (int): The number of iterations to wait for convergence.
          threshold (float): The minimum improvement required for convergence.

      Returns:
          (int): The iteration number when convergence was detected, or -1 if convergence was not detected.
      """
      x, y = self.plot_results(self.log_dir)
      best_value_x = -float("inf")
      best_value_y = -float('inf')
      wait_count = 0

      for i, value in enumerate(y):
          if value > best_value_y:
              best_value_y = value
              best_value_x = x[i]
              wait_count = 0
          else:
              wait_count += 1
              if wait_count >= patience and best_value_y - value < threshold:
                  return x[i - patience + 1]

      return -1
