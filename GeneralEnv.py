import numpy as np
import copy
import tensorflow as tf
from gymnasium import Env, spaces
from utils import *

class GeneralEnv(Env):
    def __init__(self, dataset, affected_dataset, model, model_type, protected_attribute, features_to_change, number_of_counterfactuals, counterfactuals, target=None, minimums=None, maximums=None, action_effectiveness = None):
        super(GeneralEnv, self).__init__()
        self.dataset = dataset # The entire dataset 
        self.dataset_features = list(dataset.columns[:-1]) # All feature names of the dataset
        self.affected_dataset = affected_dataset # The instances from Test Dataset that recieved unfavorable outcome
        self.protected_attribute = protected_attribute # Protected Attribute that we want to ensure the Fair Counterfactuals for its subgroups (Ex: Gender {Male, Female})
        self.action_effectiveness = action_effectiveness # For Fairness Metric 2 (Equal Choice of Recourse) not included in version 1 
        self.model = model # The Model used on the dataset
        self.model_type = model_type # Type of the model used
        self.cf_features = features_to_change # Features (to act on) that are considered in the Counterfactual Explanations
        self.nb_cf = number_of_counterfactuals # Max Number of Counterfactual Explanations we want to generate
        self.counterfactuals = counterfactuals # A dict that contains the counterafactuals we found which satisfies the fairness metric
        self.counterfactual_set = set() # used to store counterfactuals and ensure no duplicates are saved
        self.state_dim = self.nb_cf * len(self.cf_features) # Dimension of the state will be number_of_counterfactuals * number_of_features_to_change
        self.best_episode_state = {} # Keep Track of Best State in an Epsiode to use when reseting the environment
        
        self.reward_count = 0 # used as a key to store successful states in dict self.counterfactuals
        self.first_step = 0 # used to start from a pre-set initial state
        self.total_steps = 0
        
        min_max = minMax(self.dataset[self.cf_features]) # Extract minimum and maximum values of the features_to_change from the dataset
        
        # set mins and maxs of features to change
        # either pass mins and maxs as arguments or it will be extracted from the entire dataset
        if minimums:
            self.mins = minimums
        else:
            self.mins = np.array(min_max['min'].values) # Minimum values of the features cosidered in the Counterfactual Explanations
        
        if maximums:
            self.maxs = maximums
        else:
            self.maxs = np.array(min_max['max'].values) # Maximum values of the features cosidered in the Counterfactual Explanations

        # Determine the bounds of the state
        # For example if the features to change have the following mins = [0, 0, 0] and maxs = [10, 10, 30]
        # state bounds will be state_mins = [-10, -10, -30] and state_maxs = [10, 10, 30]
        self.state_mins , self.state_maxs = self.calculate_state_bounds(self.mins, self.maxs)
        self.max_l1_norm = self.max_l1_norm(self.affected_dataset)
        
        self.features_types = [self.dataset[feature].dtype for feature in self.cf_features]
        
        # Used as a test to start the initial state from
        self.medians = [int(round((min_val + max_val) / 2)) for min_val, max_val in zip(self.mins, self.maxs)]
        
        # Action Space
        # Index 0: Represents the state index we want to change
        # Index 1: Represents the % of change to the state's index value
        self.action_space = spaces.Box(low=np.array([0,-3]), high=np.array([self.state_dim-1,3]), shape=(2,) , dtype=np.float32)
        
        # State Space
        # Ex: if we want to act on 'Income' and 'Credit' and generate a max number of 3 counterfactuals
        # the state will be of dimension (2x3 = 6) [Income, Credit, Income, Credit, Income, Credit]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # Create an initial state either random or pre-set 
        self.initial_state_cfs = self.generate_state(random=False)
        self.initial_state = GenSet("CF")
        self.initial_state.set_counterfactuals(self.initial_state_cfs)
        self.state = self.initial_state
        
        # Find the initial label or the unfavorable label of the affected dataset
        if self.model_type == 'sklearn':
            self.initial_label = self.model.predict(self.affected_dataset.iloc[0].values.reshape(1, -1))
        else:
            self.initial_label = np.argmax(self.model.predict(self.affected_dataset.iloc[0].values.reshape(1, -1)))

        #-------------------------------------DEBUGGING-----------------------------------------
        #print("----------------------------------------------")
        print(f"Action Sample: {self.action_space.sample()}")
        print(f"Max L1 Norm: {self.max_l1_norm}")
        #print("----------------------------------------------")
        print(f"State: {self.state.get_counterfactuals()}")
        print(f"Initial Label = {self.initial_label}")
        #print("----------------------------------------------")
        #---------------------------------------------------------------------------------------
        
        self.desired_label = target # Outcome that we aim to achieve using the counterafactual explanations
        self.done = False # determine if the goal reached
        self.final_reward = 1000 # give high reward to the agent when goal achieved
        self.current_step = 0 # keep track of number of steps per episode 
        self.nb_episode = 0 # used to apply per episode gradual decrease % perturbation to the reset state 
        self.steps_per_episode = 1000 # max steps per episode, if goal not reached the episode will truncate
        
    # used to calculate the minimum and maximum possible values for the state   
    def calculate_state_bounds(self, mins, maxs):
        state_mins = []
        state_maxs = []

        for i in range(len(mins)):
            result = maxs[i] - mins[i]
            state_mins.append(-result)
            state_maxs.append(result)

        return state_mins, state_maxs
    
    # used to generate either a random or pre-set state
    def generate_state(self, random=True):
        initial_state = []
        if random:
            np.random.seed(42)
            for _ in range(self.nb_cf):
                for i, feature in enumerate(self.cf_features):
                    random_value = np.random.randint(low=self.state_mins[i], high=self.state_maxs[i])
                    initial_state.append(random_value)
        else:
            initial_state = [0] * self.state_dim
               
        return initial_state
    
    #used to split the dataset into 2 subgroups based on the binary protected attribute 
    def split_dataset_by_protected_attribute(self):
        group1 = self.affected_dataset[self.affected_dataset[self.protected_attribute] == 0]
        group2 = self.affected_dataset[self.affected_dataset[self.protected_attribute] == 1]
        return group1, group2
    
    
    #Evaluation of Fairness Metric 1: Equal Effectiveness
    def evaluate_fairness_metric1(self, counterfactuals, macro=False):
         
         # Split the affected_dataset into 2 subgroups based on the protected attribute
         group1, group2 = self.split_dataset_by_protected_attribute() 
         
         # Calculate the Proportion of individuals from 2 protected subgroups who can achieve recourse using the set of counterfactuals
         group1_proportion = float(f"{self.calculate_proportion(group1, counterfactuals, macro): .2f}")
         group2_proportion = float(f"{self.calculate_proportion(group2, counterfactuals, macro): .2f}")
         
         print(f"Group 1 Proportion: {group1_proportion}")
         print(f"Group 2 Proportion: {group2_proportion}")
         
         # Calculate the average success rate
         average_success_rate = float(f"{(group1_proportion + group2_proportion) / 2: .2f}")
         print(f"Success Rate: {average_success_rate}")
         
         # Calculate the penalty based on the difference in proportions
         penalty = round(abs(group1_proportion - group2_proportion), 2)
         
         return average_success_rate, penalty
     
    # Helper function for calculating Fairness Metric 1: Equal Effectiveness
    def calculate_proportion(self, group, counterfactuals, macro=False):
        total_individuals = len(group)
        success_count = 0         
        highest_success_count = 0

        #conver dataset and counterfactuals to tensors
        group_tensor = tf.convert_to_tensor(group, dtype=tf.float32)
        counterfactuals_tensor = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)
        
        #used to determine number of individuals who achieved the desired outcome
        outcome_achieved = tf.zeros(group_tensor.shape[0], dtype=tf.bool)
        
        #macro-viewpoint considers collective impact of an action
        if macro:
            for cf in counterfactuals_tensor:
                outcome_achieved = tf.zeros(group_tensor.shape[0], dtype=tf.bool)
                modified_group = self.apply_counterfactual(group_tensor, cf)
                predictions = self.test_outcome(modified_group)
                outcome_achieved = tf.logical_or(outcome_achieved, predictions)
                success_count = tf.reduce_sum(tf.cast(outcome_achieved, tf.int32)).numpy()
                if success_count > highest_success_count:
                    highest_success_count = success_count
            return highest_success_count / total_individuals
        else: #micro-viewpoint, each individual chooses the action that best benefits itself
            for cf in counterfactuals_tensor:
                modified_group = self.apply_counterfactual(group_tensor, cf)
                #modified_group = tf.cast(modified_group, dtype=tf.int32)
                predictions = self.test_outcome(modified_group)
                outcome_achieved = tf.logical_or(outcome_achieved, predictions)
            proportion = tf.reduce_mean(tf.cast(outcome_achieved, tf.float32))
            return proportion.numpy()
    
    # Apply a single counterfactual from the set of counterfactuals to the subgroup
    def apply_counterfactual(self, group_tensor, counterfactual):

        # find indices of features to change
        feature_indices = [self.dataset_features.index(feature) for feature in self.cf_features]
        
        # filter the dataset to only features to change
        selected_features = tf.gather(group_tensor, feature_indices, axis=1)
        # apply the modifications to the dataset
        modified_features = selected_features + counterfactual
        
        # Clip the modified features to ensure they stay within the min-max bounds for each feature
        # Filter min and max values for the corresponding features
        filtered_mins = [self.mins[index] for index, value in enumerate(self.cf_features)]
        filtered_maxs = [self.maxs[index] for index, value in enumerate(self.cf_features)]
      
        # Convert to tensors for broadcasting
        filtered_mins_tensor = tf.constant(filtered_mins, dtype=group_tensor.dtype)
        filtered_maxs_tensor = tf.constant(filtered_maxs, dtype=group_tensor.dtype)
       
        # Clip the modified features to stay within the bounds
        modified_features = tf.clip_by_value(modified_features, filtered_mins_tensor, filtered_maxs_tensor)
        
        updated_group_tensor = group_tensor
    
        # Iterate over each feature index and apply the changes
        for i, feature_index in enumerate(feature_indices):
            # Gather the updates for the specific feature
            update_indices = tf.expand_dims(tf.range(tf.shape(group_tensor)[0]), axis=1)
            update_indices = tf.concat([update_indices, tf.fill([tf.shape(group_tensor)[0], 1], feature_index)], axis=1)
            
            # Update the group tensor with the modified feature
            updated_group_tensor = tf.tensor_scatter_nd_update(
                updated_group_tensor,   
                update_indices, 
                modified_features[:, i]
            )

        return updated_group_tensor
    
    # used to test the outcome of individuals
    def test_outcome(self, group_tensor):
        
        if len(group_tensor.shape) == 1:
            group_tensor = tf.reshape(group_tensor, (1, -1))
        if self.model_type == 'sklearn':
            predictions = self.model.predict(pd.DataFrame(group_tensor.numpy(), columns=self.dataset_features))
            if self.desired_label:
                return predictions == self.desired_label
            return predictions != self.initial_label
        else:
            predictions = tf.argmax(self.model(group_tensor), axis=1)
            if self.desired_label:
                return predictions == self.desired_label
            return predictions != self.initial_label
    
    # used to compute l1_norm for the affected dataset (including both protected subgroups) after applying counterfactuals
    # used to ensure minimal feature changes
    def compute_individuals_l1norm(self, dataset, counterfactuals):
        
        l1s = [] # used to store l1_norm of individuals for each counterfactual
        
        #convert dataset and set of counterfactuals to tensors
        initial_dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
        counterfactuals = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)
        
        for cf in counterfactuals:
            modified_dataset_tensor = self.apply_counterfactual(initial_dataset_tensor, cf) # apply counterfactual to the dataset  
            modified_dataset_tensor = tf.cast(modified_dataset_tensor, dtype=tf.float32) 
            difference = tf.abs(initial_dataset_tensor - modified_dataset_tensor) # calculate difference between initial dataset and modified dataset
            l1_norms = tf.reduce_sum(difference, axis=1).numpy() # sum the result of difference
            l1s.append(tf.reduce_mean(l1_norms).numpy())

        return float(f"{np.mean(l1s): .2f}")
    
    def max_l1_norm(self, dataset):
        max_state = self.state_maxs * self.nb_cf
        min_state = self.state_mins * self.nb_cf
        
        max_state_l1_norm = self.compute_individuals_l1norm(dataset, max_state)
        min_state_l1_norm = self.compute_individuals_l1norm(dataset, min_state)
        
        if max_state_l1_norm > min_state_l1_norm:
            return max_state_l1_norm
        return min_state_l1_norm
    
    #Evaluation of Fairness Metric 2: Equal Choice of Recourse (Not included in version 1)
    def evaluate_fairness_metric2(self, counterfactuals, effectiveness):
        
        # Split the affected_dataset into 2 subgroups based on the protected attribute
         group1, group2 = self.split_dataset_by_protected_attribute()
         
         # Calculate the number of effective actions (counterfactuals) for the 2 protected subgroups 
         # If the counterfactual achieve recourse for a proportion >= effectiveness then we count it
         nb_actions_group1 = self.calculate_actions(group1, counterfactuals, effectiveness)
         nb_actions_group2 = self.calculate_actions(group2, counterfactuals, effectiveness)
        
         print(f"Number of Actions for Group 1: {nb_actions_group1}")
         print(f"Number of Actions for Group 2: {nb_actions_group2} ")
         
         average_nb_actions = nb_actions_group1 + nb_actions_group2 / 2
         penalty = abs(nb_actions_group1 - nb_actions_group2)
         return average_nb_actions, penalty
     
    # Helper Function to calculate Fairness Metric 2
    def calculate_actions(self, group, counterfactuals, effectiveness):
        total_individuals = len(group)
        nb_of_effective_actions = 0 # An action (counterfactual) is effective if it achieves recourse for a proportion pf individuals >= effectiveness 
        
        #conver dataset and counterfactuals to tensors
        group_tensor = tf.convert_to_tensor(group, dtype=tf.float32)
        counterfactuals_tensor = tf.convert_to_tensor(counterfactuals, dtype=tf.float32)
        
        for cf in counterfactuals_tensor:
            outcome_achieved = tf.zeros(group_tensor.shape[0], dtype=tf.bool)
            modified_group = self.apply_counterfactual(group_tensor, cf)
            predictions = self.test_outcome(modified_group)
            outcome_achieved = tf.logical_or(outcome_achieved, predictions)
            success_count = tf.reduce_sum(tf.cast(outcome_achieved, tf.int32)).numpy()
            if success_count / total_individuals >= effectiveness:
                nb_of_effective_actions += 1
        return nb_of_effective_actions     
    
    def reset(self, seed=None, options=None):
        
        self.current_step = 0
        
        self.nb_episode += 1
        #perturbation_percentage = max(1, 20 - (self.nb_episode // 100))
        
        #used to ensure we start from the pre-set state (initial state that we want to start from)
        if self.first_step == 0:
            self.first_step +=1
            print("$$$$$$$$$$$$$$$$$$$")
            print(f"RESET STATE: {self.state.get_counterfactuals()}")
            print("-----ENV RESET-----")
            print("$$$$$$$$$$$$$$$$$$$")
            return np.array(self.state.get_counterfactuals()), {}
        
        # get the best state from previous episode
        old_state = list(self.best_episode_state.values())[0]
        
        if self.done:
            # if goal reached, we get the best state from previous episode and apply [-10%, 10%] perturbation to it and reset the environment
            old_state = np.array(list(self.best_episode_state.values())[0])
            perturbation_percentage = np.random.randint(-10, 10, size=self.observation_space.shape)/100
            perturbation = old_state * perturbation_percentage
            state = old_state + perturbation
            state = [np.clip(val, self.state_mins[i % len(self.state_mins)], self.state_maxs[i % len(self.state_maxs)]) for i, val in enumerate(state)]
            state = [round(val) if self.features_types[i % len(self.features_types)] == 'int' else val for i, val in enumerate(state)]
        else:
            state = old_state # if goal not reached, we reset the environment to the best state reached in previous episode without applying perturbation
        
        new_state = GenSet("CF")
        new_state.set_counterfactuals(state)
        self.state = new_state
        self.best_episode_state.clear()
        print("$$$$$$$$$$$$$$$$$$$")
        print(f"BEFORE RESET STATE: {old_state}")
        print(f"RESET STATE: {self.state.get_counterfactuals()}")
        print("-----ENV RESET-----")
        print("$$$$$$$$$$$$$$$$$$$")
        return np.array(self.state.get_counterfactuals()), {}
    
    def _custom_rewrad(self, new_state, macro=False): # Reward Function 
        reward = 0
        cf_reward = 0 # same as 'reward' but doesn't include step penalty, used only as a key to store successful sets of CFS
        done = False
        target_success_rate = 0.8 #(Compas-0.77) #(Alzheimer-0.79) # Target average success rate to stop the episode
        target_proportion_difference = 0.09 #(Compas-0.09) #(Alzheimer-0.09) # Target penalty to stop the episode (difference between proportions)
        
        #Generate a list of counterfactuals from the state
        #Ex: if features_to_change = ["Income", "Credit", "Experience"] and number_of_counterfactuals = 2
        #A possible State Representation is [1000, 30, 1, 1200, 25, 1]
        #counterfactuals = [[1000, 30, 1], [1200, 25, 1]]
        cfs = new_state.get_counterfactuals()
        counterfactuals = [cfs[i * len(self.cf_features):(i+1) * len(self.cf_features)] for i in range(self.nb_cf)]
        print(f"counterfactuals listed {counterfactuals}")
        
        # Fairness Metric 1 'Equal Effectiveness' : Average Success Rate with Penalty for Wide Proportion Differences
        # avergage_success_rate has a range [0, 1]
        # proportions_difference has a range [0, 1]
        average_success_rate, proportions_difference = self.evaluate_fairness_metric1(counterfactuals, macro)
        print("-----------------------------")
        print(f"Group Difference: {proportions_difference}")
        print("-----------------------------")
        
        # l1_norm_mean to penalize high deviated values
        l1_norm = self.compute_individuals_l1norm(self.affected_dataset, counterfactuals)
        normalized_l1_norm = l1_norm/self.max_l1_norm 
        
        reward += (average_success_rate * 1.1 - proportions_difference*0.9 - normalized_l1_norm * 0.9) * 100 #(asr=1.2-Compas)
        cf_reward = round(reward,2)
        # Penalize Agent for taking more steps
        step_penalty = (-0.2 * self.current_step/self.steps_per_episode)*100
        reward += step_penalty
        
        print("-----------------------------")
        print(f"l1_norm: {l1_norm}")
        print("-----------------------------") 
              
        # Determine if the episode is done
        done = (average_success_rate >= target_success_rate) and (proportions_difference <= target_proportion_difference)  
        if done:
            reward += self.final_reward
            cf_reward += self.final_reward
            print("****************************************************")
            print("****************************************************")
            print("**********************DONE**************************")
            print("****************************************************")
            print("****************************************************")
            self.reward_count += 1
            # store successful CF set
            self.add_counterfactual(cf_reward, self.reward_count, average_success_rate, proportions_difference, l1_norm, counterfactuals)
       
        return reward, done  
    
    #----------------------------------Helper Functions to Store Successful CF Set-----------------------------------------  
    def add_counterfactual(self, cf_reward, reward_count, average_success_rate, proportions_difference, l1_norm, counterfactuals):
        
        modified_reward = f"{cf_reward}-{reward_count}({average_success_rate})({proportions_difference})({l1_norm})"
        
        new_counterfactuals_tuple = self.convert_to_tuple(counterfactuals)
        # Check if the new counterfactual tuple already exists in the set
        if new_counterfactuals_tuple in self.counterfactual_set:
            return
        
        # If it's unique, add it to the dictionary and the set
        self.counterfactuals[modified_reward] = counterfactuals
        self.counterfactual_set.add(new_counterfactuals_tuple)

    def convert_to_tuple(self, counterfactuals):
        return tuple(tuple(sublist) for sublist in counterfactuals)
    #----------------------------------Helper Functions to Store Successful CF Set-----------------------------------------  
     
    # used to get the successful sets of CFs at the end of training   
    def return_counterfactuals(self):
        return self.counterfactuals
    
    def _take_action(self, action):
        values = self.state.get_counterfactuals()
        new_state = GenSet(self.state.name)
        new_state.set_counterfactuals(copy.deepcopy(values))
        
        index_to_change = round(action[0]) 
        amount_of_change = action[1] #round(action[1], 2)
        
        info = {}
        
        min_val = self.state_mins[index_to_change % len(self.state_mins)]
        max_val = self.state_maxs[index_to_change % len(self.state_maxs)]
        feature_type = self.features_types[index_to_change % len(self.features_types)]
        
        old_value = values[index_to_change]
        un_normalized_change = ((amount_of_change + 100) / 200) * (max_val - min_val) + min_val
        
        # modify state index value (index_to_change) by a specific % (amount_of_change)
        new_state.modify_feature(index_to_change, amount_of_change, min_val, max_val, feature_type)
        info["info"] = 'Modifying old value of {' + str(self.cf_features[index_to_change%len(self.cf_features)]) + '} Feature - Index: ' + str(index_to_change) + ' by amount = ' + str(un_normalized_change)
            
        # Keep the counterfactual bounded
        #if new_state.counterfactuals[index_to_change] >= self.state_maxs[index_to_change % len(self.cf_features)] : 
            #new_state.counterfactuals[index_to_change] = self.state_maxs[index_to_change % len(self.cf_features)]
            
        #if new_state.counterfactuals[index_to_change] <=  self.state_mins[index_to_change % len(self.cf_features)] :
            #new_state.counterfactuals[index_to_change] = self.state_mins[index_to_change % len(self.cf_features)]
        
        #if new_state.counterfactuals[index_to_change] == 0 and old_value < 0:
            #new_state.counterfactuals[index_to_change] = 0.1
        #elif new_state.counterfactuals[index_to_change] == 0 and old_value > 0:
            #new_state.counterfactuals[index_to_change] = -0.1
        
        print("Old State: " + str(self.state.get_counterfactuals()))
        print("----------------------------------------------------")
        print(info["info"])
        print("----------------------------------------------------")
        print("New State: " + str(new_state.get_counterfactuals()))
        print("----------------------------------------------------")
        
        reward, done = self._custom_rewrad(new_state)
            
        print("***---------------************-----------------***") 
        print("Reward: " + str(reward))
        print("Done: " + str(done)) 
        print("***---------------************-----------------***")  
        print("####################################################")
        return new_state, reward, done, info
    
    def step(self, action):
        # Execute one time step within the environment
        terminated = False
        truncated = False
        
        self.current_step += 1
        self.total_steps +=1
        new_state, reward, self.done, info = self._take_action(action)
        
        # Keep track of best state in the episode
        if not self.best_episode_state:
            self.best_episode_state[reward] = new_state.get_counterfactuals()
        elif reward > list(self.best_episode_state.keys())[0]:
            self.best_episode_state.clear()
            self.best_episode_state[reward] = new_state.get_counterfactuals()
        
        if self.done == True:
            terminated = True
            print("**********(Terminated)**********")
           
        if self.current_step > self.steps_per_episode and not terminated:
            truncated = True
            print("***********(Truncated)***********")

    
        self.state = new_state
        return np.array(new_state.get_counterfactuals()), reward, terminated, truncated, info
    
        
        