# -*- coding: utf-8 -*-
"""
A Policy Gradient Algorithm
Author: W.J.A. van Heeswijk
Date: 15-4-2020
This code is supplemental to the following paper:
'Smart Containers With Bidding Capacity: 
A Policy Gradient Algorithm for Semi-Cooperative Learning'
https://arxiv.org/abs/2005.00565 (2020)
International Conference on Computational Logistics 2020 (to appear)
This code has been published under the GNU GPLv3 license
"""

import numpy as np
import scipy.stats as sc

class Instance:
    """Define performance metrics"""
    def __init__(self, transport_capacity, max_number_jobs, max_due_date, \
                 max_distance, holding_cost, penalty_failed, costs_per_mile, \
                 training_time_horizon, validation_time_horizon, number_of_episodes, \
                 alpha_mu, alpha_sigma, sharing_percentage, initial_st_dev):
        self.transport_capacity = transport_capacity
        self.max_number_jobs = max_number_jobs
        self.max_due_date = max_due_date
        self.max_distance = max_distance
        self.holding_cost = holding_cost
        self.penalty_failed = penalty_failed
        self.costs_per_mile = costs_per_mile
        self.training_time_horizon = training_time_horizon
        self.validation_time_horizon = validation_time_horizon
        self.number_of_episodes = number_of_episodes 
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.sharing_percentage = sharing_percentage
        self.initial_st_dev = initial_st_dev

class Features:
     """Define features"""
     def __init__(self, scalar, average_distance, total_volume, average_due_date, \
                  number_of_jobs, volume, due_date, distance):
         self.scalar = scalar
         self.average_distance = average_distance
         self.total_volume = total_volume
         self.average_due_date = average_due_date 
         self.number_of_jobs = number_of_jobs
         self.volume = volume
         self.due_date = due_date
         self.distance = distance

class Job:
    """Define job properties"""
    def __init__(self, volume, due_date, ori_due_date, distance, bid, \
                 net_value, shipped, rewards, cumulative_rewards, stored_actions, \
                stored_gradients, stored_mu, sharing_information):
        self.volume = volume
        self.due_date = due_date
        self.ori_due_date = ori_due_date
        self.distance = distance
        self.bid = bid
        self.net_value = net_value
        self.shipped = shipped
        self.rewards = rewards
        self.cumulative_rewards = cumulative_rewards
        self.stored_actions = stored_actions
        self.stored_gradients = stored_gradients
        self.stored_mu = stored_mu
        self.sharing_information = sharing_information
        
def SetInstanceProperties():
    """Set instance properties"""
    instance = Instance(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    instance.transport_capacity = 80
    instance.max_number_jobs = 10
    instance.max_due_date = 5 
    instance.max_distance = 100 
    instance.holding_cost = 1
    instance.penalty_failed = 20
    instance.costs_per_mile = 0.1 
    instance.training_time_horizon = 100
    instance.validation_time_horizon = 1000         
    instance.number_of_training_episodes = 4000
    instance.number_of_validation_episodes = 1
    instance.alpha_mu = 0.1
    instance.alpha_sigma  = 0.01
    instance.sharing_percentage = 100
    instance.initial_st_dev = 10
        
    return instance

def GenerateJobs(instance, all_jobs, sharing_percentage):
    """Generate new jobs"""
    NumberOfNewJobs = np.random.randint(instance.max_number_jobs + 1)
    
    for i in range(NumberOfNewJobs):
        # Generate job properties
        job = Job(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)
        job.volume = np.random.randint(1, 10 + 1) 
        job.due_date = np.random.randint(1, instance.max_due_date + 1)
        job.ori_due_date = job.due_date
        job.distance = np.random.randint(10, instance.max_distance + 1)
        job.bid = 0.00
        job.net_value = 0.00
        job.shipped = False
        job.rewards = []
        job.cumulative_rewards = []
        job.stored_actions = []
        job.stored_gradients = []
        job.stored_mu = []
        
        random_number_sharing = np.random.randint(1, 100 + 1)
        if random_number_sharing > sharing_percentage:
            job.sharing_information = False 
        
        all_jobs.append(job)
    
    return all_jobs

def ComputeCumulativeRewards(all_jobs):
    """Compute cumulative rewards for each job"""
    for job in all_jobs:
        if job.shipped == True or job.due_date == 0:
            T = len(job.stored_gradients)
            job.cumulative_rewards = [0] * (T)
            job.cumulative_rewards[T - 1] = job.rewards[T - 1] 
    
            if T > 1:
                for t in range(T - 1, 0, -1):
                    job.cumulative_rewards[t - 1] = \
                    job.cumulative_rewards[t] + job.rewards[t - 1]  

def StateTransition(all_jobs, stored_jobs):
    """Execute time step"""   
    ComputeCumulativeRewards(all_jobs)
       
    deleteList = []

    # Remove all jobs that are shipped or have reached due date 0
    for job in all_jobs:
        if job.shipped or job.due_date == 0:
            deleteList.append(job)
            stored_jobs.append(job)
        if not job.shipped and job.due_date > 0:
            job.due_date -= 1

    if len(deleteList) > 0:
        for job in deleteList:
            all_jobs.remove(job)
            
def ComputeGenericFeatures(all_jobs, transport_capacity):
   """Compute generic futures for all shared information"""  
   generic_features = Features(0, 0, 0, 0, 0, 0, 0, 0)
    
   # Initialize generic features
   generic_features.average_distance = 0
   generic_features.total_volume = 0
   generic_features.average_due_date = 0 
   generic_features.number_of_jobs = 0
    
   for job in all_jobs: 
       if job.sharing_information == True:
           generic_features.total_volume += job.volume
           generic_features.average_due_date += job.due_date
           generic_features.average_distance += job.distance
           generic_features.number_of_jobs += 1
        
           generic_features.average_due_date = \
           generic_features.average_due_date / generic_features.number_of_jobs
           generic_features.average_distance = \
           generic_features.average_distance / generic_features.number_of_jobs
        
   return generic_features
      
def KnapsackCarrier(all_jobs, instance):
    """Carrier fills transport capacity by solving 0-1 knapsack problem"""
    number_of_jobs = 0
    job_volumes = []
    job_values = []
    transport_capacity = round(instance.transport_capacity)
     
    for job in all_jobs:
        number_of_jobs += 1
        job_volumes.append(job.volume)
        job_values.append(job.net_value)

    grid = [[0] * (transport_capacity + 1)]
    for item in range(number_of_jobs):
        grid.append(grid[item].copy())
        for k in range(job_volumes[item], transport_capacity + 1):
            grid[item + 1][k] = \
            max(grid[item][k], grid[item][k - job_volumes[item]] + job_values[item])

    solution_weight = 0
    solution_costs = 0
    solution_bids = 0
    solution_volume = 0
    taken = []
    k = transport_capacity
    jobs_action = []
        
    for item in range(number_of_jobs, 0, -1):
        if grid[item][k] != grid[item - 1][k]:
            taken.append(item - 1)
            k -= job_volumes[item - 1]
            solution_weight += job_volumes[item - 1]
            
            all_jobs[item - 1].shipped = True
            jobs_action.append(all_jobs[item - 1])
            
            solution_bids += all_jobs[item - 1].bid
            solution_costs += all_jobs[item - 1].bid - \
                              all_jobs[item - 1].net_value
            solution_volume += all_jobs[item - 1].volume

def NormalizeRewards(instance, stored_jobs):
    """Normalize rewards for update procedure"""    
    # Initialize
    average_cumulative_reward = [0] * (instance.max_due_date + 1)
    st_dev_rewards = [0] * (instance.max_due_date + 1)
    number_of_jobs = [0] * (instance.max_due_date + 1)
    
    # Compute total rewards
    for job in stored_jobs:
        T = len(job.stored_gradients)
        for t in range(T): 
            help_time = job.ori_due_date - t
            average_cumulative_reward[help_time] += \
            job.cumulative_rewards[t]
            number_of_jobs[help_time] += 1
    
    # Compute average reward
    for due_date in range(instance.max_due_date + 1): 
        if number_of_jobs[due_date] > 0:
            average_cumulative_reward[due_date] = \
            average_cumulative_reward[due_date] / number_of_jobs[due_date]
     
    # Compute standard deviation 
    for job in stored_jobs:
        T = len(job.stored_gradients)
        for t in range(T): 
            help_time = job.ori_due_date - t
            st_dev_rewards[help_time] += \
            (average_cumulative_reward[help_time] - \
             job.cumulative_rewards[t])**2            

    for due_date in range(instance.max_due_date + 1): 
        if number_of_jobs[due_date] > 0:               
            st_dev_rewards[due_date] = \
            1 / number_of_jobs[due_date] * np.sqrt(st_dev_rewards[due_date])
            if st_dev_rewards[due_date] == 0:
                st_dev_rewards[due_date] = 1.0    
    
    return average_cumulative_reward, st_dev_rewards

def RunEpisode(instance, all_jobs, feature_weights, dynamic_st_dev, training): 
    """Run an episode until the end of time horizon"""    
    all_jobs = []
    stored_jobs = []
    average_rewards = [0] * (instance.max_due_date + 1)
    help_number_of_jobs = [0] * (instance.max_due_date + 1)
    
    if training:
        time_horizon = instance.training_time_horizon
    else:
        time_horizon = instance.validation_time_horizon    
    
    for t in range(time_horizon + 1):
        
         # Generate new jobs (also initial state) 
        GenerateJobs(instance, all_jobs, instance.sharing_percentage)
        
        generic_features = ComputeGenericFeatures(all_jobs, \
                                                 instance.transport_capacity) 
        
        # All jobs place stochastic bid
        for job in all_jobs:

            # Initialize job features
            feature_array = [0, 0, 0, 0, 0, 0, 0, 0]
            
            feature_array[0] = 1
            
            # Only jobs sharing information may use features
            if job.sharing_information == True:
                feature_array[1] = generic_features.total_volume / \
                (instance.max_number_jobs * instance.max_due_date * 10)
                feature_array[2] = generic_features.average_due_date /instance.max_due_date
                feature_array[3] = generic_features.average_distance / instance.max_distance
                feature_array[4] = generic_features.number_of_jobs / \
                (instance.max_number_jobs * instance.max_due_date) 
               
            # Job properties as futures 
            feature_array[5] = job.volume / 10
            feature_array[6] = job.due_date / instance.max_due_date
            feature_array[7] = job.distance / instance.max_distance
            
            
            number_of_features = len(feature_array)
            
            # Multiply features and weight
            mu_state = np.dot(feature_array, feature_weights, out = None)
            
            job.stored_mu.append(mu_state)
                      
            #Generate random probability between 0 and 1
            random_probability = np.random.uniform(0, 1)
                  
            # Sample bid using inverse normal distribution
            job.bid  = \
            sc.norm.ppf(random_probability, loc = mu_state, scale = dynamic_st_dev)

            job.net_value = job.bid - \
                    instance.costs_per_mile * job.volume * job.distance
            
            # Compute and store gradient
            gradient = []
            for f in range(number_of_features):
                gradient.append(((job.bid - mu_state) * feature_array[f]) / \
                                (dynamic_st_dev**2))
            
            job.stored_gradients.append(gradient)
            job.stored_actions.append(job.bid)

        # Solve knapsack problem for carrier
        KnapsackCarrier(all_jobs, instance)
  
        # Compute job rewards
        penalty_cost = 0
        holding_cost = 0
        for job in all_jobs:
            if job.shipped:
                job.rewards.append(- job.bid)
            elif job.due_date > 0 and not job.shipped:
                holding_cost = - (instance.holding_cost * job.volume)
                job.rewards.append(holding_cost)
            elif job.due_date == 0 and not job.shipped: 
                penalty_cost = - (instance.penalty_failed * job.volume)
                job.rewards.append(penalty_cost)

            help_number_of_jobs[job.due_date] += 1 
            average_rewards[job.due_date] += job.rewards[-1] 
            
        # Take time step
        StateTransition(all_jobs, stored_jobs)
        
    for i in range(len(average_rewards)):
        if (help_number_of_jobs[i] > 0):
            average_rewards[i] = average_rewards[i] / help_number_of_jobs[i]

    return stored_jobs

def Train(seed):

    # Initialize instance
    instance = SetInstanceProperties()
    
    # Generate initial state
    all_jobs = []
    all_jobs = GenerateJobs(instance, all_jobs, instance.sharing_percentage)
    
    # Set initial standard deviation
    dynamic_st_dev = instance.initial_st_dev

    feature_weights = []
    for i in range(0, 8):
         feature_weights.append(0)
         
    feature_weights[0] = 0

    # Train algorithm until number of training episodes has been completed
    for i in range(instance.number_of_training_episodes + 1):
        
        if np.mod(i, 100) == 0:
            print("Training episode: ", i, " / ", \
                  instance.number_of_training_episodes)
        
        stored_jobs = []

        # Run a single episode
        stored_jobs = \
        RunEpisode(instance, all_jobs, feature_weights, dynamic_st_dev, True)
        
        # Update policy
        number_of_jobs = len(stored_jobs)
        
        average_cumulative_reward, st_dev_rewards = \
                NormalizeRewards(instance, stored_jobs)
        
        for job in stored_jobs:       
            T = len(job.stored_gradients)
            for t in range(T):
                help_time = job.ori_due_date - t
                normalized_reward = 0
                normalized_reward = \
                (job.cumulative_rewards[t] - average_cumulative_reward[help_time]) / \
                st_dev_rewards[help_time]
                    
                for f in range(len(feature_weights)):
                    weight_update = 1 / number_of_jobs * \
                    (instance.alpha_mu * dynamic_st_dev**2 *  \
                    job.stored_gradients[t][f] * \
                    normalized_reward)
                   
                    feature_weights[f] = feature_weights[f] + weight_update               
                    
                # Update standard deviation
                stdev_update = 1 / number_of_jobs * \
                    instance.alpha_sigma * dynamic_st_dev**2 * normalized_reward * \
                    (((job.stored_actions[t] - job.stored_mu[t])**2 - dynamic_st_dev**2) \
                     / dynamic_st_dev**3) 
                dynamic_st_dev  = dynamic_st_dev + stdev_update

        if np.mod(i, 1000) == 0: 
            average_bidding_costs = 0
            for j in range(instance.number_of_validation_episodes):
                stored_jobs = []

                # Run episode
                stored_jobs = \
                    RunEpisode(instance, all_jobs, feature_weights, dynamic_st_dev, False)

                number_of_jobs = len(stored_jobs)
        
                average_cumulative_reward, st_dev_rewards = \
                    NormalizeRewards(instance, stored_jobs)
                
                
                T = (instance.max_due_date + 1)
                for t in range(T):        
                    average_bidding_costs += average_cumulative_reward[t] / \
                                    (T * instance.number_of_validation_episodes)
                   
            # Store average rewards
            print('Validation episode: ',i,' Average bidding costs: ', average_bidding_costs)

# Set seed value
global_seed = 0
np.random.seed(global_seed)

# Start training
Train(seed = global_seed)
