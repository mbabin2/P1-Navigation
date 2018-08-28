import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import OrderedDict, namedtuple

from SumTree import SumTree

BUFFER_SIZE = int(1.5e5) # replay buffer size
BATCH_SIZE = 64 # minibatch size
GAMMA = 0.999 # discount rate
TAU = 1e-3 # used for soft update of target-NN weights
PRI_A = 0.95 # priority experience replay (PER) coefficient 'a'
PRI_EPSILON = 1e-2 # PER coefficient epsilon 
LR_ALPHA = 5e-4 # learning rate
N = 5 # update the networks every N steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    #
    # buildModel takes in the state and action size of an environment and creates a new NN with 128 neurons in hidden layer 1
    # and 64 in hidden layer 2, with random weights determined by 'seed'
    #
    def buildModel(self, state_size, action_size, seed):
        self.seed = torch.manual_seed(seed) # initialize weights with seed 'seed'
        
        input_size = state_size # input layer size
        hidden_size = [128, 64] # hidden layer sizes
        output_size = action_size # out layer size
        
        # build NN
        model = nn.Sequential(OrderedDict([
        ('input', nn.Linear(input_size, hidden_size[0])),
        ('relu1', nn.ReLU()),
        ('h1', nn.Linear(hidden_size[0], hidden_size[1])),
        ('relu2', nn.ReLU()),
        ('output', nn.Linear(hidden_size[1], output_size))
        ]))
        return model
    
    #
    # intializes a new Q-learning Agent
    #
    def __init__(self, state_size, action_size, seed):
        
        # Spaces:
        self.state_size = state_size # size of the environment's state space
        self.action_size = action_size # size of the environment's action space

        # Q-Networks:
        self.qnetwork_local = self.buildModel(state_size, action_size, seed).to(device) # build local NN
        self.qnetwork_target = self.buildModel(state_size, action_size, seed).to(device) # build target NN
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR_ALPHA) # define optimizer
        
        # Replay Buffer:
        self.replay_buffer_Tree = SumTree(BUFFER_SIZE) # define replay buffer
        self.buffer_size = 0 # counter for the size of the buffer
        self.buffer_exp = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"]) # definition an experience to be stored in the buffer
        
        #Misc:
        self.t = 0
    
    #
    # step takes in a SARS' tuple in order to store it into the replay buffer, and train the networks every N steps
    #
    def step(self, state, action, reward, next_state):
        
        new_exp = self.buffer_exp(state, action, reward, next_state) # add new experience to buffer
        
        # calculate new priority value to store with experience:
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        
        with torch.no_grad():
            Q_local = self.qnetwork_local(torch.from_numpy(state).float().to(device)) # local Q-value 
            Q_target = max(self.qnetwork_target(torch.from_numpy(next_state).float().to(device))) # target Q-value
        
        p = (np.abs(Q_local[action] - reward + (GAMMA * Q_target))+PRI_EPSILON)**PRI_A # priority value for experience
        
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        
        self.replay_buffer_Tree.add(p, new_exp) # add experience to buffer using priority p
        self.buffer_size = min(BUFFER_SIZE, self.buffer_size+1) # update buffer size counter
       
        # check if N timesteps have occurred:
        self.t = (self.t + 1) % N
        if self.t == 0:
            if self.buffer_size > BATCH_SIZE: # check if enough samples exist in the replay buffer
                
                experiences = [self.replay_buffer_Tree.get(s) for s in np.random.uniform(0,self.replay_buffer_Tree.total(),BATCH_SIZE)] # sample a batch of experiences from the buffer using PER
                
                # create a batch of SARS' tuples from the experiences sampled:
                states = torch.from_numpy(np.vstack([e[2].state for e in experiences if e is not None])).float().to(device)
                actions = torch.from_numpy(np.vstack([e[2].action for e in experiences if e is not None])).long().to(device)
                rewards = torch.from_numpy(np.vstack([e[2].reward for e in experiences if e is not None])).float().to(device)
                next_states = torch.from_numpy(np.vstack([e[2].next_state for e in experiences if e is not None])).float().to(device)
                
                #DDQN:
                self.qnetwork_local.eval()
                self.qnetwork_target.eval()
                with torch.no_grad():
                     # determine the best action to take in the next state
                    eval_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                    
                    # calculate the target Q-value for the next state taking that action
                    Q_targets_next = self.qnetwork_target(next_states).gather(1, eval_actions)
                self.qnetwork_local.train()
                self.qnetwork_target.train()
                
                # Compute Loss:
                Q_targets = rewards + (GAMMA * Q_targets_next) #TD targets

                Q_expected = self.qnetwork_local(states).gather(1, actions) # expected Q-values

                loss = F.mse_loss(Q_expected, Q_targets) #Calculate loss using 'Mean Square Error'
                
                # Minimize the loss:
                self.optimizer.zero_grad() # reset gradients
                loss.backward() # calculate gradients
                self.optimizer.step() # optimize weights
                
                # soft update target network: 
                for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                    target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
        
    #
    # Determines what action to take via an epsilon-greedy policy
    #
    def act(self, state, epsilon):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #process state as tensor
        
        #obtain action-values from local network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # select action based on epsilon greedy policy
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) # greedy action
        else:
            return random.choice(np.arange(self.action_size)) # random action