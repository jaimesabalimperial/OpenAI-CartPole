import gym
import copy
from gym.wrappers.monitoring.video_recorder import VideoRecorder  # records videos of episodes

import numpy as np
import matplotlib.pyplot as plt  # Graphical library
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configuring Pytorch
from collections import namedtuple, deque
from itertools import count
import math
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#dont change
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#dont change
class DQN(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputs, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden-1)])
        self.output_layer = nn.Linear(hidden_size, outputs)
    
    def forward(self, x):
        x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)


class FrameStacking(gym.Wrapper):
    """Return only every 4th frame"""
    def __init__(self, env, k):
        super(FrameStacking, self).__init__(env)
        self._obs_buffer = deque([], maxlen=k)
        self._skip       = k

    @property
    def _k_states(self):
        return torch.stack(list(self._obs_buffer)).reshape(-1).unsqueeze(0)
    
    def reset_buffer(self):
        self._obs_buffer = deque([], maxlen=self._skip)


class CartpoleAgent():
    def __init__(self, NUM_EPISODES, BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, 
                 NUM_FRAMES, SKIPPED_FRAMES, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS, REPLAY_BUFFER, LR,
                 EPS_START, EPS_END, REWARD_TARGET, EPS_DECAY=None, DDQN=False, eps_decay_strat="reward",
                 ablate_target=False, ablate_replay=False):

        #initialisation attributes
        self.reward_target = REWARD_TARGET
        self.reward_threshold = 0
        self.num_episodes = NUM_EPISODES
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.target_update_freq = TARGET_UPDATE_FREQ
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.k = NUM_FRAMES
        self.skip_frames = SKIPPED_FRAMES
        self.curr_episode = 0
        self.losses = []
        self.train_rewards = []
        self.eps_decay_strat = eps_decay_strat

        self.ablate_target = ablate_target
        self.ablate_replay = ablate_replay

        if self.eps_decay_strat[:3] == "exp":
            if EPS_DECAY == None:
                raise Exception("Have to define EPS_DECAY if decay strategy is exponential")
            else:
                self.eps_decay = EPS_DECAY

        self.epsilon = self.eps_start
        self.epsilon_list = []
        self.epsilon_delta = (self.epsilon - self.eps_end)/self.reward_target

        #Get number of states and actions from gym action space
        env = gym.make("CartPole-v1")
        env.reset()
        self.state_dim = len(env.state)    #x, x_dot, theta, theta_dot
        self.n_actions = env.action_space.n
        env.close()

        self.input_dim = int(self.state_dim*self.k) #define input size of DQN

        #define policy and target networks, as well as optimizer and replay buffer
        self.policy_net = DQN(self.input_dim, self.n_actions, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS).to(device)
        self.target_net = DQN(self.input_dim, self.n_actions, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.DDQN = DDQN
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(REPLAY_BUFFER)

                            


    def select_action(self, k_states):
        """"""
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(k_states).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
    

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: torch.sum(s[0][-self.k:]).absolute().item() > 0, batch.next_state)), device=device, dtype=torch.bool)
        
        # Can safely omit the condition below to check that not all states in the
        # sampled batch are terminal whenever the batch size is reasonable and
        # there is virtually no chance that all states in the sampled batch are 
        # terminal
        if sum(non_final_mask) > 0:
            non_final_next_states = torch.cat([s for s in batch.next_state if torch.sum(s[0][-self.k:]).absolute().item() > 0])
        else:
            non_final_next_states = torch.empty(0,self.state_dim).to(device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        
        with torch.no_grad():
            # Once again can omit the condition if batch size is large enough
            if sum(non_final_mask) > 0:
                if self.DDQN:
                    #DDQN ---> update next states Q values (using target net) using the actions that maximise the policy network
                    actions_non_final = torch.zeros_like(action_batch.view(non_final_mask.shape))
                    actions_non_final = torch.argmax(self.policy_net(non_final_next_states), 1).unsqueeze(1)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, actions_non_final).flatten()
                else:
                    if self.ablate_target:
                        next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0].detach()
                    else:
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

            else:
                next_state_values = torch.zeros_like(next_state_values)


        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss_func = nn.MSELoss()
        loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Limit magnitude of gradient for update step
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self):
        """"""
        env = gym.make("CartPole-v1")
        steps = 0
        stack_frames = FrameStacking(env, self.k) #define frame stacking framework
        decay_iter = 0

        for i_episode in tqdm.tqdm(range(self.num_episodes)):
            rewards = 0
            #if i_episode % 20 == 0:
                #print("episode ", i_episode, "/", self.num_episodes)

            env.reset() #reset environment
            state = torch.tensor(env.state).float().unsqueeze(0).to(device)
            for _ in range(self.k): # Added
                stack_frames._obs_buffer.append(state) # Added

            for t in count():
                k_states = stack_frames._k_states
                temp_rewards = 0
                for _ in range(self.skip_frames):
                    #process frames to pass as input to DQN
                    action = self.select_action(k_states) 
                    _, reward, done, _ = env.step(action.item()) #take step following epsilon-greedy policy

                    if not done:
                        k_states = stack_frames._k_states
                    else:
                        break

                    temp_rewards += reward
                
                rewards += temp_rewards
                reward = torch.tensor([temp_rewards], device=device)

                # Observe new state
                if not done:
                    next_state = torch.tensor(env.state).float().unsqueeze(0).to(device)
                else:
                    next_state = torch.zeros(1, self.state_dim)

                #store new state in frames buffer and process frames to pass as input to DQN
                stack_frames._obs_buffer.append(next_state) 
                next_k_states = stack_frames._k_states

                #Store the transition in memory
                self.memory.push(k_states, action, next_k_states, reward) 

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                if done:
                    break

                # Select and perform an action
                steps += 1

            stack_frames.reset_buffer() #reset frame stack memory

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            #epsilon decay strategy
            if self.eps_decay_strat == "reward":
                #implement reward-based decay
                if self.epsilon > self.eps_end and rewards > self.reward_threshold:
                    self.epsilon -= self.epsilon_delta
                    self.reward_threshold += 1
                    
            elif self.eps_decay_strat == "exp1":
                #epsilon starts decaying exponentially at start of training 
                self.epsilon = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-i_episode/self.eps_decay)
            elif self.eps_decay_strat == "exp2":
                #epsilon starts decaying exponentially after the 100th episode
                if i_episode > 100:
                    self.epsilon = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-2*decay_iter/self.eps_decay)
                    decay_iter += 1
            elif self.eps_decay_strat == "exp3":
                #epsilon starts decaying exponentially after the 300th episode
                if i_episode > 300:
                    self.epsilon = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-4*decay_iter/self.eps_decay)
                    decay_iter += 1
            elif self.eps_decay_strat == "exp4":
                #epsilon starts decaying exponentially after the 500th episode
                if i_episode > 500:
                    self.epsilon = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-8*decay_iter/self.eps_decay)
                    decay_iter += 1    
            else:
                pass

            self.epsilon_list.append(self.epsilon)
            self.train_rewards.append(rewards) #append episode reward to list

            self.curr_episode += 1

        print("Complete")
        env.close() #close environment


    def test(self):
        """run an episode with trained agent and record video"""

        env = gym.make("CartPole-v1")
        file_path = 'video.mp4'
        recorder = VideoRecorder(env, file_path)

        env.reset()
        done = False

        stack_frames = FrameStacking(env, self.k) #define frame stacking framework

        state = torch.tensor(env.state).float().unsqueeze(0).to(device)
        for _ in range(self.k): # Added
            stack_frames._obs_buffer.append(state) # Added

        duration = 0

        while not done:
            recorder.capture_frame()
            # Select and perform an action
            k_states = stack_frames._k_states
            action = self.select_action(k_states) 
            _, reward, done, _ = env.step(action.item()) #take step following epsilon-greedy policy
            duration += 1

            next_state = torch.tensor(env.state).float().unsqueeze(0).to(device)
            
            #store new state in frames buffer and process frames to pass as input to DQN
            stack_frames._obs_buffer.append(next_state) 

        env.close()
        recorder.close()

        print("Episode duration: ", duration)


    def plot_rewards(self):
        """Plot the total non-discounted sum of rewards across the episodes (i.e duration of each episode in steps)."""
        epochs = np.linspace(0, len(self.train_rewards), len(self.train_rewards))

        plt.figure(figsize=(10,7))
        plt.grid()
        plt.plot(epochs, self.train_rewards, label="Total Non-Discounted Rewards")
        #plt.axhline(y=self.reward_target, color='black', linestyle='-', label="Reward target")
        #plt.plot(epochs, rewards_list, "b.", markersize=3)
        #plt.plot(epochs, np.poly1d(np.polyfit(epochs, rewards_list, 1))(epochs), "r")
        plt.xlabel("Episodes")
        plt.ylabel("Total Non-Discounted Reward")
        plt.ylim((0,550))
        plt.axhline(y=500, color='red', linestyle='-', label="Maximum reward")
        plt.legend(loc="best")
        plt.show()
    
    def plot_losses(self):
        """Plot the total non-discounted sum of rewards across the episodes (i.e duration of each episode in steps)."""
        epochs = np.linspace(0, len(self.losses), len(self.losses))

        plt.figure(figsize=(10,7))
        plt.grid()
        plt.plot(epochs, self.losses, label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.legend(loc="upper left")
        plt.show()

    def plot_epsilon(self):
        """Plot the total non-discounted sum of rewards across the episodes (i.e duration of each episode in steps)."""
        epochs = np.linspace(0, len(self.epsilon_list), len(self.epsilon_list))

        plt.figure(figsize=(10,7))
        plt.grid()
        plt.plot(epochs, self.epsilon_list, color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Epsilon")
        plt.show()

def replicate_CPAgent(N_REPLICATIONS,NUM_EPISODES, BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, 
                      NUM_FRAMES, SKIPPED_FRAMES, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS, 
                      REPLAY_BUFFER, LR, EPS_START, EPS_END, REWARD_TARGET, return_epsilon=False):
    """"""
    replication_rewards = []
    replication_epsilons = []
    for i in range(N_REPLICATIONS):
        print(f"Replication {i+1}")
        CP_agent = CartpoleAgent(NUM_EPISODES, BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, 
                                 NUM_FRAMES, SKIPPED_FRAMES, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS, 
                                 REPLAY_BUFFER, LR, EPS_START, EPS_END, REWARD_TARGET, random=True)
    
        CP_agent.train() #train agent
        replication_rewards.append(CP_agent.train_rewards)
        if return_epsilon:
            replication_epsilons.append(CP_agent.epsilon_list)

    mean_rewards = np.mean(np.array(replication_rewards), axis=0)
    std_rewards = np.std(np.array(replication_rewards), axis=0)

    if return_epsilon:
        mean_epsilons = np.mean(np.array(replication_epsilons), axis=0)
        std_epsilons = np.std(np.array(replication_epsilons), axis=0)

        return mean_rewards, std_rewards, mean_epsilons, std_epsilons
    else:
        return mean_rewards, std_rewards


def plot_replications(mean_rewards, std_rewards, mean_epsilons=None, std_epsilons=None):
    """"""
    episodes = np.arange(len(mean_rewards))

    plt.figure(figsize=(10,7))
    plt.grid()
    plt.plot(episodes, mean_rewards, "r", label="mean / $\mu$")
    plt.fill_between(episodes, mean_rewards-std_rewards, mean_rewards+std_rewards, color="r", alpha=0.4, label=r"std / $\sigma$")
    plt.xlabel("Episode")
    plt.ylabel("Mean Total Non-Discounted Reward")
    plt.ylim((0,550))
    plt.axhline(y=500, color='black', linestyle='-', label="Maximum reward")
    plt.legend(loc="best")
    plt.show();

    if mean_epsilons is not None and std_epsilons is not None:
        plt.figure(figsize=(10,7))
        plt.grid()
        plt.plot(episodes, mean_epsilons, "b", label="mean / $\mu$")
        plt.fill_between(episodes, mean_epsilons-std_epsilons, mean_epsilons+std_epsilons, color="b", alpha=0.4, label=r"std / $\sigma$")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend(loc="best")
        plt.show();
        

if __name__ == "__main__":
    #define hyperparameters for agent
    NUM_EPISODES = 200
    BATCH_SIZE = 128
    GAMMA = 1.0
    TARGET_UPDATE_FREQ = 40
    NUM_FRAMES = 4
    SKIPPED_FRAMES = 4
    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 150
    REPLAY_BUFFER = 100000
    LR = 0.0001
    EPS_START = 1.0 
    EPS_END = 0.0005
    REWARD_TARGET = 100

    CP_agent = CartpoleAgent(NUM_EPISODES, BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ, 
                             NUM_FRAMES,SKIPPED_FRAMES, NUM_HIDDEN_LAYERS, SIZE_HIDDEN_LAYERS, 
                            REPLAY_BUFFER, LR, EPS_START, EPS_END, REWARD_TARGET)
    
    CP_agent.train()
    CP_agent.plot_rewards()

