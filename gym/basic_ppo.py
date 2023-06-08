# define model
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=1.0)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.1)
        )

    def forward(self, state):
        action_logits = self.actor(state)
        state_values = self.critic(state)
        return Categorical(logits=action_logits), state_values


# define environment
env_name = "LunarLander-v2"
max_episode_steps = 1000
alpha_policy = 0.003
episodes = 5000
lam = 0.999
gamma = 0.99
env = gym.make(env_name, max_episode_steps=max_episode_steps, render_mode="human")
ppo_epochs = 4
ppo_clip = 0.2

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print("State Dim: ", state_dim, "Action Dim: ", action_dim)

policy = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=alpha_policy)

total_timesteps = 0
for episode in range(episodes):
    # get current state of env
    # decide on an action
    # make that action
    # collect new state and reward
    # store the data
    # repeat until not done with the episode

    states, rewards, dones, action_dists, actions, state_values = [], [], [], [], [], []

    state, _ = env.reset()
    done = False
    episode_steps = 0

    while not done and episode_steps < max_episode_steps:
        # this way of collecting data is technically not quite right, we need to continue where we left off from the last step otherwise you might never see some situations
        # however, works for most simple environments. The parallel_ppo does it correctly
        env.render()
        state = torch.tensor(state)
        with torch.no_grad():
            action_dist, state_value = policy(state)
        action = action_dist.sample()
        next_state, reward, done, truncated, info = env.step(action.item())

        states.append(state)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        action_dists.append(action_dist)
        state_values.append(state_value)

        state = next_state
        episode_steps += 1
        total_timesteps += 1

    states = torch.stack(states)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    state_values = torch.tensor(state_values, dtype=torch.float32)
    old_log_probs = torch.tensor([action_dist.log_prob(action) for action_dist, action in zip(action_dists, actions)],
                                 dtype=torch.float32)

    # GAE
    advantages = []
    gae = 0
    for i in reversed(range(len(states))):
        with torch.no_grad():
            next_value = state_values[i + 1] if i != len(states) - 1 else policy(torch.tensor(state))[1]
        delta = rewards[i] + gamma * (1 - dones[i]) * next_value - state_values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + state_values

    # PPO
    for _ in range(ppo_epochs):
        # for the states in the past, run the policy to get the new action dist
        new_log_actions, new_values = policy(states)
        new_log_probs = new_log_actions.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()

        obj1 = ratio * advantages
        obj2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
        policy_loss = -torch.min(obj1, obj2).mean()

        value_loss = F.mse_loss(new_values.squeeze(), returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * new_log_actions.entropy().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print average reward in new lines over the past 100 episodes
    if episode % 10 == 0:
        print("Episode: ", episode, "\nTimesteps: ", total_timesteps, "\nMean Reward: ", np.sum(rewards.numpy()),
              "\nPolicy Loss: ", policy_loss.item(), "\nValue Loss: ", value_loss.item(), "\nLoss: ", loss.item(),
              "\n\n")
