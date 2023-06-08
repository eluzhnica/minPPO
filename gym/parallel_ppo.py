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


# all the envs with discrete action space and continous observations
env_name = "LunarLander-v2"
# env_name = 'CartPole-v1'
# env_name = 'MountainCar-v0'
# env_name = 'Acrobot-v1'
# env_name = 'Pendulum-v0'


alpha_policy = 0.0003
ppo_epochs = 10
ppo_clip = 0.2
gamma = 0.99
lam = 0.95
batch_size = 64
num_envs = 4
total_timesteps = 3000000
rollout_steps = 300

# envs = gym.vector.make(env_name, num_envs=num_envs, render_mode="human", max_episode_steps=5000)
envs = gym.vector.make(env_name, num_envs=num_envs, max_episode_steps=1000)
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n
print("State Dim: ", state_dim, "Action Dim: ", action_dim)

policy = ActorCritic(state_dim, action_dim)
state, info = envs.reset()
print(policy.parameters())
optimizer = optim.Adam(policy.parameters(), lr=alpha_policy)

timesteps = 0
for _ in range(total_timesteps // (rollout_steps * num_envs)):
    # storage
    states = torch.zeros((rollout_steps + 1, num_envs, state_dim), dtype=torch.float32)
    actions = torch.zeros((rollout_steps, num_envs), dtype=torch.int64)
    rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
    dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
    values = torch.zeros((rollout_steps + 1, num_envs), dtype=torch.float32)
    log_probs_old = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
    entropy = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
    advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32)
    returns = torch.zeros((rollout_steps + 1, num_envs), dtype=torch.float32)

    for i in range(rollout_steps):
        timesteps += 1
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_dist, state_value = policy(state)

        action = action_dist.sample()
        print("Action", action, action_dist)
        next_state, reward, done, truncated, info = envs.step(action.numpy())

        states[i] = torch.tensor(state, dtype=torch.float32)
        actions[i] = action
        rewards[i] = torch.tensor(reward, dtype=torch.float32)
        # dones is dones OR truncated
        dones[i] = torch.tensor(done, dtype=torch.int) | torch.tensor(truncated, dtype=torch.int)
        values[i] = state_value.squeeze()
        log_probs_old[i] = action_dist.log_prob(action)
        entropy[i] = action_dist.entropy()

        state = next_state

    with torch.no_grad():
        _, state_value = policy(torch.tensor(state, dtype=torch.float32))
        values[-1] = state_value.squeeze()

    # GAE
    last_adv = 0
    for i in reversed(range(rollout_steps)):
        next_value = values[i + 1]
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        advantages[i] = delta + gamma * lam * (1 - dones[i]) * last_adv
        last_adv = advantages[i]
    returns = advantages + values[:-1, :]

    # PPO
    states = states[:-1].view(-1, state_dim)
    actions = actions.view(-1)
    returns = returns.view(-1)
    advantages = advantages.view(-1)
    log_probs_old = log_probs_old.view(-1)
    entropy = entropy.view(-1)

    b_inds = np.arange(advantages.shape[0])
    np.random.shuffle(b_inds)
    for _ in range(ppo_epochs):
        for i in range(0, b_inds.shape[0], batch_size):
            ind = b_inds[i:i + batch_size]
            states_batch = states[ind]
            actions_batch = actions[ind]
            returns_batch = returns[ind]
            advantages_batch = advantages[ind]
            log_probs_old_batch = log_probs_old[ind]

            action_dist, state_value = policy(states_batch)
            new_log_probs = action_dist.log_prob(actions_batch)
            ratio = (new_log_probs - log_probs_old_batch).exp()
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages_batch

            value_loss = F.mse_loss(state_value.squeeze(), returns_batch)

            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = action_dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if timesteps % 1000 == 0:
        print("Timesteps ", timesteps * num_envs, "\nPolicy Loss: ", policy_loss.item(), "\nValue Loss: ",
              value_loss.item(), "\nEntropy Loss: ", entropy_loss.item(), "\nLoss: ", loss.item(),
              "\nMean Episodic Reward: ", np.sum(rewards.numpy()) / num_envs, "\n")
