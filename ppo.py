import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_clip, K_epochs):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.tensor(memory.states, dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.float32)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
