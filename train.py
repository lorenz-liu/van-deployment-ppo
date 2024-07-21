from env import VanDeploymentEnv
from ppo import PPO, Memory


def train():
    vans = [10, 20, 100]
    goods = [5, 15, 80]
    env = VanDeploymentEnv(vans, goods)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    max_episodes = 10000
    max_timesteps = 100
    update_timestep = 2000
    lr = 0.002
    gamma = 0.99
    eps_clip = 0.2
    K_epochs = 4

    ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)
    memory = Memory()

    timestep = 0

    for i_episode in range(max_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            action, logprob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            if done:
                break

        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}, Average Reward: {sum(memory.rewards)/len(memory.rewards)}"
            )


if __name__ == "__main__":
    train()
