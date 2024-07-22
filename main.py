import argparse

from env import VanDeploymentEnv
from ppo import PPO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Specify the model name"
    )

    args = parser.parse_args()
    return args


def deploy_vans(ppo_agent, vans, goods):
    env = VanDeploymentEnv(vans, goods)
    state = env.reset()
    done = False
    deployments = []

    while not done:
        action, _ = ppo_agent.select_action(state)
        van_idx = action // len(vans)
        good_idx = action % len(vans)

        if env.remaining_vans[van_idx] != 0 and env.remaining_goods[good_idx] != 0:
            deployments.append((van_idx, good_idx))

        state, _, done, _ = env.step(action)

    return deployments


def main():
    args = parse_arguments()
    model_name = args.model

    # Define the state and action dimensions based on your environment
    state_dim = 6  # Assuming 3 vans and 3 goods (3 + 3 = 6)
    action_dim = 9  # Assuming 3 vans and 3 goods (3 * 3 = 9 possible assignments)

    # Create a new PPO agent
    ppo_agent = PPO(
        state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4
    )

    # Load the trained model
    ppo_agent.load(model_name)

    # Example usage
    vans = [10, 20, 100]
    goods = [5, 15, 80]

    deployments = deploy_vans(ppo_agent, vans, goods)

    print("Van deployments:")
    for van_idx, good_idx in deployments:
        print(
            f"Van with capacity {vans[van_idx]} tons assigned to good weighing {goods[good_idx]} tons"
        )


if __name__ == "__main__":
    main()
