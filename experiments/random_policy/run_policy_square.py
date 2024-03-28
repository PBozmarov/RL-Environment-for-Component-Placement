import numpy as np
import matplotlib.pyplot as plt
import argparse

from agent.random.random_policy_square import simulate, random_policy
from environment.dummy_env_square import DummyPlacementEnv


def plot_episode_returns(
    episode_returns: list,
    env_height: int,
    env_width: int,
    component_size: int,
    policy_type: str,
):
    """Plot episode returns for a given policy.

    Args:
        episode_returns (list): List of episode returns.
        env_height (int): Environment height.
        env_width (int): Environment width.
        component_size (int): Environment component size.
        policy_type (str): Policy type.
    """
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title(
        f"{policy_type.capitalize()} policy episode returns \n (env: {env_height}x{env_width}, component size: {component_size}x{component_size})"
    )
    plt.savefig(
        f"experiments/results/square_env_{policy_type}_policy_episode_returns.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", type=str, default="random")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--env_height", type=int, default=10)
    parser.add_argument("--env_width", type=int, default=10)
    parser.add_argument("--component_size", type=int, default=2)

    args = parser.parse_args()
    policy_type = args.policy_type
    num_episodes = args.num_episodes
    env_height = args.env_height
    env_width = args.env_width
    component_size = args.component_size

    env = DummyPlacementEnv(env_height, env_width, component_size)
    episode_returns = simulate(env, random_policy, n_episodes=num_episodes)
    plot_episode_returns(
        episode_returns, env_height, env_width, component_size, policy_type=policy_type
    )
    print(
        f"{policy_type.capitalize()} policy average return across {num_episodes} episodes (env: {env_height}x{env_width}, component size: {component_size}x{component_size}): {np.mean(episode_returns)}"
    )
