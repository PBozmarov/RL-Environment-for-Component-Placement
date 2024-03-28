import numpy as np
import matplotlib.pyplot as plt
import argparse

from agent.random.random_policy_rectangular import simulate, random_policy
from environment.dummy_env_rectangular import DummyPlacementEnv


def plot_episode_returns(
    episode_returns: list,
    env_height: int,
    env_width: int,
    env_min_component_w: int,
    env_max_component_w: int,
    env_min_component_h: int,
    env_max_component_h: int,
    env_max_num_components: int,
    env_min_num_components: int,
    policy_type: str,
):
    """Plot episode returns for a given policy.

    Args:
        episode_returns (list): List of episode returns.
        env_height (int): Environment height.
        env_width (int): Environment width.
        env_min_component_w (int): Environment minimum component width.
        env_max_component_w (int): Environment maximum component width.
        env_min_component_h (int): Environment minimum component height.
        env_max_component_h (int): Environment maximum component height.
        env_max_num_components (int): Environment maximum number of components.
        env_min_num_components (int): Environment minimum number of components.
        policy_type (str): Policy type.
    """

    plt.plot(episode_returns)

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title(
        f"{policy_type.capitalize()} policy episode returns \n (env: {env_height}x{env_width}, component min: {env_min_component_h}x{env_min_component_w}, \n component max: {env_max_component_h}x{env_max_component_w}, max number components: {env_max_num_components}, min number components: {env_min_num_components})"
    )
    plt.savefig(
        f"experiments/results/rect_env_{policy_type}_policy_episode_returns.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", type=str, default="random")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--env_height", type=int, default=10)
    parser.add_argument("--env_width", type=int, default=10)
    parser.add_argument("--env_min_component_w", type=int, default=2)
    parser.add_argument("--env_max_component_w", type=int, default=4)
    parser.add_argument("--env_min_component_h", type=int, default=2)
    parser.add_argument("--env_max_component_h", type=int, default=4)
    parser.add_argument("--env_max_num_components", type=int, default=20)
    parser.add_argument("--env_min_num_components", type=int, default=20)

    args = parser.parse_args()
    policy_type = args.policy_type
    num_episodes = args.num_episodes
    env_height = args.env_height
    env_width = args.env_width
    env_min_component_w = args.env_min_component_w
    env_max_component_w = args.env_max_component_w
    env_min_component_h = args.env_min_component_h
    env_max_component_h = args.env_max_component_h
    env_max_num_components = args.env_max_num_components
    env_min_num_components = args.env_min_num_components

    env = DummyPlacementEnv(
        env_height,
        env_width,
        env_min_component_w,
        env_max_component_w,
        env_min_component_h,
        env_max_component_h,
        env_max_num_components,
        env_min_num_components,
    )
    episode_returns = simulate(env, random_policy, n_episodes=num_episodes)
    plot_episode_returns(
        episode_returns,
        env_height,
        env_width,
        env_min_component_w,
        env_max_component_w,
        env_min_component_h,
        env_max_component_h,
        env_max_num_components,
        env_min_num_components,
        policy_type=policy_type,
    )
    print(
        f"{policy_type.capitalize()} policy average return across {num_episodes} episodes (env: {env_height}x{env_width}, component min: {env_min_component_h}x{env_min_component_w}, component max: {env_max_component_h}x{env_max_component_w}, max number components: {env_max_num_components}, min number components: {env_min_num_components}): {np.mean(episode_returns)}"
    )
