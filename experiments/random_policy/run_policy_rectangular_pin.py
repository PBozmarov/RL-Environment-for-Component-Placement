import numpy as np
import matplotlib.pyplot as plt
import argparse

from agent.random.random_policy_rectangular import simulate, random_policy
from environment.dummy_env_rectangular_pin import DummyPlacementEnv


def plot_episode_returns(
    episode_returns: list,
    env_height: int,
    env_width: int,
    env_components_complexity: int,
    env_nets_complexity: int,
    env_pins_complexity: int,
    env_net_distribution: int,
    env_pin_spread: int,
    env_min_component_w: int,
    env_max_component_w: int,
    env_min_componewnt_h: int,
    env_max_component_h: int,
    env_max_num_components: int,
    env_min_num_components: int,
    env_min_num_nets: int,
    env_max_num_nets: int,
    env_max_num_pins_per_net: int,
    env_min_num_pins_per_net: int,
    env_reward_type: str,
    env_reward_beam_width: int,
    env_weight_wirelength: float,
    policy_type: str,
):
    """Plot episode returns for a given policy.

    Args:
        episode_returns (list): List of episode returns.
        env_height (int): Environment height.
        env_width (int): Environment width.
        env_components_complexity (int): Environment components complexity.
        env_nets_complexity (int): Environment nets complexity.
        env_pins_complexity (int): Environment pins complexity.
        env_net_distribution (int): Environment net distribution.
        env_pin_spread (int): Environment pin spread.
        env_min_component_w (int): Environment minimum component width.
        env_max_component_w (int): Environment maximum component width.
        env_min_component_h (int): Environment minimum component height.
        env_max_component_h (int): Environment maximum component height.
        env_max_num_components (int): Environment maximum number of components.
        env_min_num_components (int): Environment minimum number of components.
        env_min_num_nets (int): Environment minimum number of nets.
        env_max_num_nets (int): Environment maximum number of nets.
        env_max_num_pins_per_net (int): Environment maximum number of pins per net.
        env_min_num_pins_per_net (int): Environment minimum number of pins per net.
        env_reward_type (str): Environment reward type.
        env_reward_beam_width (int): Environment reward beam width.
        env_weight_wirelength (float): Environment weight wirelength.
        policy_type (str): Policy type.
    """

    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title(
        f"""{policy_type.capitalize()} policy episode returns \n (env: {env_height}x{env_width},
            component min: {env_min_component_h}x{env_min_component_w}, \n component max: {env_max_component_h}x{env_max_component_w},
            max number components: {env_max_num_components}, min number components: {env_min_num_components})
            components complexity: {env_components_complexity}, nets complexity: {env_nets_complexity},
            pins complexity: {env_pins_complexity}, net distribution: {env_net_distribution},
            pin spread: {env_pin_spread}, min number nets: {env_min_num_nets}, max number nets: {env_max_num_nets},
            max number pins per net: {env_max_num_pins_per_net}, min number pins per net: {env_min_num_pins_per_net},
            reward type: {env_reward_type}, reward beam width: {env_reward_beam_width},
            weight wirelength: {env_weight_wirelength})"""
    )
    plt.savefig(
        f"experiments/results/pin_rect_env_{policy_type}_policy_episode_returns.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_type", type=str, default="random")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--env_height", type=int, default=10)
    parser.add_argument("--env_width", type=int, default=10)
    parser.add_argument("--env_components_complexity", type=int, default=5)
    parser.add_argument("--env_nets_complexity", type=int, default=5)
    parser.add_argument("--env_pins_complexity", type=int, default=5)
    parser.add_argument("--env_net_distribution", type=int, default=5)
    parser.add_argument("--env_pin_spread", type=int, default=5)
    parser.add_argument("--env_min_component_w", type=int, default=2)
    parser.add_argument("--env_max_component_w", type=int, default=2)
    parser.add_argument("--env_min_component_h", type=int, default=4)
    parser.add_argument("--env_max_component_h", type=int, default=4)
    parser.add_argument("--env_max_num_components", type=int, default=4)
    parser.add_argument("--env_min_num_components", type=int, default=4)
    parser.add_argument("--env_min_num_nets", type=int, default=3)
    parser.add_argument("--env_max_num_nets", type=int, default=3)
    parser.add_argument("--env_max_num_pins_per_net", type=int, default=3)
    parser.add_argument("--env_min_num_pins_per_net", type=int, default=3)
    parser.add_argument("--env_reward_type", type=str, default="centroid")
    parser.add_argument("--env_reward_beam_width", type=int, default=2)
    parser.add_argument("--env_weight_wirelength", type=float, default=0.5)

    args = parser.parse_args()
    policy_type = args.policy_type
    num_episodes = args.num_episodes
    env_height = args.env_height
    env_width = args.env_width
    env_components_complexity = args.env_components_complexity
    env_nets_complexity = args.env_nets_complexity
    env_pins_complexity = args.env_pins_complexity
    env_net_distribution = args.env_net_distribution
    env_pin_spread = args.env_pin_spread
    env_min_component_w = args.env_min_component_w
    env_max_component_w = args.env_max_component_w
    env_min_component_h = args.env_min_component_h
    env_max_component_h = args.env_max_component_h
    env_max_num_components = args.env_max_num_components
    env_min_num_components = args.env_min_num_components
    env_min_num_nets = args.env_min_num_nets
    env_max_num_nets = args.env_max_num_nets
    env_max_num_pins_per_net = args.env_max_num_pins_per_net
    env_min_num_pins_per_net = args.env_min_num_pins_per_net
    env_reward_type = args.env_reward_type
    env_reward_beam_width = args.env_reward_beam_width
    env_weight_wirelength = args.env_weight_wirelength

    env = DummyPlacementEnv(
        env_height,
        env_width,
        env_components_complexity,
        env_nets_complexity,
        env_pins_complexity,
        env_net_distribution,
        env_pin_spread,
        env_min_component_w,
        env_max_component_w,
        env_min_component_h,
        env_max_component_h,
        env_max_num_components,
        env_min_num_components,
        env_min_num_nets,
        env_max_num_nets,
        env_max_num_pins_per_net,
        env_min_num_pins_per_net,
        env_reward_type,
        env_reward_beam_width,
        env_weight_wirelength,
    )
    episode_returns = simulate(env, random_policy, n_episodes=num_episodes)
    plot_episode_returns(
        episode_returns,
        env_height,
        env_width,
        env_components_complexity,
        env_nets_complexity,
        env_pins_complexity,
        env_net_distribution,
        env_pin_spread,
        env_min_component_w,
        env_max_component_w,
        env_min_component_h,
        env_max_component_h,
        env_max_num_components,
        env_min_num_components,
        env_min_num_nets,
        env_max_num_nets,
        env_max_num_pins_per_net,
        env_min_num_pins_per_net,
        env_reward_type,
        env_reward_beam_width,
        env_weight_wirelength,
        policy_type=policy_type,
    )
    print(
        f"""{policy_type.capitalize()} policy episode returns \n (env: {env_height}x{env_width},
            component min: {env_min_component_h}x{env_min_component_w}, \n component max: {env_max_component_h}x{env_max_component_w},
            max number components: {env_max_num_components}, min number components: {env_min_num_components})
            components complexity: {env_components_complexity}, nets complexity: {env_nets_complexity},
            pins complexity: {env_pins_complexity}, net distribution: {env_net_distribution},
            pin spread: {env_pin_spread}, min number nets: {env_min_num_nets}, max number nets: {env_max_num_nets},
            max number pins per net: {env_max_num_pins_per_net}, min number pins per net: {env_min_num_pins_per_net},
            reward type: {env_reward_type}, reward beam width: {env_reward_beam_width},
            weight wirelength: {env_weight_wirelength}): {np.mean(episode_returns)}"""
    )
