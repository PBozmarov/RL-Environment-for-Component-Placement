"""
Utility functions for saving training data to csv files.
"""

from environment.dummy_env_rectangular_pin import Component
from typing import List, Tuple
import pandas as pd
import pickle


def save_to_file(
    components: List[Component], actions: List[Tuple[int]], file_path: str
):
    """Save components and actions to the folder specified by file_path.

    Args:
        components (List[Component]): List of components.
        actions (List[Tuple[int]]): List of actions.
        file_path (str): Path to the folder where the files will be saved.
    """
    filenames = ["components", "actions"]
    data_objects = [components, actions]
    for filename, data in zip(filenames, data_objects):
        with open(file_path + f"/{filename}.pkl", "wb") as f:
            pickle.dump(data, f)


def save_config_to_csv(config):
    """Save the config to a csv file.

    Args:
        config (dict): The config dictionary.

    Returns:
        pd.DataFrame: The config dataframe.
    """
    df = pd.DataFrame(
        {
            "custom_model": [config["model"]["custom_model"]],
            "height": [config["model"]["custom_model_config"]["height"]],
            "width": [config["model"]["custom_model_config"]["width"]],
            "max_num_components": [
                config["model"]["custom_model_config"]["max_num_components"]
            ],
            "min_num_components": [
                config["model"]["custom_model_config"]["min_num_components"]
            ],
            "max_num_pins_per_component": [
                config["model"]["custom_model_config"]["max_num_pins_per_component"]
            ],
            "component_feature_vector_width": [
                config["model"]["custom_model_config"]["component_feature_vector_width"]
            ],
            "pin_feature_vector_width": [
                config["model"]["custom_model_config"]["pin_feature_vector_width"]
            ],
            "env": [config["env"]],
            "env_height": [config["env_config"]["height"]],
            "env_width": [config["env_config"]["width"]],
            "net_distribution": [config["env_config"]["net_distribution"]],
            "pin_spread": [config["env_config"]["pin_spread"]],
            "min_component_w": [config["env_config"]["min_component_w"]],
            "max_component_w": [config["env_config"]["max_component_w"]],
            "min_component_h": [config["env_config"]["min_component_h"]],
            "max_component_h": [config["env_config"]["max_component_h"]],
            "max_num_components_env": [config["env_config"]["max_num_components"]],
            "min_num_components_env": [config["env_config"]["min_num_components"]],
            "min_num_nets": [config["env_config"]["min_num_nets"]],
            "max_num_nets": [config["env_config"]["max_num_nets"]],
            "max_num_pins_per_net": [config["env_config"]["max_num_pins_per_net"]],
            "min_num_pins_per_net": [config["env_config"]["min_num_pins_per_net"]],
            "reward_type": [config["env_config"]["reward_type"]],
            "reward_beam_width": [config["env_config"]["reward_beam_width"]],
            "weight_wirelength": [config["env_config"]["weight_wirelength"]],
        }
    )
    return df
