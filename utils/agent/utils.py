"""
Utility functions for agents.
"""

from environment.dummy_env_square import DummyPlacementEnv as DummyPlacementEnvSquare
from environment.dummy_env_rectangular import DummyPlacementEnv as DummyPlacementEnvRect
from environment.dummy_env_rectangular_pin import (
    DummyPlacementEnv as DummyPlacementEnvRectPin,
)
from environment.dummy_env_rectangular_pin_spatial import (
    DummyPlacementEnv as DummyPlacementEnvRectPinSpatial,
)

from utils.environment.env_wrappers import (
    FlatteningActionWrapperSquare,
    FlatteningActionMaskObservationWrapperSquare,
    FlatteningActionWrapperRect,
    FlatteningActionMaskObservationWrapperRect,
)

from utils.agent.factorized_action_distributions import (
    FactorisedActionDistributionOrientation,
    FactorisedActionDistributionCoordinates,
)
from utils.agent.callbacks import CustomCallbackClass

from utils.visualization.csv_utils import save_config_to_csv, save_to_file

from agent.models.square_model import SquareModel
from agent.models.rectangle_model import RectangleModel
from agent.models.rectangle_model_factorized import RectangleFactorizedModel
from agent.models.rectangle_pin_model import RectanglePinModel
from agent.models.rectangle_pin_attn_component_model import RectanglePinAttnCompModel
from agent.models.rectangle_pin_attn_component_pin_model import (
    RectanglePinAttnCompPinModel,
)
from agent.models.rectangle_pin_factorized_model import RectanglePinFactorizedModel
from agent.models.rectangle_pin_attn_all_model_no_grid import (
    RectanglePinAttnAllNoGridModel,
)
from agent.models.rectangle_pin_all_attn_factorized import (
    RectanglePinAllAttnFactorized,
)
from agent.models.rectangle_pin_spatial_model import RectanglePinSpatialModel
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig  # type: ignore
from ray.rllib.algorithms.ppo import PPO  # type: ignore

import json
import ray  # type: ignore
import os
import glob
from pathlib import Path
from ray import tune  # type: ignore
from ray.rllib.models import ModelCatalog  # type: ignore
from ray.rllib.utils.framework import try_import_tf  # type: ignore
from ray.rllib.algorithms.ppo.ppo import PPOConfig  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


model_dict = {
    "square": SquareModel,
    "rectangle": RectangleModel,
    "rectangle_factorized": RectangleFactorizedModel,
    "rectangle_pin": RectanglePinModel,
    "rectangle_pin_attn_component": RectanglePinAttnCompModel,
    "rectangle_pin_attn_all": RectanglePinAttnCompPinModel,
    "rectangle_factorized_pin": RectanglePinFactorizedModel,
    "rectangle_pin_all_attn_factorized": RectanglePinAllAttnFactorized,
    "rectangle_pin_attn_all_no_grid": RectanglePinAttnAllNoGridModel,
    "rectangle_spatial_pin": RectanglePinSpatialModel,
}

model_json_dict = {
    "square": "square_model.json",
    "rectangle": "rectangle_model.json",
    "rectangle_factorized": "rectangle_model_factorized.json",
    "rectangle_pin": "rectangle_pin_model.json",
    "rectangle_pin_attn_component": "rectangle_pin_attn_component_model.json",
    "rectangle_pin_attn_all": "rectangle_pin_attn_component_pin_model.json",
    "rectangle_factorized_pin": "rectangle_pin_factorized_model.json",
    "rectangle_pin_all_attn_factorized": "rectangle_pin_all_attn_factorized_model.json",
    "rectangle_pin_attn_all_no_grid": "rectangle_pin_attn_all_no_grid_model.json",
    "rectangle_spatial_pin": "rectangle_pin_spatial_model.json",
}


def read_json(file, config):
    """Adds model and env config to PPO config from json file.

    Args:
        file (str): Name of json file.
        config (PPOConfig): PPO config.

    Returns:
        config (PPOConfig): PPO config with model and env config.
    """
    with open("agent/config/" + file, "r") as f:
        config_json = json.load(f)
    config["model"] = config_json["model"]
    config["env_config"] = config_json["env_config"]
    return config


def get_activation(config):
    """Convert activation function from string to tensorflow function

    Args:
        config (PPOConfig): PPO config.

    Returns:
        config (PPOConfig): PPO config with activation function.
    """
    if (
        config["model"]["custom_model_config"]
        .keys()
        .__contains__("activation_component_grid")
    ):
        if (
            config["model"]["custom_model_config"]["activation_component_grid"]
            == "relu"
        ):
            config["model"]["custom_model_config"][
                "activation_component_grid"
            ] = tf.nn.relu
        elif (
            config["model"]["custom_model_config"]["activation_component_grid"]
            == "tanh"
        ):
            config["model"]["custom_model_config"][
                "activation_component_grid"
            ] = tf.nn.tanh
        elif (
            config["model"]["custom_model_config"]["activation_component_grid"]
            == "sigmoid"
        ):
            config["model"]["custom_model_config"][
                "activation_component_grid"
            ] = tf.nn.sigmoid
        else:
            raise ValueError("Invalid activation function")
    if config["model"]["custom_model_config"]["activation"] == "relu":
        config["model"]["custom_model_config"]["activation"] = tf.nn.relu
    elif config["model"]["custom_model_config"]["activation"] == "tanh":
        config["model"]["custom_model_config"]["activation"] = tf.nn.tanh
    elif config["model"]["custom_model_config"]["activation"] == "sigmoid":
        config["model"]["custom_model_config"]["activation"] = tf.nn.sigmoid
    else:
        raise ValueError("Invalid activation function")
    return config


def generate_rollouts(config, model_type):
    """
    Generates rollouts for the last checkpoint of a given run.

    Args:
        config (PPOConfig): PPO config.
        model_type (str): Type of model.
    """
    ray.shutdown()

    # sample rollout
    dir_path = str(Path.home()) + "/ray_results/PPO/"
    all_files = [
        f for f in glob.glob(os.path.join(dir_path, "*")) if not f.endswith(".json")
    ]
    all_files.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = all_files[0]
    logdir = most_recent_file

    # get latest checkpoint folder
    checkpoint_folders = [
        f for f in glob.glob(os.path.join(logdir, "*")) if os.path.isdir(f)
    ]
    checkpoint_folders.sort(key=os.path.getmtime, reverse=True)
    logdir_checkpoint = checkpoint_folders[0]

    components, actions = sample_rollout(config, logdir_checkpoint, model_type)
    save_to_file(components, actions, logdir)

    # save the model input parameters to a csv file
    df = save_config_to_csv(config)
    df.to_csv(f"{logdir}/{model_type}.csv", index=False)


def sample_rollout(  # noqa: max-complexity: 14
    config: PPOConfig, checkpoint_path: str, model_type: str, num_samples: int = 5
):
    """Sample num_samples rollouts using the trained model.

    Args:
        config (PPOConfig): Configuration for the environment and the model.
        checkpoint_path (str): Path to the checkpoint.
        model_type (str): Type of the model.
        num_samples (int, optional): Number of samples to take. Defaults to 5.

    Returns:
        Tuple[List[List[Component]], List[List[Tuple[int]]]]: List of components
            and list of sampled actions.
    """
    ray.init(local_mode=True)
    tune.register_env(model_type, create_env)
    ModelCatalog.register_custom_model(model_type, model_dict[model_type])
    if (
        model_type == "rectangle_factorized_pin"
        or model_type == "rectangle_pin_all_attn_factorized"
    ):
        if config["model"]["custom_model_config"]["factorization"] == "orientation":
            ModelCatalog.register_custom_action_dist(
                "factorised_action_dist", FactorisedActionDistributionOrientation
            )
        else:
            ModelCatalog.register_custom_action_dist(
                "factorised_action_dist", FactorisedActionDistributionCoordinates
            )
    agent = PPO(AlgorithmConfig.from_dict(config))
    agent.restore(checkpoint_path)

    env = init_env(config["env_config"])
    if (
        model_type == "rectangle_pin"
        or model_type == "rectangle_pin_attn_component"
        or model_type == "rectangle_pin_attn_all"
        or model_type == "rectangle_pin_attn_all_no_grid"
        or model_type == "rectangle_spatial_pin"
    ):
        flattening_action_wrapper = FlatteningActionWrapperRect(env)
        env = FlatteningActionMaskObservationWrapperRect(env)
        env = FlatteningActionWrapperRect(env)

    samples_components = []
    samples_actions = []

    for _ in range(num_samples):
        observation = env.reset()
        components = env.components
        done = False
        actions = []

        while not done:
            action = agent.compute_single_action(observation=observation, explore=False)
            observation, reward, done, info = env.step(action)
            if (
                model_type == "rectangle_pin"
                or model_type == "rectangle_pin_attn_component"
                or model_type == "rectangle_pin_attn_all"
                or model_type == "rectangle_pin_attn_all_no_grid"
                or model_type == "rectangle_spatial_pin"
            ):
                action = flattening_action_wrapper.action(action)
            actions.append(action)

        samples_components.append(components)
        samples_actions.append(actions)

    ray.shutdown()
    return samples_components, samples_actions


def get_config(model_type):
    """Gets the configuration for a specified model type.

    Args:
        model_type (str): The type of the model for which to get the configuration.

    Returns:
        dict: A dictionary containing the configuration for the specified model type.

    Raises:
        KeyError: If the specified model type is not found in the model_dict or
            model_json_dict dictionaries.
    """
    tune.register_env(model_type, create_env)
    ModelCatalog.register_custom_model(model_type, model_dict[model_type])

    if (
        model_type == "square"
        or model_type == "rectangle"
        or model_type == "rectangle_factorized"
    ):
        config = PPOConfig()

    if (
        model_type == "rectangle_pin"
        or model_type == "rectangle_pin_attn_component"
        or model_type == "rectangle_pin_attn_all"
        or model_type == "rectangle_pin_attn_all_no_grid"
        or model_type == "rectangle_spatial_pin"
        or model_type == "rectangle_factorized_pin"
        or model_type == "rectangle_pin_all_attn_factorized"
    ):
        config = PPOConfig().callbacks(CustomCallbackClass)

    config = read_json(model_json_dict[model_type], config)

    if (
        model_type == "rectangle_factorized_pin"
        or model_type == "rectangle_factorized"
        or model_type == "rectangle_pin_all_attn_factorized"
    ):
        if config["model"]["custom_model_config"]["factorization"] == "orientation":
            ModelCatalog.register_custom_action_dist(
                "factorised_action_dist", FactorisedActionDistributionOrientation
            )
        else:
            ModelCatalog.register_custom_action_dist(
                "factorised_action_dist", FactorisedActionDistributionCoordinates
            )

    config["env"] = model_type
    config["model"]["custom_model"] = model_type
    return get_activation(config)


def init_env(env_config: PPOConfig):
    """Initialize the environment using env_config.

    Args:
        env_config (PPOConfig): Environment configuration.

    Returns:
        DummyPlacementEnv: Environment object.
    """
    if env_config["type"] == "square":
        env = DummyPlacementEnvSquare(
            env_config["height"], env_config["width"], env_config["component_n"]
        )
    elif (
        env_config["type"] == "rectangle"
        or env_config["type"] == "rectangle_factorized"
    ):
        env = DummyPlacementEnvRect(
            env_config["height"],
            env_config["width"],
            env_config["min_component_w"],
            env_config["max_component_w"],
            env_config["min_component_h"],
            env_config["max_component_h"],
            env_config["max_num_components"],
            env_config["min_num_components"],
        )
    elif (
        env_config["type"] == "rectangle_pin"
        or env_config["type"] == "rectangle_factorized_pin"
        or env_config["type"] == "rectangle_pin_attn_component"
        or env_config["type"] == "rectangle_pin_attn_all"
        or env_config["type"] == "rectangle_pin_attn_all_no_grid"
        or env_config["type"] == "rectangle_pin_all_attn_factorized"
    ):
        env = DummyPlacementEnvRectPin(
            env_config["height"],
            env_config["width"],
            env_config["net_distribution"],
            env_config["pin_spread"],
            env_config["min_component_w"],
            env_config["max_component_w"],
            env_config["min_component_h"],
            env_config["max_component_h"],
            env_config["max_num_components"],
            env_config["min_num_components"],
            env_config["min_num_nets"],
            env_config["max_num_nets"],
            env_config["max_num_pins_per_net"],
            env_config["min_num_pins_per_net"],
            env_config["reward_type"],
            env_config["reward_beam_width"],
            env_config["weight_wirelength"],
        )
    elif env_config["type"] == "rectangle_spatial_pin":
        env = DummyPlacementEnvRectPinSpatial(
            env_config["height"],
            env_config["width"],
            env_config["net_distribution"],
            env_config["pin_spread"],
            env_config["min_component_w"],
            env_config["max_component_w"],
            env_config["min_component_h"],
            env_config["max_component_h"],
            env_config["max_num_components"],
            env_config["min_num_components"],
            env_config["min_num_nets"],
            env_config["max_num_nets"],
            env_config["max_num_pins_per_net"],
            env_config["min_num_pins_per_net"],
            env_config["reward_type"],
            env_config["reward_beam_width"],
            env_config["weight_wirelength"],
        )
    return env


def create_env(env_config: PPOConfig):
    """Create the environment using env_config, then flatten the action and action mask.

    Args:
        env_config (PPOConfig): Environment configuration.

    Returns:
        DummyPlacementEnv: Environment object.
    """
    env = init_env(env_config)
    if env_config["type"] == "square":
        env = FlatteningActionMaskObservationWrapperSquare(env)
        env = FlatteningActionWrapperSquare(env)
    elif (
        env_config["type"] == "rectangle"
        or env_config["type"] == "rectangle_pin"
        or env_config["type"] == "rectangle_pin_attn_component"
        or env_config["type"] == "rectangle_pin_attn_all"
        or env_config["type"] == "rectangle_pin_attn_all_no_grid"
        or env_config["type"] == "rectangle_spatial_pin"
    ):
        env = FlatteningActionMaskObservationWrapperRect(env)
        env = FlatteningActionWrapperRect(env)

    return env
