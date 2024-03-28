from utils.agent.utils import get_config, generate_rollouts

import argparse
import ray  # type: ignore
from ray import tune  # type: ignore
from ray.rllib.utils.framework import try_import_tf  # type: ignore


_, tf, _ = try_import_tf()
tfkl = tf.keras.layers

"""
Types:
    "square" : square grid
    "rectangle" : rectangle grid
    "rectangle_factorized" : rectangle grid with factorized action space
    "rectangle_pin" : rectangle grid with pin
    "rectangle_pin_attn_component" : rectangle grid with pin and attention on component
    "rectangle_pin_attn_all" : rectangle grid with pin and attention on component and pin
    "rectangle_pin_all_attn_factorized" : rectangle grid with pin and attention on component and pin, factorized action space
    "rectangle_pin_attn_all_no_grid" : rectangle grid with pin and attention on component and pin, no grid
    "rectangle_factorized_pin" : rectangle grid with pin with factorized action space
    "rectangle_spatial_pin" : rectangle grid with pin with spatial model inputs
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="square",
    )
    args = parser.parse_args()
    config = get_config(args.type)

    ray.init(local_mode=True)
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 1},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        keep_checkpoints_num=5,
        restore=None,
    )

    if (
        args.type != "square"
        and args.type != "rectangle"
        and args.type != "rectangle_factorized"
    ):
        generate_rollouts(config, args.type)
