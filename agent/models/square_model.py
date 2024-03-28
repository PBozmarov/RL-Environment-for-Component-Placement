"""
Policy model for the square component placement task.
"""

from agent.models.model_building_blocks import ConvBlocks
from ray.rllib.models.tf.tf_modelv2 import TFModelV2  # type: ignore
from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class SquareModel(TFModelV2):
    """A model to predict the action logits and value for a square
    component placement task.

    This model takes a 2D grid as input and learns an encoding of the grid using
    a convolutional neural network with:

        - Convolutional layers
        - Batch normalization
        - ReLU activations
        - Optional max pooling

    The encoding is flattened and passed through two separate dense layers to
    generate action logits and a value prediction. The dimensions of
    the encoding depends on the specified convolutional layers.

    Attributes:
        action_space (gym.spaces): Action space of the environment.
        grid_input (tensor): Input tensor for the grid.
        encoding_grid (tensor): Encoded grid.
        logits_model (tf.keras.Model): Logits model.
        value_model (tf.keras.Model): Value model.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initializes an instance of the SquareModel class.

        Args:
            obs_space (gym.Space): Observation space of the environment.
            action_space (gym.Space): Action space of the environment.
            num_outputs (int): Number of output neurons.
            model_config (Dict): Model configuration dictionary.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, None, model_config, name)
        self.action_space = action_space
        self.grid_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["height"],
                model_config["custom_model_config"]["width"],
            ),
            dtype=tf.float32,
            name="grid_input",
        )

        self.build_encoding_square(model_config)
        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_square()
            self.build_value_model_square()

    def build_encoding_square(self, model_config):
        """Build a convolutional neural network to encode the input grid.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        self.encode_grid(model_config)

    def build_logits_model_square(self):
        """Build a model to calculate the logits.

        Use a dense layer from the encoded grid to calculate the logits,
        where the number of output neurons is equal to size of the action space.
        """
        logits = tfkl.Dense(self.action_space.n)(self.encoding_grid)
        self.logits_model = tf.keras.Model(self.grid_input, logits, name="logits_model")

    def build_value_model_square(self):
        """Build a model to calculate the value function.

        Use a dense layer from the encoded grid to calculate the value function,
        where the number of output neurons is equal to 1 (the value of the state).
        """
        value = tfkl.Dense(1)(self.encoding_grid)
        self.value_model = tf.keras.Model(self.grid_input, value, name="value_model")

    def encode_grid(self, model_config):
        """Encode the grid using a convolutional neural network.

        Process the grid using the specified number of convolutional blocks,
        where each block consists of a convolutional layer, an activation layer,
        and an optimal max pooling layer.

        The encoded grid is flattened and used as input to the logits model. The dimensions
        of the grid are specified in the model configuration dictionary (height and width).
        The dimensions of the encoded grid depend on the specific configuration of the
        ConvBlocks class.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        processed_grid = ConvBlocks(
            num_conv_blocks=model_config["custom_model_config"]["num_conv_blocks"],
            num_conv_filters=model_config["custom_model_config"]["num_conv_filters"],
            conv_kernel_size=model_config["custom_model_config"]["conv_kernel_size"],
            activation=model_config["custom_model_config"]["activation"],
            max_pool=model_config["custom_model_config"]["max_pool"],
            max_pool_kernel_size=model_config["custom_model_config"][
                "max_pool_kernel_size"
            ],
        )(self.grid_input)

        self.encoding_grid = tfkl.Flatten()(processed_grid)

    def forward(self, input_dict, state, seq_lens):
        """Perform forward pass through the model.

        Args:
            input_dict (Dict): Input dictionary containing "obs" and "action_mask" tensors.
            state: Model state tensor.
            seq_lens: Sequence length tensor.

        Returns:
            Logits and model state tensors.

        Note:
            An action mask is applied to the logits to prevent the agent from
                selecting invalid actions.
        """
        logits = self.logits_model(input_dict["obs"]["grid"])
        self._value_out = self.value_model(input_dict["obs"]["grid"])

        logits += tf.maximum(
            tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min
        )

        return logits, state

    def value_function(self):
        """Calculate the value function of the model.

        Returns:
            Tensor containing the value function.
        """
        return tf.reshape(self._value_out, [-1])
